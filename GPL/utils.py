import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn as nn
import dgl


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input):
        src, dst = graph.edges()
        input_size = input.size()
        # Ensure (i,j) and (j,i) has the same element in adjacency matrix
        oneside_input = input[src < dst]
        oneside_input_src = src[src<dst]
        oneside_input_dst = dst[src < dst]

        cat_dist = Categorical(logits=oneside_input)
        cat_sample = cat_dist.sample().unsqueeze(-1).float()
        neg_sample = 1-cat_sample

        output = torch.zeros(input_size)
        half_inputs = torch.cat([neg_sample, cat_sample], axis=-1)
        edge_id_oneside = graph.edge_id(oneside_input_src,oneside_input_dst)
        output[edge_id_oneside] = half_inputs

        edge_id_secondside = graph.edge_id(oneside_input_dst,oneside_input_src)
        output[edge_id_secondside] = half_inputs

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x, graph):
        x = STEFunction.apply(graph, x)
        return x

def onehot_from_logits(logits):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_tensor = (logits == logits.max(1, keepdim=True)[0]).float()
    return argmax_tensor

def sample_gumbel(graph, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """

    noises = []
    for unbatched_graph in dgl.unbatch(graph):
        num_node = unbatched_graph.number_of_nodes()
        src, dst = unbatched_graph.edges()
        a1 = torch.rand(*[num_node, num_node]).float()
        a2 = torch.rand(*[num_node, num_node]).float()
        tri_tensor1, tri_tensor2 = torch.tril(a1, -1) + torch.tril(a1, -1).permute(1, 0), \
                                   torch.tril(a2, -1) + torch.tril(a2, -1).permute(1, 0)
        noise = torch.cat([tri_tensor1[src, dst][:, None], tri_tensor2[src, dst][:, None]], dim=-1)
        noises.append(noise)

    U = torch.cat(noises, dim=0)
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, graph, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(graph, eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / tau, dim=-1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
# def gumbel_softmax_sample(logits, graph, temperature):
#     """ Draw a sample from the Gumbel-Softmax distribution"""
#     y = logits + sample_gumbel(graph).to(logits.device)
#     print(y.shape, logits, temperature)
#     exit()
#     return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, graph, temperature=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, graph, tau=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return F.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)

