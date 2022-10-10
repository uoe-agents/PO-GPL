import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from utils import gumbel_softmax

# File containing implementation of networks used in this research
class RFMBlock(nn.Module):
    def __init__(self, dim_in_node, dim_in_edge, dim_in_u, hidden_dim, hidden_dim2, dim_out):
        super(RFMBlock, self).__init__()
        self.fc_edge = nn.Sequential(
            nn.Linear(dim_in_edge+2*dim_in_node+dim_in_u,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2)
        )
        self.fc_edge2 = nn.Linear(hidden_dim2, dim_out)
        self.fc_node = nn.Sequential(
            nn.Linear(dim_in_node+dim_in_u+dim_out,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2)
        )
        self.fc_node2 = nn.Linear(hidden_dim2, dim_out)
        self.fc_u = nn.Sequential(
            nn.Linear(2*dim_out + dim_in_u,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2)
        )
        self.fc_u2 = nn.Linear(hidden_dim2, dim_out)
        self.dim_out = dim_out
        # Check Graph batch

        self.graph_msg = fn.copy_edge(edge='edge_feat', out='m')
        self.graph_reduce = fn.sum(msg='m', out='h')

    def graph_message_func(self,edges):
        return {'m': edges.data['edge_feat'] }

    def graph_reduce_func(self,nodes):
        msgs = torch.sum(nodes.mailbox['m'], dim=1)
        return {'h': msgs}

    def compute_edge_repr(self, graph, edges, g_repr):
        edge_nums = graph.batch_num_edges
        u = torch.cat([g[None,:].repeat(num_edge,1) for g, num_edge
                       in zip(g_repr,edge_nums)], dim=0)

        inp = torch.cat([edges.data['edge_feat'],edges.src['node_feat'],edges.dst['node_feat'], u], dim=-1)
        return {'edge_feat' : self.fc_edge2(F.relu(self.fc_edge(inp)))}

    def compute_node_repr(self, graph, nodes, g_repr):
        node_nums = graph.batch_num_nodes
        u = torch.cat([g[None, :].repeat(num_node, 1) for g, num_node
                       in zip(g_repr, node_nums)], dim=0)

        if 'h' not in nodes.data.keys():
            messages = torch.zeros([nodes.data['node_feat'].shape[0], self.dim_out]).double()
        else:
            messages = nodes.data['h']
        inp = torch.cat([nodes.data['node_feat'], messages, u], dim=-1)
        return {'node_feat' : self.fc_node2(F.relu(self.fc_node(inp)))}

    def compute_u_repr(self, n_comb, e_comb, g_repr):
        inp = torch.cat([n_comb, e_comb, g_repr], dim=-1)
        return self.fc_u2(F.relu(self.fc_u(inp)))

    def forward(self, graph, edge_feat, node_feat, g_repr):
        node_trf_func = lambda x: self.compute_node_repr(nodes=x, graph=graph, g_repr=g_repr)

        graph.edata['edge_feat'] = edge_feat
        graph.ndata['node_feat'] = node_feat
        edge_trf_func = lambda x : self.compute_edge_repr(edges=x, graph=graph, g_repr=g_repr)

        graph.apply_edges(edge_trf_func)
        graph.update_all(self.graph_message_func, self.graph_reduce_func, node_trf_func)

        e_comb = dgl.sum_edges(graph, 'edge_feat')
        n_comb = dgl.sum_nodes(graph, 'node_feat')

        e_out = graph.edata['edge_feat']
        n_out = graph.ndata['node_feat']

        e_keys = list(graph.edata.keys())
        n_keys = list(graph.ndata.keys())
        for key in e_keys:
            graph.edata.pop(key)
        for key in n_keys:
            graph.ndata.pop(key)

        return e_out, n_out, self.compute_u_repr(n_comb, e_comb, g_repr)

class GNNBlock(nn.Module):
    """
       A class that implements a single layer of the GNNs used within this research.
    """
    def __init__(self, dim_in_node, dim_in_u, edge_dim, hidden_dim, hidden_dim2, dim_out, with_edge_learning=False):
        """
            Constructor for the GNN model.
            Args:
                dim_in_node : Integer representing length of node feature vectors.
                dim_in_u : Integer representing length of graph level feature vectors.
                edge_dim : Integer representing the length of message vectors propagated through edges.
                hidden_dim : Integer representing the size of the first hidden layer when processing input vectors.
                hidden_dim2 : Integer representing the size of the second hidden layer when processing input vectors.
                dim_out : Integer representing the length of the outputted representations from the GNN
                with_edge_learning : A flag to decide whether the GNN includes operations to learn graph structure or not.

        """

        super(GNNBlock, self).__init__()
        self.with_edge_learning = with_edge_learning
        self.edge_dim = edge_dim

        # Add fully connected layers to produce edge representations
        self.fc_edge = nn.Linear(2*dim_in_node + dim_in_u + edge_dim, hidden_dim)
        self.fc_edge2 = nn.Linear(hidden_dim, edge_dim)

        # Add fully connected layers to produce node representations
        self.fc_node = nn.Linear(dim_in_node + dim_in_u + edge_dim, hidden_dim)
        self.fc_node2 = nn.Linear(hidden_dim, hidden_dim2)

        # Add fully connected layers to produce edge logits (\in R^{2}) to decide the existence of edges
        if self.with_edge_learning:
            self.fc_edge_learning = nn.Linear(2 * hidden_dim2 + dim_in_u, hidden_dim)
            self.fc_edge2_learning = nn.Linear(hidden_dim, edge_dim)
            self.edge_logit_readout = nn.Linear(edge_dim, 2)

        # Add final readout layer to produce final representation of nodes/edges
        self.fc_node_readout1 = nn.Linear(hidden_dim2, dim_out)
        self.fc_edge_readout1 = nn.Linear(edge_dim, dim_out)

        # Add DGL functions for edge message aggregation
        self.graph_reduce = fn.sum(msg='m', out='h')

    def graph_message_func(self,edges):
        """
            Method to propagate edge messages (edge_feat) to destination nodes
            Args:
                edges : edges of a DGLGraph.

            Output:
                Dictionary of edge messages (message stored in m)
        """
        return {'m': edges.data['edge_feat'] }

    def compute_edge_repr(self, graph, edges):
        """
            Method to compute edge representation from an input DGLGraph.
            Args:
                graph : Input DGLGraph.
                edges : Edges of DGLGraph.

            Output:
                Dictionary of updated edge representation (Stored in edge_feat)
        """

        # Concatenate message from source node (edges.src['h']) and edge's source and destination node features.
        inp = torch.cat([
            edges.src['h'], edges.src['node_feat'], edges.dst['node_feat']
        ], dim=-1)

        # Compute messgaes being passed
        out = self.fc_edge2(F.relu(self.fc_edge(inp)))

        # If edge data is provided as input (usually after another network learns the graph structure/adjacency matrix)
        # Then filter message according to the adjacency matrix passed into the network
        # (through edges.data["edge_existence"])
        if not self.with_edge_learning:
            out = torch.unsqueeze(out, -2)
            out = torch.cat([torch.zeros_like(out).double(), out], axis=-2)
            edge_weights = torch.unsqueeze(edges.data["edge_existence"], -1)
            out = torch.sum(out * edge_weights, axis=-2)

        return {'edge_feat' : out}

    def compute_edge_logits(self, graph, edges):
        """
            Method to compute edge logits from an input DGLGraph for adjacency matrix reasoning.
            Args:
                graph : Input DGLGraph.
                edges : Edges of DGLGraph.

            Output:
                Dictionary of updated edge logits for adjacency matrix inference. (Stored in edge_combined_rep)
        """

        inp = torch.cat([
            edges.src['node_feat'], edges.dst['node_feat']
        ], dim=-1)

        inp_flip = torch.cat([
            edges.dst['node_feat'], edges.src['node_feat']
        ], dim=-1)

        combined_rep =  self.fc_edge2_learning(F.relu(self.fc_edge_learning(inp))) + \
                        self.fc_edge2_learning(F.relu(self.fc_edge_learning(inp_flip)))
        return {
            'edge_combined_rep' : combined_rep
        }


    def compute_node_repr(self, graph, nodes):
        """
            Method to compute node representation for nodes in an input DGLGraph.
            Args:
                graph : Input DGLGraph.
                nodes : Nodes of input DGLGraph.

            Output:
                Dictionary of updated node representation. (Stored in node_feat)
        """
        node_nums = graph.batch_num_nodes
        if 'h' not in nodes.data.keys():
            messages = torch.zeros([nodes.data['node_feat'].shape[0], self.edge_dim]).double()
        else:
            messages = nodes.data['h']
        inp = torch.cat([nodes.data['node_feat'], messages], dim=-1)
        return {'node_feat' : self.fc_node2(F.relu(self.fc_node(inp)))}

    def forward(self, graph, node_feat, msg_passing_steps=1, edge_structure=None):
        """
            Method that implements forward propagation of input in network.
            Args:
                graph : Input DGLGraph.
                node_feat : Node features of input DGLGraph.
                msg_passing_steps : Number of message passing steps during inference.
                edge_structure : Input edge adjacency matrix (Only used when edge adjacency reasoning not needed)

            Output:
                n_out : Updated node representations
                e_out : Updated edge representations
        """
        node_trf_func = lambda x: self.compute_node_repr(nodes=x, graph=graph)
        edge_trf_func = lambda x: self.compute_edge_repr(edges=x, graph=graph)
        edge_aggregate_func = lambda x: self.compute_edge_logits(edges=x, graph=graph)

        if not self.with_edge_learning:
            assert edge_structure != None
            graph.edata['edge_existence'] = edge_structure

        graph.edata['edge_feat'] = torch.zeros([graph.number_of_edges(), self.edge_dim]).to(node_feat.device).double()
        graph.ndata['node_feat'] = node_feat
        graph.ndata['h'] = torch.zeros([graph.number_of_nodes(), self.edge_dim]).to(node_feat.device).double()

        for _ in range(msg_passing_steps):
            graph.apply_edges(edge_trf_func)
            graph.update_all(self.graph_message_func, self.graph_reduce)

        graph.apply_nodes(node_trf_func)
        n_out = self.fc_node_readout1(graph.ndata['node_feat'])

        edge_readout_func = lambda x: self.compute_edge_logits(edges=x, graph=graph)
        graph.apply_edges(edge_readout_func)
        e_out_readout = self.fc_edge_readout1(graph.edata['edge_combined_rep'])

        if self.with_edge_learning:
            graph.apply_edges(edge_aggregate_func)
            e_out = self.edge_logit_readout(graph.edata['edge_combined_rep'])

        e_keys = list(graph.edata.keys())
        n_keys = list(graph.ndata.keys())
        for key in e_keys:
            graph.edata.pop(key)
        for key in n_keys:
            graph.ndata.pop(key)

        if not self.with_edge_learning:
            return n_out, e_out_readout

        return n_out, e_out_readout, e_out

class Encoder(nn.Module):
    """
        A class that encodes learner's observation/actions.
    """

    def __init__(self, nr_inputs, layer_dims, batch_norm=False):
        """
            Constructor for the encoder.
            Args:
                nr_inputs : Feature length for input
                layer_dims : List containing hidden layer sizes (from first hidden layer to last)
                batch_norm : Flag on whether to add batch normalization in networks or not.

        """
        super(Encoder, self).__init__()
        if batch_norm:
            self.enc = nn.Sequential(
                nn.Linear(nr_inputs, layer_dims[0]),
                nn.BatchNorm1d(layer_dims[0]),
                nn.ReLU(),
                nn.Linear(layer_dims[0], layer_dims[1]),
                nn.BatchNorm1d(layer_dims[1]))
        else:
            self.enc = nn.Sequential(
                nn.Linear(nr_inputs, layer_dims[0]),
                nn.ReLU(),
                nn.Linear(layer_dims[0], layer_dims[1]))

    def forward(self, input):
        """
            Method that implements forward propagation of input in encoder network.
            Args:
                input : Input tensor (Can be obs/action tensor).

            Output :
                rep : Output representation after encoding.
        """
        rep = self.enc(input)
        return rep

class Decoder(nn.Module):
    """
        A class that decodes learner's representation of obs into the original obs.
    """

    def __init__(self, nr_inputs, layer_dims, batch_norm=False):
        """
            Constructor for the decoder.
            Args:
                nr_inputs : Feature length for observation
                layer_dims : List containing hidden layer sizes of encoder (from first hidden layer to last)
                batch_norm : Flag on whether to add batch normalization in networks or not.

        """
        super(Decoder, self).__init__()
        if batch_norm:
            self.decoder = nn.Sequential(
                nn.Linear(layer_dims[1], layer_dims[0]),
                nn.BatchNorm1d(layer_dims[0]),
                nn.ReLU(),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(layer_dims[1], layer_dims[0]),
                nn.ReLU(),
            )

        # Network to compute Gaussian mean of obs distribution
        self.dec_mean = nn.Linear(layer_dims[0], nr_inputs)

        # Network to compute Gaussian variance of obs distribution
        self.dec_std = nn.Sequential(
            nn.Linear(layer_dims[0], nr_inputs),
            nn.Softplus())

    def forward(self, input):
        """
            Method that implements forward propagation oin decoder network.
            Args:
                input : Input representation tensor.

            Output :
                rep : Output mean and variance of obs distribution after decoding.
        """
        rep = self.decoder(input)
        mean, var = self.dec_mean(rep), self.dec_std(rep)
        return mean, var

class JointUtilLayer(nn.Module):
    """
        Constructor for GPL's Coordination Graph-based joint action value network.
        Args:
            dim_in_node : Size of input node representations
            mid_pair : Size of network hidden layers for computing pairwise utility terms.
            mid_nodes : Size of network hidden layers for computing individual utility terms.
            num_acts : Output (Action space) size
            mid_pair_out : Matrix rank for pairwise utility term's low rank decomposition.
    """
    def __init__(self, dim_in_node, mid_pair, mid_nodes, num_acts, mid_pair_out=8, device=None):
        super(JointUtilLayer, self).__init__()
        self.mid_pair = mid_pair
        self.num_acts = num_acts

        # Define network components for joint utility term computation
        self.ju1 = nn.Linear(3*dim_in_node, self.mid_pair)
        self.ju3 = nn.Linear(self.mid_pair, self.mid_pair)
        self.mid_pair_out = mid_pair_out
        self.ju2 = nn.Linear(self.mid_pair,num_acts*self.mid_pair_out)

        # Define network components for individual utility term computation
        self.iu1 = nn.Linear(2*dim_in_node, mid_nodes)
        self.iu3 = nn.Linear(mid_nodes, mid_nodes)
        self.iu2 = nn.Linear(mid_nodes, num_acts)

        self.num_acts = num_acts
        self.device = device


    def compute_node_data(self, nodes):
        """
            Method that computes individual utility terms. It updates input node by adding the 'indiv_util' field
            to each node.
            Args:
                nodes : dgl graph nodes that at least contains 'node_feat_u' features in it.
        """
        node_data = self.iu2(F.relu(self.iu3(F.relu(self.iu1(nodes.data['node_feat_u'])))))
        node_data = node_data * nodes.data['agent_existence_flag']

        return {'indiv_util': node_data}

    def compute_edge_data(self, edges):
        """
            Method that computes pairwise utility terms. It updates input edges by adding the 'edge_feat_u'
            and 'edge_feat_reflected_u' field to every edge.
            Args:
                edges : dgl graph edges that at least contains 'edge_feat_u' and edge_feat_reflected_u' features in it.
        """
        inp_u = edges.data['edge_feat_u']
        inp_reflected_u = edges.data['edge_feat_reflected_u']
        src_flag, dst_flag = edges.src['agent_existence_flag'].unsqueeze(-1), edges.dst['agent_existence_flag'].unsqueeze(-1)
        src_flag = src_flag.repeat([1,self.num_acts, self.mid_pair_out])
        dst_flag = dst_flag.repeat([1,self.num_acts, self.mid_pair_out])

        # Compute the util components
        util_comp = self.ju2(F.relu(self.ju3(F.relu(self.ju1(inp_u))))).view(-1,self.num_acts, self.mid_pair_out) * src_flag
        util_comp_reflected = (self.ju2(F.relu(self.ju3(F.relu(self.ju1(inp_reflected_u))))).view(-1,self.num_acts,
                                                                                self.mid_pair_out) * dst_flag).permute(0,2,1)

        util_vals = torch.bmm(util_comp, util_comp_reflected).permute(0,2,1)
        final_u_factor = util_vals
        reflected_util_vals = final_u_factor.permute(0, 2, 1)

        return {
            'util_vals': final_u_factor,
            'reflected_util_vals': reflected_util_vals
        }

    def graph_pair_inference_func(self, edges):
        """
            Method that computes the expected contribution from joint utility terms after being weighted
            based on the likelihood of the actions associated to the terms. This function updates the edges
            by adding the "edge_all_sum_prob" field to the edges.
            Args:
                edges : dgl graph edges that at least contains 'util_vals' and 'probs' features in it.
        """

        src_prob, dst_prob = edges.src["probs"], edges.dst["probs"]
        src_prob = src_prob * edges.src['agent_existence_flag']
        dst_prob = dst_prob * edges.dst['agent_existence_flag']

        edge_all_sum = (edges.data["util_vals"] * src_prob.unsqueeze(1) *
                        dst_prob.unsqueeze(-1)).sum(dim=-1).sum(dim=-1,
                        keepdim=True)

        return {'edge_all_sum_prob': edge_all_sum}

    def graph_pair_inference_func_train(self, edges):
        """
            Same as graph_pair_inference_func, but only for training. This function updates the edges
            by adding the "edge_all_sum_prob_train" field to the edges.
            Args:
                edges : dgl graph edges that at least contains 'util_vals' and 'probs' features in it.
        """

        src_prob, dst_prob = edges.src["probs"], edges.dst["probs"]
        src_prob = src_prob * edges.src['agent_existence_flag'] * (1-edges.src['agent_visible_flag'])
        dst_prob = dst_prob * edges.dst['agent_existence_flag'] * (1-edges.dst['agent_visible_flag'])

        edge_all_sum = (edges.data["util_vals"] * src_prob.unsqueeze(1) *
                        dst_prob.unsqueeze(-1)).sum(dim=-1).sum(dim=-1,
                        keepdim=True)

        return {'edge_all_sum_prob_train': edge_all_sum}

    def graph_dst_inference_func(self, edges):
        """
            Method that computes the expected contribution from joint utility terms after being weighted
            based on the likelihood of the actions associated to the source node for the edge. This function
            updates the edges by adding the "marginalized_u" field to the edges.
            Args:
                edges : dgl graph edges that at least contains 'util_vals' and 'probs' features in it.
        """
        src_prob = edges.src["probs"]
        src_prob = src_prob * edges.src['agent_existence_flag']
        u_message = (edges.data["util_vals"] * src_prob.unsqueeze(1)).sum(dim=-1)

        return {'marginalized_u': u_message}

    def graph_dst_inference_func_train(self, edges):
        """
            Similar with graph_dst_inference_func. However, this message is onky computed if src is
            observed and dst is observed.This function
            updates the edges by adding the "marginalized_u_train" field to the edges.
            Args:
                edges : dgl graph edges that at least contains 'util_vals' and 'probs' features in it.
        """
        src_prob = edges.src["probs"]
        src_prob = src_prob * edges.src['agent_existence_flag'] * (1-edges.src['agent_visible_flag']) * edges.dst['agent_existence_flag']* edges.dst['agent_visible_flag']
        u_message = (edges.data["util_vals"] * src_prob.unsqueeze(1)).sum(dim=-1)

        return {'marginalized_u_train': u_message}

    def graph_node_inference_func(self, nodes):
        """
            Method that computes the expected contribution from individual terms after being weighted
            based on the likelihood of the actions of agents associated to the term. This function
            updates the nodes by adding the "expected_indiv_util" field to the edges.
            Args:
                nodes : dgl graph nodes that at least contains 'indiv_util' and 'probs' features in it.
        """
        indiv_util = nodes.data["indiv_util"]
        weighting = nodes.data["probs"]
        weighting = weighting * nodes.data['agent_existence_flag']

        return {"expected_indiv_util" : (indiv_util*weighting).sum(dim=-1)}

    def graph_node_inference_func_train(self, nodes):
        """
            Same as graph_node_inference_func, but only used in training. This function
            updates the nodes by adding the "expected_indiv_util_train" field to the edges.
            Args:
                nodes : dgl graph nodes that at least contains 'indiv_util' and 'probs' features in it.
        """
        indiv_util = nodes.data["indiv_util"]
        weighting = nodes.data["probs"]
        weighting = weighting * nodes.data['agent_existence_flag'] *  (1-nodes.data["agent_visible_flag"])

        return {"expected_indiv_util_train" : (indiv_util*weighting).sum(dim=-1)}

    def graph_reduce_func(self, nodes):
        """
            Method that sums up all the marginalized_u messages collected by a node.
            Args:
                nodes : dgl graph nodes.
        """
        return {'utility_output': torch.sum(nodes.mailbox['marginalized_u'], dim=1)}

    def graph_reduce_func_train(self, nodes):
        """
            Method that sums up all the marginalized_u messages collected by a node.
            Args:
                nodes : dgl graph nodes.
        """
        return {'utility_output_train': torch.sum(nodes.mailbox['marginalized_u_train'], dim=1)}

    def graph_u_sum(self, graph, edges, acts):
        src, dst = graph.edges()
        acts_src = torch.Tensor([acts[idx] for idx in src.tolist()]).to(self.device)

        u = edges.data['util_vals']
        reshaped_acts = acts_src.view(u.shape[0], 1, -1).long().repeat(1, self.num_acts, 1)
        u_msg = u.gather(-1, reshaped_acts).permute(0,2,1).squeeze(1) * edges.src["agent_visible_flag"] * edges.dst["agent_visible_flag"]
        return {'u_msg': u_msg}

    def graph_sum_all(self, nodes):
        util_msg = torch.sum(nodes.mailbox['u_msg'], dim=1)
        return {'u_msg_sum': util_msg}

    def forward(self, graph,
                node_feats_u, edge_feats_u,
                edge_feat_reflected_u, existence_flags,
                mode="train",
                node_probability = None,
                joint_acts=None,
                visibility_flags=None):
        """
            Method that computes the output of the joint action value model.
            Args:
                graph : dgl graph used as the models Coordination Graph model.
                edge_feats_u : Edge features for the model. (Indexed based on <src, dst>)
                node_feats_u : Node features for the model.
                edge_feats_reflected_u : Reflected edge features for model. (Indexed based on <dst, src>)
                mode : Value computation mode (Must either be "train" or "inference"). If mode is "train",
                       network output is the joint action value for a single possible joint action provided by
                       the joint_acts variable. Otherwise if mode is "inference", network output is the marginalized
                       expected return weighted by the action probability distribution provided through the
                       node_probability variable.
                node_probability : Teammate predicted action probability distribution. Only used when mode is "inference".
                joint_acts : The joint action which return is queried. Only used when mode is "train"
        """

        graph.edata['edge_feat_u'] = edge_feats_u
        graph.edata['edge_feat_reflected_u'] = edge_feat_reflected_u
        graph.ndata['node_feat_u'] = node_feats_u
        graph.ndata['agent_existence_flag'] = existence_flags

        n_weights = torch.zeros([node_feats_u.shape[0],1]).to(self.device).double()

        zero_indexes, offset = [0], 0
        num_nodes = graph.batch_num_nodes

        # Mark all 0-th index nodes
        for a in num_nodes[:-1]:
            offset += a
            zero_indexes.append(offset)

        n_weights[zero_indexes] = 1
        graph.ndata['weights'] = n_weights
        graph.ndata['mod_weights'] = 1-n_weights

        # Compute individual and pairwise utility terms
        graph.apply_nodes(self.compute_node_data)
        graph.apply_edges(self.compute_edge_data)

        if "inference" in mode:

            # Compute learner's action value through a weighted sum of the joint action values
            # (weighted by joint action probability)
            graph.ndata["probs"] = node_probability
            src, dst = graph.edges()
            src_list, dst_list = src.tolist(), dst.tolist()

            # Mark edges not connected to zero
            e_nc_zero_weight = torch.zeros([edge_feats_u.shape[0],1]).to(self.device).double()
            all_nc_edges = [idx for idx, (src, dst) in enumerate(zip(src_list,dst_list)) if
                            (not src in zero_indexes) and (not dst in zero_indexes)]
            e_nc_zero_weight[all_nc_edges] = 0.5
            graph.edata["nc_zero_weight"] = e_nc_zero_weight

            graph.apply_edges(self.graph_pair_inference_func)
            graph.update_all(message_func=self.graph_dst_inference_func, reduce_func=self.graph_reduce_func,
                             apply_node_func=self.graph_node_inference_func)

            total_connected = dgl.sum_nodes(graph, 'utility_output', 'weights')
            total_n_connected = dgl.sum_edges(graph, 'edge_all_sum_prob', 'nc_zero_weight')
            total_expected_others_util = dgl.sum_nodes(graph, "expected_indiv_util", "mod_weights").view(-1,1)
            total_indiv_util_zero = dgl.sum_nodes(graph, "indiv_util", "weights")

            returned_values = (total_connected + total_n_connected) + \
                              (total_expected_others_util + total_indiv_util_zero)

            e_keys = list(graph.edata.keys())
            n_keys = list(graph.ndata.keys())

            for key in e_keys:
                graph.edata.pop(key)

            for key in n_keys:
                graph.ndata.pop(key)

            return returned_values

        graph.ndata["agent_visible_flag"] = visibility_flags
        graph.ndata["probs"] = node_probability
        # Compute input joint action value
        # But first marginalize the contribution of edges connecting two existing nodes that are both not visible

        graph.apply_edges(self.graph_pair_inference_func_train)
        graph.update_all(message_func=self.graph_dst_inference_func_train, reduce_func=self.graph_reduce_func_train,
                         apply_node_func=self.graph_node_inference_func_train)
        m_func = lambda x: self.graph_u_sum(graph, x, joint_acts)
        graph.update_all(message_func=m_func,
                        reduce_func=self.graph_sum_all)

        indiv_u_zeros = graph.ndata['indiv_util']
        marginalized_util_pairs_u = graph.ndata["utility_output_train"]
        u_msg_sum_zeros = 0.5 * graph.ndata['u_msg_sum']

        graph.ndata['utils_sum_all'] = (
                indiv_u_zeros + u_msg_sum_zeros + marginalized_util_pairs_u
        ).gather(-1,torch.Tensor(joint_acts)[:,None].to(self.device).long())
        q_values = dgl.sum_nodes(graph, 'utils_sum_all','agent_visible_flag')
        q_values = q_values + dgl.sum_nodes(graph, "expected_indiv_util_train").view(-1,1) + dgl.sum_edges(graph, 'edge_all_sum_prob_train')

        e_keys = list(graph.edata.keys())
        n_keys = list(graph.ndata.keys())

        for key in e_keys:
            graph.edata.pop(key)

        for key in n_keys:
            graph.ndata.pop(key)

        return q_values
