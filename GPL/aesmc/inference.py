from . import math
import numpy as np
import torch


def sample_ancestral_index(log_weight):
    """Sample ancestral index using systematic resampling.

    input:
        log_weight: log of unnormalized weights, Tensor/Variable
            [batch_size, num_particles]
    output:
        zero-indexed ancestral index: LongTensor/Variable
            [batch_size, num_particles]
    """

    device = log_weight.device
    if torch.sum(log_weight != log_weight) != 0:
        print(log_weight)
    assert(torch.sum(log_weight != log_weight) == 0)
    batch_size, num_particles = log_weight.size()
    indices = np.zeros([batch_size, num_particles])

    uniforms = np.random.uniform(size=[batch_size, 1])
    pos = (uniforms + np.arange(0, num_particles)) / num_particles

    normalized_weights = np.exp(math.lognormexp(
        log_weight.cpu().detach().numpy(),
        dim=1
    ))
    # np.ndarray [batch_size, num_particles]
    cumulative_weights = np.cumsum(normalized_weights, axis=1)

    # trick to prevent numerical issues
    cumulative_weights = cumulative_weights / np.max(
        cumulative_weights,
        axis=1,
        keepdims=True
    )

    for batch in range(batch_size):
        indices[batch] = np.digitize(pos[batch], cumulative_weights[batch])

    temp = torch.from_numpy(indices).long().to(device)

    return temp

def sample_ancestral_index_ratio(log_weight, threshold, variance):
    """Sample ancestral index using systematic resampling with ratio changes. 
    Only resample when the weights variance is above some threshold


    input:
        log_weight: log of unnormalized weights, Tensor/Variable
            [batch_size, num_particles]
    output:
        zero-indexed ancestral index: LongTensor/Variable
            [batch_size, num_particles]
    """

    device = log_weight.device
    if torch.sum(log_weight != log_weight) != 0:
        print(log_weight)
    assert(torch.sum(log_weight != log_weight) == 0)
    batch_size, num_particles = log_weight.size()
    indices = np.zeros([batch_size, num_particles])

    uniforms = np.random.uniform(size=[batch_size, 1])
    pos = (uniforms + np.arange(0, num_particles)) / num_particles

    normalized_weights = np.exp(math.lognormexp(
        log_weight.cpu().detach().numpy(),
        dim=1
    ))
    # np.ndarray [batch_size, num_particles]
    cumulative_weights = np.cumsum(normalized_weights, axis=1)

    # trick to prevent numerical issues
    cumulative_weights = cumulative_weights / np.max(
        cumulative_weights,
        axis=1,
        keepdims=True
    )

    # implementing resampling based on particle ratio. Based on: http://www.cs.cmu.edu/~16831-f14/notes/F11/16831_lecture04_tianyul.pdf
    for batch in range(batch_size):
        # if particle weight variance is above some threshold we resample
        # meaning we are certain about some state
        if variance[batch] > threshold:
            indices[batch] = np.digitize(pos[batch], cumulative_weights[batch])
        else: 
            # if the variance is small, we do not resample, as all states might be likely
            indices[batch] = np.arange(num_particles)
        
        
    
    temp = torch.from_numpy(indices).long().to(device)

    return temp
