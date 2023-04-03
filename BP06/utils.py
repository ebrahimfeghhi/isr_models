import torch 
import numpy as np

# define helper functions
def softmax(probs):
    return np.exp(probs)/np.sum(np.exp(probs+1e-6))

def sigmoid(probs):
    return 1 / (1 + np.exp(-probs))

def clipper(probs):
    return np.clip(probs, 0, 1.0)

def vec_mag(vec):
    return np.sqrt(np.sum(vec**2))

def pearson_corr(vec1, vec2):
    vec1_centered = vec1 - np.mean(vec1)
    vec2_centered = vec2 - np.mean(vec2)
    vec1_var = np.var(vec1)
    vec2_var = np.var(vec2)
    return np.dot(vec1_centered, vec2_centered)

def cosine_sim(vec1, vec2):
    vec1mag = vec_mag(vec1)
    vec2mag= vec_mag(vec2)
    return round(np.dot(vec1, vec2) / (vec1mag * vec2mag),3)

# define functions for later
def softmax(probs):
    return np.exp(probs)/np.sum(np.exp(probs))

def sigmoid(probs):
    return 1 / (1 + np.exp(-probs))


def contextual_input_generator(timestep, list_length, batch_size, doe, context_neurons_size = 100, 
ramp_down_frac=0.7, S0=1.0, E0=1.0, S=0.8, E=0.8):

    context_signal = torch.zeros(batch_size, context_neurons_size)

    # if delay or end cue is the output, turn context neurons off
    if doe:
        return context_signal

    # all ramp down and ramp up neurons will fire at the same value 
    ramp_down_fr = S0*(S**(timestep))
    ramp_up_fr = E0*(E**(list_length-(timestep+1)))

    # designate ramp down and ramp up neurons 
    ramp_down_neurons = int(ramp_down_frac*context_neurons_size)

    # generate contextual signal 
    context_signal[:, :ramp_down_neurons] = ramp_down_fr
    context_signal[:, ramp_down_neurons:] = ramp_up_fr

    return context_signal


def EM_retrieval(h, h_prev_tensor, batch_idx):

    # no EM store exists 
    if batch_idx == 0:
        return h

    cos_sim = torch.nn.CosineSimilarity(dim=1)
    cos_sim_values = cos_sim(h, h_prev_tensor)
    softmax = torch.nn.Softmax(dim=0)
    softmax_cos_sim_values = softmax(cos_sim_values*100)
    EM = softmax_cos_sim_values@h_prev_tensor
    h_EM = .9*h + .1*EM
    return h_EM 
    
