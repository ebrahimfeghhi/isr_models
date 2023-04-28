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

def encoding_policy_presentation_recall(list_length, delay_start, delay_middle):
    
    '''
    only encodes to EM at the end of list presentation and list recall (once for each)
    '''

    EM_encode_timesteps = []
    
    list_presented_time = delay_start + list_length - 1 # subtract 1 for 0 indexing
    list_recalled_time = list_presented_time + delay_middle + list_length
    EM_encode_timesteps.append(list_presented_time)
    EM_encode_timesteps.append(list_recalled_time)
    
    return EM_encode_timesteps
    
    
    
        
        
        
        
    
    