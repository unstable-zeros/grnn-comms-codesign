# These are some wrappers for running the code from [GS20]
import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import GS21.architecturesTime as archit
import GS21
from controller import GCNNController

nNodes = 20 # Number of nodes
S = np.random.random((nNodes,nNodes))

def generate_model(S, device):
    model_params = {} # Model parameters for the Local GNN (LclGNN)
    model_params['dimNodeSignals'] = [1, 32] # Features per layer
    model_params['nFilterTaps'] = [5] # Number of filter taps per layer
    model_params['bias'] = False
    model_params['nonlinearity'] = nn.Tanh # Selected nonlinearity
    model_params['dimReadout'] = [1] # Dimension of the fully connected layers
    model_params['GSO'] = S
    model = archit.LocalGNN_t(**model_params)
    model.to(device)
    return model

def get_trajectory(model, env, x0s, T):
    # Their architecture takes in an additional time dimension
    batch_size = x0s.size(0)
    N = x0s.size(1)
    x = torch.zeros(batch_size, T+1, N, 1, device=x0s.device)
    u = torch.zeros(batch_size, T, N, 1, device=x0s.device)
    x[:,0] = x0s
    for t in range(T):
        ut = model.forward(x[:,t].clone().reshape(batch_size, 1, 1, N)).reshape(
                batch_size, N, 1)
        x[:, t+1] = env.step(x[:, t].clone(), ut)
        u[:, t] = ut
    return x, u

def get_gcnn_controller(model, N):
    return GCNNController(model, N)

def gcnn_penalty(gcnnmodel, env):
    device = env.device
    size_penalty = GS21.penalty.GNNsize(gcnnmodel, None, device)
    stability_penalty = GS21.penalty.L2stabilityConstant(
            gcnnmodel, env.A, env.B, device)
    return size_penalty + stability_penalty
