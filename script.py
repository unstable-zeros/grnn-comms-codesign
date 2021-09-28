########################################################
# Script for running all experiments
# Fengjun Yang, 2021
########################################################

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import time

# Source Files
import grnn, exp_utils, controller
import env.dlqr

# For saving data
import json

# Other bookkeeping
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
np.set_printoptions(precision=3)
plt.style.use('seaborn-deep')
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "sans-serif",
#    "font.sans-serif": ["Palatino"]})
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rcParams.update({'font.size': 15})

########################################################
# Define the costs being used
########################################################

def grnn_criterion(x_traj, u_traj, env, model):
    norms = torch.norm(model.S_()) + torch.norm(model.A) + torch.norm(model.B)
    return env.cost(x_traj, u_traj) + 2 * norms

def gcnn_criterion(x_traj, u_traj, env, model):
    return env.cost(x_traj, u_traj)

# beta = 1        # For |A|=0.995
beta = 2
def sparse_criterion(x_traj, u_traj, env, model):
    norms = torch.norm(model.S) + torch.norm(model.A)
    return env.cost(x_traj, u_traj) + \
        beta * torch.sum(torch.abs(model.S_())) + 2 * norms

########################################################
# Experiment 1: Benchmarking GRNN Performance
########################################################
import experiments.lossdecrease
import experiments.decrease_plot

# Load parameters
with open('./config/benchmark.json') as f:
    params = json.load(f)
params['device'] = device
params['grnn_criterion'] = grnn_criterion
params['gcnn_criterion'] = gcnn_criterion
params['sparse_criterion'] = sparse_criterion

filename = 'benchmark_stable.data'
params['filename'] = filename
params['A_norm'] = 0.995
experiments.lossdecrease.run(**params)
experiments.decrease_plot.summarize(filename, params['log_interval'])

filename = 'benchmark_unstable.data'
params['filename'] = filename
params['A_norm'] = 1.05
experiments.lossdecrease.run(**params)
experiments.decrease_plot.summarize(filename, params['log_interval'])

########################################################
# Experiment 2: Communication Topology Co-Design
########################################################
import experiments.tradeoff
import experiments.tradeoff_plot

def sparse_criterion(x_traj, u_traj, env, model, beta):
    norms = torch.norm(model.S) + torch.norm(model.A)
    return env.cost(x_traj, u_traj) + \
        beta * torch.sum(torch.abs(model.S_())) + 2 * norms

with open('./config/tradeoff.json') as f:
    params = json.load(f)

# Open data file and erase the data in there
filename = 'tradeoff.data'
with open(filename, 'w') as f:
    pass

params['filename'] = filename
params['device'] = device
params['grnn_criterion'] = grnn_criterion
params['sparse_criterion'] = sparse_criterion

params['A_norm'] = 0.995
params['betas'] = np.logspace(-2.5, 1, 6)
experiments.tradeoff.run(**params)

params['A_norm'] = 1.005
params['betas'] = np.logspace(-2.5, 1, 6)
experiments.tradeoff.run(**params)

params['A_norm'] = 1.05
params['betas'] = np.logspace(-2, 1, 6)
experiments.tradeoff.run(**params)

params['A_norm'] = 1.1
params['betas'] = np.logspace(-2, 1.5, 6)
experiments.tradeoff.run(**params)

params['A_norm'] = 1.2
params['betas'] = np.logspace(-1, 3, 6)
experiments.tradeoff.run(**params)

experiments.tradeoff_plot.summarize(filename)
