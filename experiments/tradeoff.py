import torch
import numpy as np
import time
import json

# Source Files
import sys
sys.path.append("..")
import grnn
import exp_utils
import controller
import env.dlqr

# Savedir
filename = 'tradeoff.data'

## Environment Parameters
#N = 20
#degree = 5 + 1
#T = 50
#p = 1
#q = 1
#h = 5
#B_norm = 1
#
## Stability of the system
##A_norm = 0.995
##betas = np.logspace(-2.5, 1, 6)
##A_norm = 1.005
##betas = np.logspace(-2.5, 1, 6)
##A_norm = 1.05
##betas = np.logspace(-2, 1, 6)
##A_norm = 1.1
##betas = np.logspace(-2, 1.5, 6)
#A_norm = 1.2
#betas = np.logspace(-1, 3, 6)
#
## Training Parameters
#num_epoch = 150
#batch_size = 20
#ensemble_size = 3
#val_size = 200
#grnn_hidden_dim = 5
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
## Experiment Parameters
#threshold = 4e-3
#num_topologies = 30
#num_x0s = 100
#verbose = True

# Training losses for different setups
#def grnn_criterion(x_traj, u_traj, env, model):
#    norms = torch.norm(model.S_()) + torch.norm(model.A)
#    return env.cost(x_traj, u_traj) + 2 * norms
#
#def sparse_criterion(x_traj, u_traj, env, model, beta):
#    norms = torch.norm(model.S) + torch.norm(model.A)
#    return env.cost(x_traj, u_traj) + \
#        beta * torch.sum(torch.abs(model.S_())) + 2 * norms

#####################  SCRIPT  ##############################
def run(filename, N, degree, T, p, q, h, A_norm, B_norm, betas, num_epoch, batch_size,
        ensemble_size, val_size, grnn_hidden_dim, device, threshold,
        num_topologies, num_x0s, grnn_criterion, sparse_criterion, verbose):

    num_controllers = 6

    # Group parameters that are reused
    model_params = {
            'N':N,
            'T':T,
            'p':p,
            'q':p,
            'h':grnn_hidden_dim
    }
    training_params = {
            'T': T,
            'device': device,
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'ensemble_size': ensemble_size,
            'val_size': val_size,
    }

    # Create empty arrays to store results
    num_edges = np.zeros((num_topologies, len(betas)))
    reg_costs = np.zeros((num_topologies, len(betas)))
    retrain_costs = np.zeros((num_topologies, len(betas)))
    num_env_edges = np.zeros(num_topologies)

    for j in range(num_topologies):
        if verbose:
            print(j, end=', ')

        dlqrenv, G = env.dlqr.generate_lq_env(
                N, degree, device, A_norm=A_norm, B_norm=B_norm)
        if verbose:
            print(exp_utils.estimate_controller_cost(
              dlqrenv, T, [controller.ZeroController(N, q)], 1000
            ))
        num_env_edges[j] = torch.sum(dlqrenv.S > 0).item()
        for i, beta in enumerate(betas):
            # Train on full support to get a model and topology
            model = exp_utils.generate_model(model_params, dlqrenv,
                    use_given_support=False, S=None,
                    criterion=lambda x,u,e,m: sparse_criterion(x,u,e,m,beta),
                    **training_params)
            grnn_S = model.S_().detach()
            reg_costs[j, i] = exp_utils.estimate_controller_cost(
                    dlqrenv, T, [model.get_controller(1000)],1000)[1].item()

            # Retrain the model
            new_S = grnn_S.clone()
            new_S[torch.abs(new_S) < threshold] = 0
            num_edges[j, i] = torch.sum(new_S != 0).item()
            model = exp_utils.generate_model(model_params, dlqrenv,
                    use_given_support=True, S=new_S,
                    criterion=grnn_criterion,**training_params)
            retrain_costs[j, i] = exp_utils.estimate_controller_cost(
                    dlqrenv, T, [model.get_controller(1000)],1000)[1].item()

    avg_num_edges = num_edges.mean(0)
    avg_reg_costs = reg_costs.mean(0)
    avg_retrain_costs = retrain_costs.mean(0)
    average_edges = num_env_edges.mean()

    print(avg_num_edges)
    print(avg_reg_costs)
    print(avg_retrain_costs)

    with open(filename, 'a') as f:
        result_dict = {
                'Anorm': A_norm,
                'num_edges': num_edges.tolist(),
                'reg_costs': reg_costs.tolist(),
                'retrain_cost': retrain_costs.tolist(),
                'env_edges': num_env_edges.tolist(),
                'betas': betas.tolist()
        }
        f.write(json.dumps(result_dict))
        f.write('\n')
#########################################################
