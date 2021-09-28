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
import gcnn

# Savedir
# filename = 'exp1.data'

# # Environment Parameters
# N = 20
# degree = 5 + 1
# T = 50
# p = 1
# q = 1
# h = 5
# A_norm = 0.995
# B_norm = 1
# 
# # Training Parameters
# num_epoch = 100
# batch_size = 20
# ensemble_size = 2
# val_size = 50
# grnn_hidden_dim = 5
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 
# # Experiment Parameters
# num_topologies = 10
# num_x0s = 100
# verbose = True

# Training losses for different setups
def grnn_criterion(x_traj, u_traj, env, model):
    norms = torch.norm(model.S_()) + torch.norm(model.A) + torch.norm(model.B)
    return env.cost(x_traj, u_traj) + 2 * norms

def gcnn_criterion(x_traj, u_traj, env, model):
    return env.cost(x_traj, u_traj)


#####################  SCRIPT  ##############################
def run(filename, N, degree, T, p, q, h, A_norm, B_norm, num_epoch, batch_size,
        ensemble_size, val_size, grnn_hidden_dim, num_topologies, num_x0s,
        verbose, device, grnn_criterion, gcnn_criterion):

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
            'val_size': val_size
    }

    # Create arrays to store results
    total_rel_costs = torch.zeros(num_controllers, device=device)
    rel_costs_table = torch.zeros((num_topologies, num_controllers), device=device)
    envs = []

    for counter in range(num_topologies):

        # Generate environment
        dlqrenv, G = env.dlqr.generate_lq_env(
                N, degree, device, A_norm=A_norm, B_norm=B_norm)

        # Controller 0: Zero control (i.e. autonomous system)
        controllers = [controller.ZeroController(N, q)]

        # Controller 1: grnn with untrainable S
        model_params['S_trainable'] = False
        grnn_fixed_S = exp_utils.generate_model(
                model_params, dlqrenv, use_given_support=True,
                S=dlqrenv.S.clone(), criterion=grnn_criterion,
                **training_params)
        controllers.append( grnn_fixed_S.get_controller(num_x0s) )

        # Controller 2: grnn with untrainable S
        model_params['S_trainable'] = True
        grnn_support_S = exp_utils.generate_model(
                model_params, dlqrenv, use_given_support=True,
                S=dlqrenv.S.clone(), criterion=grnn_criterion,
                **training_params)
        controllers.append( grnn_support_S.get_controller(num_x0s) )

        # Controller 3: gcnn as in [GS20]
        gcnn_model = exp_utils.generate_gcnn_model(
                S=dlqrenv.S.clone(), N=N, env=dlqrenv,
                criterion=gcnn_criterion, **training_params)
        controllers.append( gcnn.get_gcnn_controller(gcnn_model, N) )

        # Controller 4: grnn with dense S
        grnn_dense_S = exp_utils.generate_model(
                model_params, dlqrenv, use_given_support=False,
                S=None, criterion=grnn_criterion,
                **training_params)
        controllers.append( grnn_dense_S.get_controller(num_x0s) )

        # Test the performance of GRNN on this env
        rel_costs = exp_utils.estimate_controller_cost(
            dlqrenv, T, controllers, num_x0s=num_x0s)
        total_rel_costs += rel_costs

        rel_costs_table[counter] = rel_costs
        envs.append(dlqrenv)

        # Print progress
        if verbose:
            print('Iteration: {}'.format(counter+1))
            print(rel_costs.detach().cpu().numpy())
            print(total_rel_costs.data.detach().cpu().numpy() / (counter+1))

    # Print result
    print(total_rel_costs / num_topologies)

    with open(filename, 'w') as f:
        f.write(json.dumps(rel_costs_table.tolist()))
#############################################################
