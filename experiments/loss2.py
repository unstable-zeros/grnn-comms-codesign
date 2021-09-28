import torch
import torch.optim as optim
import numpy as np
import time
import json

# Source Files
import sys
sys.path.append("..")
import grnn, gcnn
import exp_utils
import controller
import env.dlqr

# Savedir
filename = 'benchmark.data'

# Environment Parameters
N = 20
degree = 5 + 1
T = 50
p = 1
q = 1
h = 5
A_norm = 1.05
B_norm = 1

# Training Parameters
num_epoch = 750
log_interval = 10
batch_size = 20
grnn_hidden_dim = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gcnn_decay_interval = 10
ensemble_size = 2
val_size = 50
threshold = 4e-3

# Experiment Parameters
num_topologies = 2
num_x0s = 100
verbose = True
num_controllers = 5
validation_size = 100

# Training losses for different setups
def grnn_criterion(x_traj, u_traj, env, model):
    norms = torch.norm(model.S_()) + torch.norm(model.A) + torch.norm(model.B)
    return env.cost(x_traj, u_traj) + 2 * norms

def gcnn_criterion(x_traj, u_traj, env, model):
    return env.cost(x_traj, u_traj)

def sparse_criterion(x_traj, u_traj, env, model):
    beta = 0.3
    norms = torch.norm(model.S) + torch.norm(model.A)
    return env.cost(x_traj, u_traj) + \
        beta * torch.sum(torch.abs(model.S_())) + 2 * norms
#####################  SCRIPT  ##############################
def run():

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
    num_log_points = int(num_epoch / log_interval)
    rel_costs_table = torch.zeros(
            (num_topologies, num_controllers, num_log_points), device=device)
    autonomous_costs = torch.zeros(num_topologies)
    num_sparse_edges = torch.zeros(num_topologies)
    num_env_edges = torch.zeros(num_topologies)

    for counter in range(num_topologies):

        # Generate environment
        dlqrenv, G = env.dlqr.generate_lq_env(
                N, degree, device, A_norm=A_norm, B_norm=B_norm)
        S = dlqrenv.S.clone()
        num_env_edges[counter] = torch.sum(S!=0).item()

        # Compute the cost for autonomous system
        zero_ctrl = controller.ZeroController(N, q)
        auto_cost = exp_utils.estimate_controller_cost(
                dlqrenv, T, [zero_ctrl], validation_size)
        autonomous_costs[counter] = auto_cost[1]

        # Define the models and optimizers
        optimizers = []
        schedulers = []

        # Model 1: GRNN with untrainable S
        model_params['S_trainable'] = False
        grnn_models = [grnn.GRNN(S, **model_params).to(device)]
        # Model 2: GRNN with trainable S
        model_params['S_trainable'] = True
        grnn_models.append(grnn.GRNN(S, **model_params).to(device))
        # Model 3: GRNN with full support
        grnn_models.append(grnn.GRNN(None, **model_params).to(device))
        for model in grnn_models:
            optimizer = optim.Adam(model.parameters(), lr=0.02)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch)
            optimizers.append(optimizer)
            schedulers.append(scheduler)

        # Model 4: GNN from [GS20]
        gcnn_model = gcnn.generate_model(S.cpu().numpy(), device)
        gcnn_optimizer = optim.Adam(gcnn_model.parameters(), lr=0.01)
        gcnn_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.9)

        # Write down the train x0s to reuse for training the sparsified model
        train_x0s = []

        for epoch in range(num_epoch):
            x0s = dlqrenv.random_x0(batch_size)
            train_x0s.append(x0s)

            # Train the GRNN models
            for i, model in enumerate(grnn_models):
                model.zero_grad()
                xtraj, utraj = model.forward(x0s, dlqrenv.step)
                error = grnn_criterion(xtraj, utraj, dlqrenv, model)
                loss = torch.sum(error) / batch_size
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizers[i].step()
                schedulers[i].step()

            # Train the GCNN model
            gcnn_model.zero_grad()
            xtraj, utraj = gcnn.get_trajectory(gcnn_model, dlqrenv, x0s, T)
            error = gcnn_criterion(xtraj, utraj, dlqrenv, None)
            loss = torch.sum(error) / batch_size
            loss.backward()
            gcnn_optimizer.step()
            if(epoch % gcnn_decay_interval == 0):
                gcnn_scheduler.step()

            if(epoch % log_interval == 0):
                log_ind = int(epoch / log_interval)
                controllers = [model.get_controller(validation_size) \
                               for model in grnn_models]
                controllers.append(gcnn.get_gcnn_controller(gcnn_model, N))
                costs = exp_utils.estimate_controller_cost(
                        dlqrenv, T, controllers, validation_size)
                rel_costs_table[counter, 0:4, log_ind] = costs[1:]

        # Model 5: Sparsified GRNN
        dense_model = exp_utils.generate_model(model_params, dlqrenv,
                use_given_support=False, S=None,
                criterion=sparse_criterion,
                **training_params)
        new_S = dense_model.S.clone()
        new_S[torch.abs(new_S) < threshold] = 0
        num_sparse_edges[counter] = torch.sum(new_S != 0).item()
        model_params['S_trainable'] = True
        sparse_model = grnn.GRNN(new_S, **model_params).to(device)
        sparse_optimizer = optim.Adam(sparse_model.parameters(), lr=0.02)
        sparse_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                sparse_optimizer, num_epoch)
        for epoch in range(num_epoch):
            x0s = train_x0s[epoch]
            # Train the GRNN models
            sparse_model.zero_grad()
            xtraj, utraj = sparse_model.forward(x0s, dlqrenv.step)
            error = grnn_criterion(xtraj, utraj, dlqrenv, sparse_model)
            loss = torch.sum(error) / batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sparse_model.parameters(), 10)
            sparse_optimizer.step()
            sparse_scheduler.step()
            if(epoch % log_interval == 0):
                log_ind = int(epoch / log_interval)
                controllers = [sparse_model.get_controller(validation_size)]
                costs = exp_utils.estimate_controller_cost(
                        dlqrenv, T, controllers, validation_size)
                rel_costs_table[counter, 4, log_ind] = costs[1]

        # Print progress
        if verbose:
            print('Iteration: {}'.format(counter+1))
            print(rel_costs_table[counter, :, -1].detach().cpu().numpy())

    with open(filename, 'w') as f:
        result_dict = {
                'Anorm': A_norm,
                'num_sparse_edges': num_sparse_edges.cpu().numpy().tolist(),
                'num_env_edges': num_env_edges.cpu().numpy().tolist(),
                'auto_costs': autonomous_costs.cpu().numpy().tolist(),
                'cost_table': rel_costs_table.cpu().numpy().tolist()
        }
        f.write(json.dumps(result_dict))
#############################################################
