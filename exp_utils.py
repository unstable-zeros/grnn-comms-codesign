import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import grnn, env.dlqr
import gcnn

def train_it(model, env, criterion, batch_size, seed=None):
    x0s = env.random_x0(batch_size, seed)
    xtraj, utraj = model.forward(x0s, env.step)
    error = criterion(xtraj, utraj, env, model)
    loss = torch.sum(error) / batch_size
    return loss

def train_it_gcnn(model, env, criterion, batch_size, T, seed=None):
    x0s = env.random_x0(batch_size, seed)
    xtraj, utraj = gcnn.get_trajectory(model, env, x0s, T)
    error = criterion(xtraj, utraj, env, model)
    loss = torch.sum(error) / batch_size
    return loss

def train_model(model, env, criterion,
        batch_size=20, num_epoch=100, lr=0.03, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
              optimizer, num_epoch)
    for epoch in range(num_epoch):
        model.zero_grad()
        loss = train_it(model, env, criterion, batch_size=batch_size)
        if(verbose and epoch % 10 == 0):
            print('Epoch: {} \t Loss: {}'.format(epoch+1, loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        scheduler.step()
    return model

def generate_model(model_params, env, criterion, T, device,
        use_given_support, S=None, batch_size=20, num_epoch=100, ensemble_size=2,
        val_size=50, verbose=False):
    # Generate graph support
    assert (not use_given_support) or \
            (use_given_support and (not S is None))
    if not use_given_support:
        S = None

    # Train multiple models
    models, costs = [], []
    for _ in range(ensemble_size):
        model = grnn.GRNN(S, **model_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.03)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                  optimizer, num_epoch)
        for epoch in range(num_epoch):
            model.zero_grad()
            loss = train_it(model, env, criterion, batch_size=batch_size)
            if(verbose and epoch % 10 == 0):
                print('Epoch: {} \t Loss: {}'.format(epoch+1, loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            scheduler.step()
        models.append(model)
        costs.append(estimate_controller_cost(env, T,
            [model.get_controller(val_size)], num_x0s=val_size)[1])
    return models[np.argmin(costs)]

def generate_gcnn_model(S, N, env, criterion, device, T,
        batch_size=20, num_epoch=100, ensemble_size=2, val_size=50, verbose=False):
    import gcnn
    # Train multiple models
    models, costs = [], []
    for _ in range(ensemble_size):
        model = gcnn.generate_model(S, device)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                  optimizer, num_epoch)
        for epoch in range(num_epoch):
            model.zero_grad()
            loss = train_it_gcnn(model, env, criterion, batch_size, T)
            if(verbose and epoch % 10 == 0):
                print('Epoch: {} \t Loss: {}'.format(epoch+1, loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            scheduler.step()
        models.append(model)
        costs.append(estimate_controller_cost(env, T,
            [gcnn.get_gcnn_controller(model, N)], num_x0s=val_size)[1])
    return models[np.argmin(costs)]

def grnn_topology(env, model_params, training_params, verbose=False):
    model = generate_model(model_params, env, use_given_support=False,
            **training_params)
    return model.S_().detach(), model

def sim_controllers(env, x0s, controllers, T, device, rel_cost=True):
    """ Simulates a list of controllers for one episode
    """
    xs, us, costs = [], [], []
    for c in controllers:
        x, u = env.sim_forward(c, T, x0s=x0s)
        cost = env.cost(x, u)
        xs.append(x.detach()); us.append(u.detach())
        costs.append(cost.detach())
    xs = torch.stack(xs, dim=0)
    us = torch.stack(us, dim=0)
    costs = torch.stack(costs, dim=0)
    if rel_cost:
        optimal_cost = torch.diag(env.P).sum()
        costs_to_ret = costs.mean(1)
        costs_to_ret = costs_to_ret / optimal_cost
    else:
        costs_to_ret = costs.mean(1)
    return xs, us, costs_to_ret

def relative_costs(costs, rel_cost_wrt):
    """ Normalize the cost by dividing all costs by one single cost
    Parameters:
        - costs:            np.array, costs for different controllers
        - rel_cost_wrt:     int, the index w.r.t. which we normalize the cost
    Return:
        - rel_costs:        the normalized costs
    """
    return costs / costs[rel_cost_wrt]

def plot_controllers(xs, names, costs, rel_cost_wrt=None, dim2plot=0):
    """ Plot trajectories
    Parameters:
        - rel_cost_wrt:     if is None, print absolute cost, o.w. normalize
        - dim2plot:         which dimension of states to plot
    Return:
        - None
    """
    num_controllers, T, N, p = xs.shape
    costs_to_print = costs.clone()
    if rel_cost_wrt is not None:
        costs_to_print = relative_costs(costs_to_print, rel_cost_wrt)
    plt.figure(figsize=(num_controllers * 5, 5))
    for i in range(num_controllers):
        plt.subplot(1, num_controllers, i+1)
        for j in range(N):
          plt.plot(np.arange(T), xs[i, :, j, dim2plot])
          plt.title(names[i]+'\nCost={:.3f}'.format(costs_to_print[i]))

def estimate_controller_cost(env, T, controllers, num_x0s=100):
    """ Estimate the cost of GRNN on the given environment and compare to other
    controllers.
    """
    optctrl = env.get_optimal_controller()
    controllers = [optctrl] + controllers
    x0s = env.random_x0(num_x0s)
    _, _, costs = sim_controllers(env, x0s, controllers, T, x0s.device)
    return costs
