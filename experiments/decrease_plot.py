import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

def summarize(filename, log_interval, ci_percent=.75):
    # Load data
    with open(filename) as f:
        data = json.load(f)
    print('Sparse edges:', np.mean(data['num_sparse_edges']))
    print('Env edges:', np.mean(data['num_env_edges']))

    # Plot the loss decrease
    cost_table = np.array(data['cost_table'])
    print('Average Costs:', np.median(cost_table[:,:,-1], 0))
    print('Autonomous Costs:', np.mean(data['auto_costs']))
    names = ['GRNN-Fixed', 'GRNN', 'GRNN-Full', 'GCNN', 'GRNN-Sparse']
    for i in [0,1,2,4,3]:
        if i == 0:
            continue
        cost = cost_table[:,i,:]
        median_cost = np.median(cost, axis=0)
        lower_ci = np.quantile(cost, 1-ci_percent, axis=0)
        upper_ci = np.quantile(cost, ci_percent, axis=0)
        ind = np.arange(len(median_cost)) * log_interval
        p = plt.semilogy(ind, median_cost, label=names[i])
        color = p[0].get_color()
        plt.fill_between(ind, lower_ci, upper_ci, color=color, alpha=0.3)

    plt.title(r'Performance vs. Number of Batches ($\|A\|_2={})$'.format(
        data['Anorm']))
    plt.xlabel('Number of Batches')
    plt.ylabel('Normalized Cost')
    plt.ylim([1,4.5])
    plt.legend()
    plt.show()
