import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import sys


def plot_frontier(num_edges, retrain_costs, Anorm):
    # Make Pareto frontier graph
    median_edges = np.median(num_edges, axis=0)
    median_costs = np.median(retrain_costs, axis=0)
    plt.plot(median_edges, median_costs,
            label=r'$\|A\|_2={}$'.format(Anorm), alpha=0.8)
    color = plt.gca().lines[-1].get_color()
    plt.scatter(median_edges, median_costs, c=color)
    #plt.scatter(num_edges[:], retrain_costs[:], s=10, marker='*', c=color, alpha=0.3)
    #yerr = retrain_costs.std(0)
    #xerr = num_edges.std(0)
    edge_lower = np.quantile(num_edges, 0.25, axis=0)
    edge_upper = np.quantile(num_edges, 0.75, axis=0)
    edge_err = np.vstack([median_edges - edge_lower, edge_upper - median_edges])
    cost_lower = np.quantile(retrain_costs, 0.25, axis=0)
    cost_upper = np.quantile(retrain_costs, 0.75, axis=0)
    cost_err = np.vstack([median_costs - cost_lower, cost_upper - median_costs])
    plt.errorbar(median_edges,median_costs, xerr=edge_err, yerr=cost_err,
            linestyle="None", alpha=0.3, c=color)

def summarize(datafilename):
    # Load dataset
    with open(datafilename) as f:
        data = f.readlines()
    for datum in data:
        datum = json.loads(datum)
        Anorm = datum['Anorm']
        num_edges = np.array(datum['num_edges'])
        reg_costs = np.array(datum['reg_costs'])
        retrain_costs = np.array(datum['retrain_cost'])
        env_edges = np.array(datum['env_edges'])
        betas = np.array(datum['betas'])
        avg_num_edges = num_edges.mean(0)
        avg_reg_costs = reg_costs.mean(0)
        avg_retrain_costs = retrain_costs.mean(0)
        average_edges = env_edges.mean()
        plot_frontier(num_edges, retrain_costs, Anorm)

    plt.semilogx()
    #plt.xlim([10, 330])
    #plt.ylim([1, 1.65])
    plt.xlabel('Number of Directed Edges')
    plt.ylabel('Normalized Cost')
    plt.legend()
    plt.show()

'''
# Make bar graph
fig = plt.figure()
ax = plt.gca()
ax.bar(betas, avg_num_edges, width=1*betas, log=True, alpha=0.5, color='tab:blue')
ax.plot(betas, [average_edges]*len(betas), '--')
ax.set_xscale('log')
ax.set_xlim([betas[0]/2, betas[-1]*2])
ax.set_yscale('linear')
ax.set_ylim([0, 300])
ax.set_xlabel('betas')
ax.set_ylabel('Number of Edges', color='tab:blue')
ax2 = ax.twinx()
ax2.plot(betas, avg_reg_costs, c='tab:red', label='Reg Cost')
ax2.plot(betas, avg_retrain_costs, c='y', label='Retrain Cost')
ax2.legend()
ax2.set_ylabel('LQR Cost', color='tab:red')
plt.show()
'''

'''
# Make tradeoff spread graph
plt.figure(figsize=(20, 10))
plt.title('Num Edges vs. Costs')
plt.subplot(2,4,1)
plt.scatter(num_edges.flatten(), reg_costs.flatten(), c='tab:blue', alpha=0.3,
        edgecolors='none')
ax = plt.gca()
plt.title('Overall')
plt.xlabel('Number of Edges')
plt.ylabel('Costs after Retrain')
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
for i in range(len(betas)):
    plt.subplot(2, 4, i+2)
    plt.scatter(num_edges[:,i], retrain_costs[:,i], c='tab:blue', alpha=0.5,
          edgecolors='none')
    plt.scatter(num_edges[:,i].mean(), retrain_costs[:,i].mean(), c='tab:red',
          marker='*', s=100)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel('Number of Edges')
    plt.ylabel('Costs after Retrain')
    plt.title(r'$\beta={}$'.format(betas[i]))
plt.show()
#plt.savefig('beta dist.png', dpi=200)
'''
