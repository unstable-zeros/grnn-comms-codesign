import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from controller import GRNNController

class GRNN(nn.Module):
    """ A simplified version of the Graph Recurrent Neural Network
    """
    def __init__(self, S, N, T, p=1, q=1, h=5, S_trainable=True):
        """ Constructor
        Parameters:
            - S:            None/torch.tensor(N,N), the communication topolgy
                            if is None, then the topology will be treated as a
                            parameter to be optimized
            - N:            integer, number of agents
            - T:            integer, time horizon
            - p:            integer, state dimension of each agent
            - q:            integer, control dimension of each agent
            - h:            integer, hidden state dimension of each agent
            - S_trainable:  bool, indicates whether we can optimize the entries
                            in S
        """
        super().__init__()
        self.N, self.T, self.p, self.q, self.h = N, T, p, q, h
        # Initialize S
        if S is None:
            # If S is none, we treat the graph topology as a parameter we want
            # to optimize
            initial_S = torch.rand((N,N), dtype=torch.double)
            initial_S = initial_S / initial_S.sum(1)[:,None]
            self.register_parameter(name='S',
                    param=torch.nn.Parameter(initial_S))
            self.S_entries = None
            self.S_inds = None
        else:
            # If S is given, we do not design the topology
            if S_trainable:
                self.S_inds = (S != 0)
                self.register_parameter(name='S_entries',
                        param=torch.nn.Parameter(S[self.S_inds]))
            else:
                self.S = S
                self.S_entries = None
                self.S_inds = None

        A = np.random.random((p,h))
        B = np.random.random((h,q))
        W = np.random.random((h,h))
        A = A / np.linalg.norm(A, ord=2)
        B = B / np.linalg.norm(B, ord=2)
        W = W / np.linalg.norm(W, ord=2)
        self.register_parameter(name='A',
                param=torch.nn.Parameter(torch.tensor(A, dtype=torch.double)))
        self.register_parameter(name='B',
                param=torch.nn.Parameter(torch.tensor(B, dtype=torch.double)))
        self.register_parameter(name='W',
                param=torch.nn.Parameter(torch.tensor(W, dtype=torch.double)))


    def _graph_conv(self, X, Z):
        Z_new = torch.tanh( torch.matmul(self.S_(), Z) @ self.W + torch.matmul(X, self.A) )
        u = torch.matmul(Z_new, self.B)
        return Z_new, u

    def forward(self, x0, step):
        batch_size = x0.size(0)
        x_traj = self.A.new_empty((batch_size, self.T+1, self.N, self.p))
        u_traj = self.A.new_empty((batch_size, self.T, self.N, self.q))
        x_traj[:,0,:,:] = x0
        Z = self.A.new_zeros((batch_size, self.T+1, self.N, self.h))
        for t in range(self.T):
          # Something annoying about the in-place operations here with pytorch
          # Hence the .clone() for some of the tensors we are using
          xt = x_traj[:,t,:,:].clone()
          Zt, ut = self._graph_conv(xt, Z[:,t,:,:].clone())
          x_traj[:,t+1,:,:] = step(xt, ut)
          u_traj[:,t,:,:] = ut
          Z[:,t+1,:,:] = Zt
        return x_traj, u_traj

    def S_(self):
        """ This gives us the S matrix for different training configurations.
        Always use this instead of using self.S
        """
        if self.S_entries is None:
            return self.S
        else:
            #TODO: make this more efficient. Allocating a new array every time
            #seems very wastful
            S = self.S_entries.new_zeros((self.N, self.N))
            S[self.S_inds] = self.S_entries
            return S

    def get_params(self):
        return self.S_().detach().clone(),\
                self.A.detach().clone(),\
                self.B.detach().clone(),\
                self.W.detach().clone()

    def get_controller(self, batch_size):
        return GRNNController(self, batch_size)

