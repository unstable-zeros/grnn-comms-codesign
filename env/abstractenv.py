class AbstractEnv:
    def __init__(self):
        pass
    def step(self, x, u):
        raise NotImplementedError
    def sim_forward(self, controller, x0=None):
        raise NotImplementedError
    def generate_x0(self):
        raise NotImplementedError
    def traj_cost(self, x):
        raise NotImplementedError
