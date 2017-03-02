import nengo
import numpy as np

class Pendulum(object):
    def __init__(self, mass=1.0, length=1.0, dt=0.001, g=10.0, seed=None,
                 max_torque=2, max_speed=8):
        self.mass = mass
        self.length = length
        self.dt = dt
        self.g = g
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.reset(seed)
        
    def reset(self, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.theta = self.rng.uniform(-np.pi, np.pi)
        self.dtheta = self.rng.uniform(-1, 1)
        
    def step(self, u):
        u = np.clip(u, -1, 1) * self.max_torque

        self.dtheta += (-3*self.g/(2*self.length)*np.sin(self.theta+np.pi) + 
                         3./(self.mass*self.length**2)*u) * self.dt
        self.theta += self.dtheta * self.dt
        self.dtheta = np.clip(self.dtheta, -self.max_speed, self.max_speed)
        
        self.theta = (self.theta + np.pi) % (2*np.pi) - np.pi
        
    def generate_html(self, desired):
        len0 = 40*self.length
        x1 = 50
        y1 = 50
        x2 = x1 + len0 * np.sin(self.theta)
        y2 = y1 - len0 * np.cos(self.theta)
        x3 = x1 + len0 * np.sin(desired)
        y3 = y1 - len0 * np.cos(desired)
        return '''
        <svg width="100%" height="100%" viewbox="0 0 100 100">
            <line x1="{x1}" y1="{y1}" x2="{x3}" y2="{y3}" style="stroke:blue"/>
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="stroke:black"/>
        </svg>
        '''.format(**locals())
        


class PendulumNode(nengo.Node):
    def __init__(self, **kwargs):
        self.env = Pendulum(**kwargs)
        def func(t, x):
            self.env.step(x[0])
            func._nengo_html_ = self.env.generate_html(desired=x[1])
            return self.env.theta, np.sin(self.env.theta), np.cos(self.env.theta), self.env.dtheta
        super(PendulumNode, self).__init__(func, size_in=2)
        
class PID(object):
    def __init__(self, Kp, Ki_scale, Kd_scale, dimensions=1, dt=0.001):
        self.Kp = Kp
        self.Ki = Ki_scale*Kp
        self.Kd = Kd_scale*Kp
        self.dt = dt
        self.last_error = np.zeros(dimensions)
        self.sum_error = np.zeros(dimensions)
    def step(self, error):
        self.sum_error += error * self.dt
        derror = (error - self.last_error) / self.dt
        self.last_error = error
        return self.Kp*error + self.Ki*self.sum_error + self.Kd * derror

class PIDNode(nengo.Node):
    def __init__(self, dimensions, **kwargs):
        self.pid = PID(dimensions=dimensions, **kwargs)
        super(PIDNode, self).__init__(self.step, size_in=dimensions)
    def step(self, t, x):
        x = (x[0] + np.pi) % (2*np.pi) - np.pi
        return self.pid.step(x)
        

model = nengo.Network()
with model:
    env = PendulumNode(seed=1, mass=4, max_torque=100)
        
    desired = nengo.Node([0])
    nengo.Connection(desired, env[1], synapse=None)
    
    pid = PIDNode(dimensions=1, Kp=1, Kd_scale=1, Ki_scale=0)
    nengo.Connection(pid, env[0], synapse=None)
    nengo.Connection(desired, pid, synapse=None, transform=1)
    nengo.Connection(env[0], pid, synapse=0, transform=-1)
    
    state = nengo.Ensemble(n_neurons=500, dimensions=2,
                           intercepts=nengo.dists.Uniform(0.2,1.0),
                           )
    nengo.Connection(env[1:3], state, synapse=None)
    
    c = nengo.Connection(state, env[0], synapse=0.01,
                         function=lambda x: 0,
                         learning_rule_type=nengo.PES(learning_rate=1e-4))
    nengo.Connection(pid, c.learning_rule, synapse=None, transform=-1)                        
                         
    