import nengo
import numpy as np

class Pendulum(object):
    def __init__(self, mass=1.0, length=1.0, dt=0.001, g=10.0, seed=None,
                 max_torque=2, max_speed=8, limit=2.0, bounds=None):
        self.mass = mass
        self.length = length
        self.dt = dt
        self.g = g
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.limit = limit
        self.extra_mass = 0
        self.bounds = bounds
        self.reset(seed)
        
    def reset(self, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.theta = self.rng.uniform(-self.limit, self.limit)
        self.dtheta = self.rng.uniform(-1, 1)
        
    def step(self, u):
        u = np.clip(u, -1, 1) * self.max_torque

        mass = self.mass + self.extra_mass
        self.dtheta += (-3*self.g/(2*self.length)*np.sin(self.theta+np.pi) + 
                         3./(mass*self.length**2)*u) * self.dt
        self.theta += self.dtheta * self.dt
        self.dtheta = np.clip(self.dtheta, -self.max_speed, self.max_speed)
        
        if self.bounds:
            self.theta = np.clip(self.theta, self.bounds[0], self.bounds[1])
        self.theta = (self.theta + np.pi) % (2*np.pi) - np.pi
        
    def set_extra_mass(self, mass):
        self.extra_mass = mass
        
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
        
class PendulumNetwork(nengo.Network):
    def __init__(self, label=None, **kwargs):
        super(PendulumNetwork, self).__init__(label=label)
        self.env = Pendulum(**kwargs)
        
        with self:
            def func(t, x):
                self.env.set_extra_mass(x[2])
                self.env.step(x[0])
                func._nengo_html_ = self.env.generate_html(desired=x[1])
                return (self.env.theta, self.env.dtheta)
            self.pendulum = nengo.Node(func, size_in=3)
            
            self.q_target = nengo.Node(None, size_in=1)
            nengo.Connection(self.q_target, self.pendulum[1], synapse=None)

            
            self.u = nengo.Node(None, size_in=1)
            nengo.Connection(self.u, self.pendulum[0], synapse=0)
            self.u_extra = nengo.Node(None, size_in=1)
            nengo.Connection(self.u_extra, self.pendulum[0], synapse=0)
            
            self.q = nengo.Node(None, size_in=1)
            self.dq = nengo.Node(None, size_in=1)
            nengo.Connection(self.pendulum[0], self.q, synapse=None)
            nengo.Connection(self.pendulum[1], self.dq, synapse=None) 
            
            self.extra_mass = nengo.Node(None, size_in=1)
            nengo.Connection(self.extra_mass, self.pendulum[2], synapse=None)

    
model = nengo.Network(seed=3)
with model:
    env = PendulumNetwork(mass=4, max_torque=100, seed=1)
    
    def sine_wave(t):
        return np.sin(t*np.pi)
    q_target = nengo.Node(sine_wave)
    nengo.Connection(q_target, env.q_target, synapse=None)
    
    dq_target = nengo.Node(None, size_in=1)
    nengo.Connection(q_target, dq_target, synapse=None, transform=1000)
    nengo.Connection(q_target, dq_target, synapse=0, transform=-1000)
    
    
    context = nengo.Ensemble(n_neurons=500, dimensions=3, radius=2,
                             intercepts=nengo.dists.Uniform(-0.1, 0.9))
    
    
    nengo.Connection(q_target, context[0], synapse=None)
    nengo.Connection(env.q, context[1], synapse=None)
    nengo.Connection(dq_target, context[2], synapse=None, transform=1)
    nengo.Connection(env.dq, context[2], synapse=None, transform=-1)

    def pd(x):
        q_target, q, dq_diff = x
        
        Kp = 1.0
        Kd = 0.2
        
        return Kp*(q_target-q) + Kd*(dq_diff)
    
    c = nengo.Connection(context, env.u, function=pd,
                         synapse=None,
                         learning_rule_type=nengo.PES(learning_rate=1e-4))
                         
    nengo.Connection(context, c.learning_rule, function=pd, transform=-1)
    

    import nengo_learning_display
    
    domain = np.zeros((30, 3))
    domain[:,1] = np.linspace(-1, 1, 30)
    domain[:,0] = np.linspace(-1, 1, 30)
    
    learned = nengo_learning_display.Plot1D(c, domain=domain,
                                            range=(-1,1))
                                            
                                            
def on_step(sim):
    learned.update(sim)