import nengo
import numpy as np

class Pendulum(object):
    def __init__(self, mass=1.0, length=1.0, dt=0.001, g=10.0, seed=None,
                 max_torque=2, max_speed=8, limit=2.0):
        self.mass = mass
        self.length = length
        self.dt = dt
        self.g = g
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.limit = limit
        self.reset(seed)
        
    def reset(self, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.theta = self.rng.uniform(-self.limit, self.limit)
        self.dtheta = self.rng.uniform(-1, 1)
        
    def step(self, u):
        u = np.clip(u, -1, 1) * self.max_torque

        self.dtheta += (-3*self.g/(2*self.length)*np.sin(self.theta+np.pi) + 
                         3./(self.mass*self.length**2)*u) * self.dt
        self.theta += self.dtheta * self.dt
        self.dtheta = np.clip(self.dtheta, -self.max_speed, self.max_speed)
        
        self.theta = np.clip(self.theta, -self.limit, self.limit)
        #self.theta = (self.theta + np.pi) % (2*np.pi) - np.pi
        
    
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
    def __init__(self, Kp, Ki, Kd, dimensions=1, dt=0.001):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.last_error = np.zeros(dimensions)
        self.sum_error = np.zeros(dimensions)
    def step(self, error, derror):
        self.sum_error += error * self.dt
        #derror2 = (error - self.last_error) / self.dt
        #if not np.allclose(derror, derror2):
        #    print derror, derror2
        #derror = derror2
        #self.last_error = error
        return self.Kp*error + self.Ki*self.sum_error + self.Kd * derror

class PIDNode(nengo.Node):
    def __init__(self, dimensions, **kwargs):
        self.dimensions = dimensions
        self.pid = PID(dimensions=dimensions, **kwargs)
        self.last_desired = None
        super(PIDNode, self).__init__(self.step, size_in=dimensions*4)
    def step(self, t, x):
        desired, actual = x[:self.dimensions], x[self.dimensions:self.dimensions*2]
        ddesired, dactual = x[self.dimensions*2:self.dimensions*3], x[self.dimensions*3:self.dimensions*4]
        diff = desired - actual
        #if self.last_desired is None or not np.allclose(desired, self.last_desired):
        #    diff *= 0
        #    print t, 'changed'
        #    self.last_desired = desired
        
        #print t, diff
        
        #if diff[0]>2 or diff[0]<-2:
        #    diff *=0
        #print t, x
        
        return self.pid.step(diff, ddesired-dactual)
        
class FunctionPlot(nengo.Node):
    def __init__(self, ens, pts):
        self.w = np.zeros((1, ens.n_neurons))
        if len(pts.shape)==1:
            pts.shape = pts.shape[0],1
        self.pts = pts
        self.ens = ens
        self.sim = None
        min_x = -2
        max_x = 2
        self.svg_x = (pts[:,0] - min_x) * 100 / (max_x - min_x)
        def plot(t):
            #plot._nengo_html_ = ''
            #return None
            if self.sim is not None:
                _, a = nengo.utils.ensemble.tuning_curves(self.ens, self.sim, self.pts)
            else:
                a = np.zeros((len(self.pts), self.ens.n_neurons))
            y = np.dot(a, self.w.T)
            
            min_y = -1.0
            max_y = 1.0
            data = (-y - min_y) * 100 / (max_y - min_y)

            paths = []
            # turn the data into a string for svg plotting
            path = []
            for j in range(len(data)):
                path.append('%1.0f %1.0f' % (self.svg_x[j], data[j]))
            paths.append('<path d="M%s" fill="none" stroke="blue"/>' %
                         ('L'.join(path)))

            plot._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
                %s
                <line x1=50 y1=0 x2=50 y2=100 stroke="#aaaaaa"/>
                <line x1=0 y1=50 x2=100 y2=50 stroke="#aaaaaa"/>
            </svg>
            ''' % (''.join(paths))     
        super(FunctionPlot, self).__init__(plot, size_in=0)
        
    
    
model = nengo.Network(seed=3)
with model:
    env = PendulumNode(seed=1, mass=4, max_torque=100)
        
    #desired = nengo.Node([0])
    desired = nengo.Node(lambda t: np.sin(t*np.pi))
    nengo.Connection(desired, env[1], synapse=None)

    pid = PIDNode(dimensions=1, Kp=1, Kd=0.2, Ki=0)
    nengo.Connection(pid, env[0], synapse=None)
    
    nengo.Connection(desired, pid[0], synapse=None, transform=1)
    nengo.Connection(env[0], pid[1], synapse=0, transform=1)
    nengo.Connection(env[3], pid[3], synapse=0, transform=1)
    
    nengo.Connection(desired, pid[2], synapse=None, transform=1000)
    nengo.Connection(desired, pid[2], synapse=0, transform=-1000)

    state = nengo.Ensemble(n_neurons=1000, dimensions=1, 
                           radius=1.5,
                           neuron_type=nengo.LIFRate(),
                           #intercepts=nengo.dists.Uniform(0.2,1.0),
                           )
    nengo.Connection(env[0], state, synapse=None)
    

    c = nengo.Connection(state, env[0], synapse=0,
                         function=lambda x: 0,
                         learning_rule_type=nengo.PES(learning_rate=1e-5))
    nengo.Connection(pid, c.learning_rule, synapse=None, transform=-1)                        
                       
                      
    plot = FunctionPlot(state, pts=np.linspace(-2, 2, 11))  
    p = nengo.Probe(c, 'weights')
                
            
  

def on_step(sim):
    if sim is None:
        return
    plot.sim = sim
    plot.w = sim._probe_outputs[p][-1]
    del sim._probe_outputs[p][:]
