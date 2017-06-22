import nengo
import grid
import nengo_xbox
import numpy as np

map = '''
#####
#   #
#   #
#   #
#####
'''

class Cell(grid.Cell):
    def load(self, c):
        if c == '#':
            self.wall = True
    
    def color(self):
        if self.wall:
            return 'black'
            
            
class Prey(grid.ContinuousAgent):
    color = 'green'
    shape = 'circle'
            
world = grid.World(Cell, map=map, directions=4)
body = grid.ContinuousAgent()
world.add(body, x=2, y=2, dir=1)

prey = Prey()
world.add(prey, x=2, y=3)


class FacingReward(nengo.Node):
    def __init__(self, body, target):
        self.body = body
        self.target = target
        self.last_t = None
        self.last_theta = None
        super(FacingReward, self).__init__(self.update)
    def update(self, t):
        if self.last_t is not None and t < self.last_t:
            self.last_t = None
            
        angle = (body.dir-1) * 2 * np.pi / 4
        food_angle = np.arctan2(self.target.y - self.body.y,
                              self.target.x - self.body.x)
                              
        theta = angle - food_angle
        if theta > np.pi:
            theta -= np.pi * 2
        if theta < -np.pi:
            theta += np.pi * 2
        
        theta = np.abs(theta)

        value = 0
        if self.last_t is not None:
            if theta < self.last_theta:
                value = 1
            elif theta > self.last_theta:
                value = -1
        else:
            value = 0
        self.last_t = t
        self.last_theta = theta
        
        return value
            
class State(nengo.Node):
    def __init__(self, body, food):
        self.body = body
        self.food = food
        super(State, self).__init__(self.update)
        
    def update(self, t):
        angle = (body.dir-1) * 2 * np.pi / 4
        dy = self.food.y - self.body.y
        dx = self.food.x - self.body.x
        food_angle = np.arctan2(dy, dx)
        theta = angle - food_angle
        dist2 = dy**2 + dx**2
        
        if dist2 == 0:
            r = 1.0
        else:
            r = min(0.5 / dist2, 1.0)
        return -np.sin(theta)*r, np.cos(theta)*r
        


        


model = nengo.Network()
with model:
    env = grid.GridNode(world)
    
    speed = nengo.Node(lambda t, x: body.go_forward(x[0]*0.01), size_in=1)
    turn = nengo.Node(lambda t, x: body.turn(x[0]*0.01), size_in=1)
    
    facing_reward = FacingReward(body, prey)
    
    state = State(body, prey)
    
    s = nengo.Ensemble(n_neurons=200, dimensions=2, intercepts=nengo.dists.Uniform(0,1))
    nengo.Connection(state, s, synapse=None)
    
    
    q = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=2)
    
    noise = nengo.Node(nengo.processes.WhiteSignal(period=10.0, high=0.5, 
                        rms=0.1), size_out=2)
    nengo.Connection(noise, q.output)
    
    conn = nengo.Connection(s, q.input, function=lambda x: [0,0],
                            learning_rule_type=nengo.PES(learning_rate=1e-4))
    
    nengo.Connection(q.output[0], turn, transform=-1, synapse=0.1)
    nengo.Connection(q.output[1], turn, transform=1, synapse=0.1)
    
    error = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=2)
    nengo.Connection(q.output, error.input)

    def compute_target(t, x):
        reward, left, right = x
        if left > 0:
            t_left = reward
        else:
            t_left = -reward
        if right > 0:
            t_right = reward
        else:
            t_right = -reward
            
        return t_left, t_right
    target = nengo.Node(compute_target, size_in=3)
    nengo.Connection(facing_reward, target[0])
    nengo.Connection(q.output, target[1:])
    nengo.Connection(target, error.input, transform=-1)


    nengo.Connection(error.output, conn.learning_rule)
    

    
    