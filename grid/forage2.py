seed = 1
import numpy as np
import random
random.seed(seed)
np.random.seed(seed=seed)


import nengo
import grid

map = '''
#######
#     #
#     #
#     #
#     #
#     #
#######
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
        eps = 1e-5
        if self.last_t is not None:
            if theta < self.last_theta-eps:
                value = 1
            elif theta > self.last_theta-eps:
                value = -1
        else:
            value = -1
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
            r = np.clip(0.5 / dist2, 0.2, 1.0)
            
        x = self.body.x / world.width * 2 - 1
        y = self.body.y / world.height * 2 - 1
        cos_a = np.cos(self.body.dir * np.pi * 2 / world.directions)
        sin_a = np.sin(self.body.dir * np.pi * 2 / world.directions)
            
        return [-np.sin(theta)*r, np.cos(theta)*r,
                x, y, cos_a, sin_a]
        


        
D = 3

model = nengo.Network(seed=seed)
with model:
    env = grid.GridNode(world)
    
    speed = nengo.Node(lambda t, x: body.go_forward(x[0]*0.01), size_in=1)
    turn = nengo.Node(lambda t, x: body.turn(x[0]*0.003), size_in=1)
    ctrl_speed = nengo.Node([0.3])
    nengo.Connection(ctrl_speed, speed)
    
    facing_reward = FacingReward(body, prey)
    sensors = State(body, prey)
    
    state = nengo.Ensemble(n_neurons=200, dimensions=2, 
                           intercepts=nengo.dists.Uniform(-0.1,1))
    nengo.Connection(sensors[:2], state, synapse=None)
    
    allo_state = nengo.Ensemble(n_neurons=500, dimensions=4, radius=1.7,
                                intercepts=nengo.dists.Uniform(0.5, 1))
    nengo.Connection(sensors[2:], allo_state)
    
    pred_action = nengo.Node(None, size_in=D)
    allo_conn = nengo.Connection(allo_state, pred_action,
                                 function=lambda x: [0]*D,
                                 learning_rule_type=nengo.PES(learning_rate=1e-2,
                                                              pre_tau=0.05))
    
    

    bg = nengo.networks.BasalGanglia(D)
    thal = nengo.networks.Thalamus(D)
    nengo.Connection(bg.output, thal.input)
    
    motor_command = nengo.Node(None, size_in=D)
    nengo.Connection(thal.output, motor_command)

    q = nengo.Node(None, size_in=D)
    nengo.Connection(q, bg.input, transform=2)
    bias = nengo.Node(0.5, label='')
    nengo.Connection(bias, bg.input, transform=np.ones((D, 1)))
    
    conn = nengo.Connection(state, q, function=lambda x: [0]*D,
                            learning_rule_type=nengo.PES(learning_rate=1e-4,
                                                         pre_tau=0.05))
    nengo.Connection(motor_command[0], turn, transform=-1, synapse=0.1)
    nengo.Connection(motor_command[1], turn, transform=1, synapse=0.1)


    nengo.Connection(pred_action, allo_conn.learning_rule)
    nengo.Connection(thal.output, allo_conn.learning_rule, transform=-1)
    
    nengo.Connection(pred_action, motor_command, transform=0.2)


    def target(t, x):
        index = np.argmax(x[1:])
        r = x[0]
        
        if index in [0,1]:
            r = r - 0.3
            
        
        result = np.ones(D) * -r
        result[index] = r
        return result
    target = nengo.Node(target, size_in=D+1)

    nengo.Connection(motor_command, target[1:])
    nengo.Connection(facing_reward, target[0])
    
    error = nengo.Node(None, size_in=D)
    
    nengo.Connection(q, error)
    nengo.Connection(target, error, transform=-1)

    nengo.Connection(error, conn.learning_rule)
    
    
    facing_reward.label = 'Reward (improved orientation)'
    sensors.label = 'Sensors'
    state.label = 'Cortex'
    turn.label = 'Motor Actions'
    q.label = 'Q'
    target.label = 'Learning Target'
    error.label = 'Error'
    for net in bg.networks:
        net.label = ''
    for node in bg.nodes:
        node.label = ''
    for net in thal.networks:
        net.label = ''
    for node in thal.all_nodes:
        node.label = ''
    for ens in thal.all_ensembles:
        ens.label = ''
    
    
    positions = [(2,2), (4,2), (4,4), (2,4)]
    pos_counter = 0
    def move_prey(t):
        global pos_counter
        dy = prey.y - body.y
        dx = prey.x - body.x
        dist2 = dx**2 + dy**2
        
        
        while dist2 < 0.25:
            prey.x, prey.y = positions[pos_counter]
            pos_counter = (pos_counter + 1) % len(positions)
            #prey.x = np.random.uniform(1, world.width-2)
            #prey.y = np.random.uniform(1, world.height-2)
            dy = prey.y - body.y
            dx = prey.x - body.x
            dist2 = dx**2 + dy**2
    move_prey = nengo.Node(move_prey)
            

    import nengo_learning_display
    theta = np.linspace(-np.pi, np.pi, 30)
    domain = np.array([np.sin(theta), np.cos(theta)]).T
    learned_far = nengo_learning_display.Plot1D(conn, domain=domain*0.3, 
                                                range=(-1.0,1.0))
    learned_near = nengo_learning_display.Plot1D(conn, domain=domain*1.0, 
                                                 range=(-1.0,1.0))
    learned_far.label = '&nbsp;&nbsp;&nbsp;&nbsp;Learned Action Utilities (far target)'
    learned_near.label = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Learned Action Utilities (near target)'
    
def on_step(sim):
    learned_far.update(sim)
    learned_near.update(sim)
