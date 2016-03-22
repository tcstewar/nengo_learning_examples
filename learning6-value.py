# Predicted Value Learning

# Here, we learn to predict future rewards.  The idea is to learn the
# function V(s) where V(s) = R(t) + 0.9*R(t+1) + 0.9**2*R(t+2) + ...

# You can do this by substitution, getting V(s,t) = R(t) + 0.9*V(s+1, t+1).
# This suggests an error rule: Error(t) = V(t-1) - R(t-1) - 0.9*V(t)
# We convert this to a continuous domain by using a long time constant for t-1
# and a short time constant for t.

# In the model below, the agent ALWAYS MOVES RANDOMLY.  It is not *using* what
# it learns to change its movement.  The goal here is simply to anticipate
# future rewards.  Connecting this learning to the action selection system
# as in the previous example is left as an exercise for the reader.

# The agent is given a reward whenever it is in the green square, and a 
# punishment (negative reward) whenever it is in the red square.  After some
# time, it should learn to start anticipating this reward/punishment as
# shown in the value graph.  In particular, the value graph should drop to a
# low value when it turns right at the top of the T-maze, and it should
# go to a high value when it turns left at the top of the T-maze.

import nengo
import numpy as np    

# requires CCMSuite https://github.com/tcstewar/ccmsuite/
import ccm.lib.grid
import ccm.lib.continuous
import ccm.ui.nengo

mymap="""
#######
#     #
# # # #
# # # #
#G   B#
#######
"""

class Cell(ccm.lib.grid.Cell):
    def color(self):
        if self.wall:
            return 'black'
        elif self.reward > 0:
            return 'green'
        elif self.reward < 0:
            return 'red'
        return None
    def load(self, char):
        if char == '#':
            self.wall = True
        self.reward = 0
        if char == 'G':
            self.reward = 10
        elif char == 'B':
            self.reward = -10

world = ccm.lib.grid.World(Cell, map=mymap, directions=4)

body = ccm.lib.continuous.Body()
world.add(body, x=1, y=2, dir=2)

def move(t, x):
    speed, rotation = x
    dt = 0.001
    max_speed = 20.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate)
    body.go_forward(speed * dt * max_speed)
    
    if int(body.x) == 1:
        world.grid[4][4].wall = True
        world.grid[4][2].wall = False
    if int(body.x) == 4:
        world.grid[4][2].wall = True
        world.grid[4][4].wall = False
    

def detect(t):
    angles = (np.linspace(-0.5, 0.5, 3) + body.dir ) % world.directions
    return [body.detect(d, max_distance=4)[0] for d in angles]


model = nengo.Network()
with model:
    movement = nengo.Node(move, size_in=2)

    env = ccm.ui.nengo.GridNode(world, dt=0.005)

    stim_radar = nengo.Node(detect)

    radar = nengo.Ensemble(n_neurons=50, dimensions=3, radius=4, seed=2,
                noise=nengo.processes.WhiteSignal(10, 0.1, rms=1))
    nengo.Connection(stim_radar, radar)


    def braiten(x):
        turn = x[2] - x[0]
        spd = x[1] - 0.5
        return spd, turn
    nengo.Connection(radar, movement, function=braiten)  
    
    def position_func(t):
        return body.x / world.width * 2 - 1, 1 - body.y/world.height * 2, body.dir / world.directions
    position = nengo.Node(position_func)
    
    state = nengo.Ensemble(100, 3)

    nengo.Connection(position, state, synapse=None)
    
    reward = nengo.Node(lambda t: body.cell.reward)
        
    tau=0.1
    value = nengo.Ensemble(n_neurons=50, dimensions=1)

    learn_conn = nengo.Connection(state, value, function=lambda x: 0,
                                  learning_rule_type=nengo.PES(learning_rate=1e-4,
                                                               pre_tau=tau))
    nengo.Connection(reward, learn_conn.learning_rule, 
                     transform=-1, synapse=tau)
    nengo.Connection(value, learn_conn.learning_rule, 
                     transform=-0.9, synapse=0.01)
    nengo.Connection(value, learn_conn.learning_rule, 
                     transform=1, synapse=tau)
        
        
        
        