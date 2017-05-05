# Reinforcement Learning

# Here we have a simple agent in a simple world.  It has three actions
# (go forward, turn left, and turn right), and its only sense are three
# range finders (radar).  

# Initially, its basal ganglia action selection system is set up to have
# fixed utilities for the three actions (moving forward has a utility of 0.8,
# which is larger than turning left or right (0.7 and 0.6), so it should always
# go forward.

# The reward system is set up to give a positive value when moving forward, and
# a negative value if the agent crashes into a wall.  In theory, this should
# cause it to learn to avoid obstacles.  In particular, it will start to turn
# before it hits the obstacle, and so the reward should be negative much less
# often.

# The error signal in this case is very simple: the difference between the
# computed utility and the instantaneous reward.  This error signal should
# only be applied to whatever action is currently being chosen (although it
# isn't quite perfect at doing this).  Note that this means it cannot learn
# to do actions that will lead to *future* rewards.

import nengo
import numpy as np    

# requires CCMSuite https://github.com/tcstewar/ccmsuite/
import ccm.lib.grid
import ccm.lib.continuous
import ccm.ui.nengo

mymap="""
#########
#       #
#       #
#   ##  #
#   ##  #
#       #
#########

"""

class Cell(ccm.lib.grid.Cell):
    def color(self):
        return 'black' if self.wall else None
    def load(self, char):
        if char == '#':
            self.wall = True

world = ccm.lib.cellular.World(Cell, map=mymap, directions=4)

body = ccm.lib.continuous.Body()
world.add(body, x=1, y=3, dir=2)

model = nengo.Network(seed=8)
with model:
    def move(t, x):
        speed, rotation = x
        dt = 0.001
        max_speed = 20.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        success = body.go_forward(speed * dt * max_speed)
        if not success:
            return 0
        else:
            return 0.4+speed

    movement = nengo.Ensemble(n_neurons=100, dimensions=2, radius=1.4)
    
    movement_node = nengo.Node(move, size_in=2, label='reward')
    nengo.Connection(movement, movement_node)

    env = ccm.ui.nengo.GridNode(world, dt=0.005)

    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir ) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    stim_radar = nengo.Node(detect)

    radar = nengo.Ensemble(n_neurons=50, dimensions=3, radius=4)
    nengo.Connection(stim_radar, radar)
    
    
    bg = nengo.networks.actionselection.BasalGanglia(3)
    thal = nengo.networks.actionselection.Thalamus(3)
    nengo.Connection(bg.output, thal.input)
    
    def u_fwd(x):
        return 0.8
    def u_left(x):
        return 0.6
    def u_right(x):
        return 0.7

    conn_fwd = nengo.Connection(radar, bg.input[0], function=u_fwd, learning_rule_type=nengo.PES())
    conn_left = nengo.Connection(radar, bg.input[1], function=u_left, learning_rule_type=nengo.PES())
    conn_right = nengo.Connection(radar, bg.input[2], function=u_right, learning_rule_type=nengo.PES())
        
    nengo.Connection(thal.output[0], movement, transform=[[1],[0]])
    nengo.Connection(thal.output[1], movement, transform=[[0],[1]])
    nengo.Connection(thal.output[2], movement, transform=[[0],[-1]])
    
    errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=3)
    nengo.Connection(movement_node, errors.input, transform=-np.ones((3,1)))
    nengo.Connection(bg.output[0], errors.ensembles[0].neurons, transform=np.ones((50,1))*4)    
    nengo.Connection(bg.output[1], errors.ensembles[1].neurons, transform=np.ones((50,1))*4)    
    nengo.Connection(bg.output[2], errors.ensembles[2].neurons, transform=np.ones((50,1))*4)    
    nengo.Connection(bg.input, errors.input, transform=1)
    
    nengo.Connection(errors.ensembles[0], conn_fwd.learning_rule)
    nengo.Connection(errors.ensembles[1], conn_left.learning_rule)
    nengo.Connection(errors.ensembles[2], conn_right.learning_rule)
    
    
    
    
    