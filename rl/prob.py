import nengo
import numpy as np
import nengo.spa as spa
from nengolib.signal import z

class Task(object):
    def __init__(self, seed, vocab_state, vocab_action, 
                 time_step=0.1, 
                 offset_state=0, offset_action=0, offset_reward=0,
                 ):
        self.rng = np.random.RandomState(seed=seed)
        self.state = []
        self.action = []
        self.reward = []
        self.time_step = time_step
        
        self.offset_state = offset_state
        self.offset_action = offset_action
        self.offset_reward = offset_reward
        
        self.vocab_state = vocab_state
        self.vocab_action = vocab_action
        vocab_state.parse('S0+S1+S2')
        vocab_action.parse('A1+A2')
        
        
    def state_func(self, t):
        index = int((t-self.offset_state)/self.time_step)
        while index >= len(self.state):
            self.generate_next()
        if index < 0:
            return self.vocab_state.parse('0').v
        else:
            return self.vocab_state.parse(self.state[index]).v
            
    def action_func(self, t):
        index = int((t-self.offset_action)/self.time_step)
        while index >= len(self.action):
            self.generate_next()
        if index < 0:
            return self.vocab_action.parse('0').v
        else:
            return self.vocab_action.parse(self.action[index]).v
            
    def reward_func(self, t):
        index = int((t-self.offset_reward)/self.time_step)
        while index >= len(self.reward):
            self.generate_next()
        if index < 0:
            return 0
        else:
            return self.reward[index]

    def generate_next(self):
        self.state.append('S0')
        p1 = 0.5
        p2 = 0.3
        if self.rng.rand()<p1:
            self.action.append('A1')
            self.reward.append(0)
            self.state.append('S1')
            if self.rng.rand()<p2:
                self.action.append('A1')
                self.reward.append(1)
            else:
                self.action.append('A2')
                self.reward.append(0)
        else:
            self.action.append('A2')
            self.reward.append(0)
            self.state.append('S2')
            if self.rng.rand()<1-p2:
                self.action.append('A1')
                self.reward.append(0)
            else:
                self.action.append('A2')
                self.reward.append(1)
        
        
time_step=0.1
delay_synapse = z**(-int(time_step/0.001))
task = Task(seed=0,
            time_step=time_step,
            vocab_state=spa.Vocabulary(3, randomize=False),
            vocab_action=spa.Vocabulary(2, randomize=False))
model = nengo.Network()
with model:
    stim_state = nengo.Node(task.state_func)
    stim_action = nengo.Node(task.action_func)


    state = nengo.Ensemble(500, 3)
    nengo.Connection(stim_state, state, synapse=None)
    
    model.actnets = []
    for a in range(2):
        actnet = nengo.Network()
        model.actnets.append(actnet)
        
        with actnet:
            actnet.prob = nengo.Ensemble(300, dimensions=3)
            actnet.error = nengo.Ensemble(300, dimensions=3)
            nengo.Connection(actnet.prob, actnet.error,
                             synapse=delay_synapse)
            actnet.inhibit = nengo.Ensemble(n_neurons=50, dimensions=1,
                                            encoders=nengo.dists.Choice([[1]]),
                                            intercepts=nengo.dists.Uniform(0.3, 1))
            nengo.Connection(actnet.inhibit, actnet.error.neurons,
                             transform=-2*np.ones((actnet.error.n_neurons, 1)))
        nengo.Connection(stim_state, actnet.error, transform=-1, synapse=None)
        c = nengo.Connection(state, actnet.prob,
                             function=lambda x: [0,0,0],
                             learning_rule_type=nengo.PES(pre_synapse=delay_synapse))
        nengo.Connection(actnet.error, c.learning_rule)
        
        t = np.ones((1,2))
        t[:,a] = 0
        nengo.Connection(stim_action, actnet.inhibit, transform=t)
        
        
    del actnet    
    
    