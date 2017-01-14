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
        p2 = 0.5
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
            if self.rng.rand()<p2:
                self.action.append('A1')
                self.reward.append(0)
            else:
                self.action.append('A2')
                self.reward.append(1)
        
        
time_step=0.1
task = Task(seed=0,
            time_step=time_step,
            vocab_state=spa.Vocabulary(3, randomize=False),
            vocab_action=spa.Vocabulary(2, randomize=False))
model = nengo.Network()
with model:
    state = nengo.Node(task.state_func)
    action = nengo.Node(task.action_func)
    reward = nengo.Node(task.reward_func)
    
    do_learn_next_state = True
    if do_learn_next_state:
        learn_next = nengo.Network()
        with learn_next:
            state_action = nengo.Ensemble(n_neurons=1000, dimensions=5,
                                          intercepts=nengo.dists.Uniform(-0.1, 0.5),
                                          radius=1.4)
            
            next_state = nengo.Ensemble(n_neurons=100, dimensions=3)
            c = nengo.Connection(state_action, next_state,
                                    synapse=None,
                                    function=lambda x: np.zeros(3),
                                    learning_rule_type=nengo.PES(pre_synapse=z**(-int(time_step/0.001)),
                                                                 learning_rate=3e-5))
                                                                 
            nengo.Connection(next_state, c.learning_rule, synapse=z**(-int(time_step/0.001)))
        nengo.Connection(state, c.learning_rule, transform=-1, synapse=None)
        nengo.Connection(state, state_action[:3], synapse=None)
        nengo.Connection(action, state_action[3:], synapse=None)
        
    do_learn_value = True
    
    if do_learn_value:
        learn_value = nengo.Network()
        with learn_value:
            context = nengo.Ensemble(n_neurons=100, dimensions=1)
            value = nengo.Ensemble(n_neurons=1000, dimensions=3)
            
            c = nengo.Connection(context, value,
                                 function=lambda x: np.zeros(3),
                                 learning_rule_type=nengo.PES())
            
            v_dot_s = nengo.networks.Product(n_neurons=100, dimensions=3)
            v_dot_s.label = 'v_dot_s'
            nengo.Connection(value, v_dot_s.A, synapse=None)
            
            e_dot_s = nengo.networks.Product(n_neurons=100, dimensions=3)
            e_dot_s.label = 'e_dot_s'
            nengo.Connection(v_dot_s.output, e_dot_s.A, transform=np.ones((3,3)))
            nengo.Connection(e_dot_s.output, c.learning_rule)
            
        nengo.Connection(state, v_dot_s.B, synapse=None)
        nengo.Connection(state, e_dot_s.B, synapse=None)
        
        nengo.Connection(reward, e_dot_s.A, transform=-1*np.ones((3,1)))
        
    p_dot_v = nengo.networks.Product(n_neurons=100, dimensions=3)
    p_dot_v.label = 'p_dot_v'
    
    q = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(p_dot_v.output, q, transform=np.ones((1,3)))
    nengo.Connection(v_dot_s.output, q, transform=np.ones((1,3)))
    nengo.Connection(next_state, p_dot_v.A)
    nengo.Connection(value, p_dot_v.B)
        
    
