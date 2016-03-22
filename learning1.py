# The basic simplest learning model

import nengo
import numpy as np

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: np.sin(t*2*np.pi))
    
    pre = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(stim, pre)
    
    post = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    def init_func(x):
        return 0
    learn_conn = nengo.Connection(pre, post, function=init_func,
                                  learning_rule_type=nengo.PES())
                                  
    error = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    def desired_func(x):
        # adjust this to change what function is learned
        return x
    nengo.Connection(stim, error, function=desired_func, transform=-1)
    nengo.Connection(post, error, transform=1)
    
    nengo.Connection(error, learn_conn.learning_rule)
    
    stop_learn = nengo.Node(0)
    nengo.Connection(stop_learn, error.neurons, transform=-10*np.ones((100,1)))