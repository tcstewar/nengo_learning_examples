import nengo
import numpy as np

model = nengo.Network()
#model.config[nengo.Ensemble].neuron_type=nengo.Direct()
with model:
    
    x = nengo.Ensemble(100, 1, neuron_type=nengo.LIF())
    
    y = nengo.Ensemble(100, 1, neuron_type=nengo.LIF())
    
    
    def func(t, x):
        x, offset = x
        return 1-(x-offset)**2
        
    stim_offset= nengo.Node(0)
    f = nengo.Node(func, size_in=2)
    nengo.Connection(x, f[0])
    nengo.Connection(stim_offset, f[1])
    
    nengo.Connection(f, y)

    dx = nengo.Ensemble(n_neurons=2000, dimensions=1)
    dy = nengo.Ensemble(n_neurons=2000, dimensions=1)

    nengo.Connection(x, dx, synapse=0.01, transform=-1)
    nengo.Connection(y, dy, synapse=0.01, transform=-1)
    nengo.Connection(x, dx, synapse=0.001)
    nengo.Connection(y, dy, synapse=0.001)
    
    stim_x = nengo.Node(0)
    nengo.Connection(stim_x, x)
    
    nengo.Connection(x, x, synapse=0.1)

    signs = nengo.Ensemble(n_neurons=200, dimensions=2)
    nengo.Connection(dx, signs[0], function = lambda x: 1 if x>0 else -1)
    nengo.Connection(dy, signs[1], function = lambda x: 1 if x>0 else -1)
    
    nengo.Connection(signs, x, function=lambda x: x[0]*x[1]*1.5, synapse=0.01)
    

    

    
    