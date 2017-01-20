import nengo
import numpy as np

model = nengo.Network()
#model.config[nengo.Ensemble].neuron_type=nengo.Direct()
with model:
    
    x = nengo.Ensemble(100, 1, neuron_type=nengo.LIF())
    
    y = nengo.Ensemble(100, 1, neuron_type=nengo.LIF())
    
    nengo.Connection(x, y,
            function=lambda x: (2-4*(x-0.5)**2)*0.5)
            
    dx = nengo.Ensemble(n_neurons=2000, dimensions=1)
    dy = nengo.Ensemble(n_neurons=2000, dimensions=1)

    nengo.Connection(x, dx, synapse=0.02, transform=-1)
    nengo.Connection(y, dy, synapse=0.02, transform=-1)
    nengo.Connection(x, dx, synapse=0.01)
    nengo.Connection(y, dy, synapse=0.01)
    
    stim_x = nengo.Node(0)
    nengo.Connection(stim_x, x)
    
    nengo.Connection(x, x, synapse=0.1)

    signs = nengo.Ensemble(n_neurons=200, dimensions=2)
    nengo.Connection(dx, signs[0], function = lambda x: 1 if x>0 else -1)
    nengo.Connection(dy, signs[1], function = lambda x: 1 if x>0 else -1)
    
    nengo.Connection(signs, x, function=lambda x: x[0]*x[1])
    
    
    

    
    