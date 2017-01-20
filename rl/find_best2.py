import nengo
import numpy as np

model = nengo.Network()
#model.config[nengo.Ensemble].neuron_type=nengo.Direct()
with model:
    
    x = nengo.Ensemble(100, 1, neuron_type=nengo.LIF())
    
    y = nengo.Ensemble(100, 1, neuron_type=nengo.LIF())
    
    nengo.Connection(x, y,
            function=lambda x: 1-4*(x-0.5)**2)
            
    
    scale_y = 20
    dy = nengo.Ensemble(100, 1)
    nengo.Connection(y, dy, synapse=0.001, transform=scale_y)
    nengo.Connection(y, dy, synapse=0.02, transform=-scale_y)

    scale_x = 5
    dx = nengo.Ensemble(100, 1)
    nengo.Connection(dx, dx, synapse=0.1)
    nengo.Connection(dx, x, transform=0.1)

    prod = nengo.networks.Product(100, 1)
    nengo.Connection(dx, prod.A)
    nengo.Connection(dy, prod.B)
    nengo.Connection(prod.output, dx, synapse=0.01, transform=0.5)

    
    stim_x = nengo.Node(0)
    nengo.Connection(stim_x, x)
    nengo.Connection(x, x, synapse=0.01)
    
    
    