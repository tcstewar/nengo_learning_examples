import nengo
import numpy as np

model = nengo.Network()
model.config[nengo.Ensemble].neuron_type=nengo.Direct()
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
    nengo.Connection(x, dx, synapse=0.01, transform=scale_x)
    nengo.Connection(x, dx, synapse=0.02, transform=-scale_x)
    
    prod = nengo.networks.Product(100, 1)
    nengo.Connection(dx, prod.A)
    nengo.Connection(dy, prod.B)
    nengo.Connection(prod.output, x, synapse=0.01, transform=0.5)
    #for ens in prod.all_ensembles:
    #    ens.neuron_type = nengo.Direct()

    
    stim_x = nengo.Node(0)
    nengo.Connection(stim_x, x)
    nengo.Connection(x, x, synapse=0.01)
    
    
    