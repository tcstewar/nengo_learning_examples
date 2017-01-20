import nengo
import numpy as np

model = nengo.Network()
#model.config[nengo.Ensemble].neuron_type=nengo.Direct()
with model:
    
    x = nengo.Ensemble(100, 1, neuron_type=nengo.LIF())
    
    y = nengo.Ensemble(100, 1, neuron_type=nengo.LIF())
    
    nengo.Connection(x, y,
            function=lambda x: 1-4*(x-0.5)**2)
            
            
    def deriv(q):
        x0, y0, x1, y1 = q
        dx = x1 - x0
        dy = y1 - y0
        
        if dx>0 and dy>0:
            return 1
        elif dx>0 and dy<0:
            return -1
        elif dx<0 and dy<0:
            return 1
        elif dx<0 and dy>0:
            return -1
        else:
            return 0

    dydx_node = nengo.Ensemble(n_neurons=2000, dimensions=4, radius=2, 
                               neuron_type=nengo.Direct(),
                               )
    nengo.Connection(x, dydx_node[0], synapse=0.02)
    nengo.Connection(y, dydx_node[1], synapse=0.02)
    nengo.Connection(x, dydx_node[2], synapse=0.01)
    nengo.Connection(y, dydx_node[3], synapse=0.01)
    
    stim_x = nengo.Node(0)
    nengo.Connection(stim_x, x)
    
    nengo.Connection(x, x, synapse=0.1)
    nengo.Connection(dydx_node, x, function=deriv, transform=0.1)
    
    
    

    
    