import nengo
import numpy as np

model = nengo.Network()
#model.config[nengo.Ensemble].neuron_type=nengo.Direct()
with model:
    
    x = nengo.Ensemble(100, 1, neuron_type=nengo.LIF())
    
    y = nengo.Ensemble(100, 1, neuron_type=nengo.LIF())
    
    nengo.Connection(x, y,
            function=lambda x: 1-4*(x-0.5)**2)
            
            
    def deriv(t, q):
        x0, y0, x1, y1 = q
        dx = x1 - x0
        dy = y1 - y0
        
        delta_y = 1 - y1
        if np.abs(dy) < 1e-2:
            if dy<0:
                dy = -1e-2
            else:
                dy = 1e-2
        delta_x = delta_y * dx / dy
        return delta_x

    dydx_node = nengo.Node(deriv, size_in=4)
    nengo.Connection(x, dydx_node[0], synapse=0.02)
    nengo.Connection(y, dydx_node[1], synapse=0.02)
    nengo.Connection(x, dydx_node[2], synapse=0.01)
    nengo.Connection(y, dydx_node[3], synapse=0.01)
    
    stim_x = nengo.Node(0)
    nengo.Connection(stim_x, x)
    
    nengo.Connection(x, x, synapse=0.1)
    nengo.Connection(dydx_node, x, transform=1.0)
    
    
    

    
    