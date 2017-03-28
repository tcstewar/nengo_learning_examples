import nengo
import numpy as np

import ev3link
import pid

if not hasattr(ev3link, 'link'):
    ev3link.link = ev3link.EV3Link('10.42.0.3')
    
link = ev3link.link

path0 = '/sys/class/tacho-motor/motor0/'
link.write(path0 + 'command', 'run-direct')

model = nengo.Network()
with model:
    
    def ev3_system(t, x):
        value = int(100 * x[0])
        if value > 100:
            value = 100
        if value < -100:
            value = -100
        value = '%d' % value
        ev3link.link.write(path0 + 'duty_cycle_sp', value)
        try:
            p = link.read(path0 + 'position')
            return float(p) / 180 * np.pi
        except:
            return 0
    
    ev3 = nengo.Node(ev3_system, size_in=1, size_out=1)
    
    
    pid = pid.PID(0.5,1, 0, tau_d=0.001)
    control = nengo.Node(lambda t, x: pid.step(x[:1], x[1:]), size_in=2)
    nengo.Connection(ev3, control[:1], synapse=0)
    nengo.Connection(control, ev3, synapse=None)
    
    adapt = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.LIFRate())
    nengo.Connection(ev3, adapt[0], synapse=None)
    nengo.Connection(ev3, adapt[1], synapse=0.01)
    conn = nengo.Connection(adapt, ev3, synapse=0.001, 
                            function=lambda x: 0,
                            learning_rule_type=nengo.PES(learning_rate=1e-4))
    nengo.Connection(control, conn.learning_rule, transform=-1, synapse=None)
    
    
    desired = nengo.Node(lambda t: np.sin(t*2*np.pi))
    nengo.Connection(desired, control[1:], synapse=None)
    
    