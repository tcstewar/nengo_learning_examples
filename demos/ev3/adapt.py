import nengo
import numpy as np
import timeit
import pytry

import sys
sys.path.append('.')
import ev3link
import pid

class AdaptingController(pytry.NengoTrial):
    def params(self):
        self.param('ev3 address', ev3='10.42.0.3')
        self.param('motor id', motor=0)
        self.param('Kp', Kp=1.0)
        self.param('Kd', Kd=1.0)
        self.param('Ki', Ki=0.0)
        self.param('time for estimating PID derivative', tau_d=0.001)
        self.param('learning rate', learning_rate=1e-4)
        self.param('target Hz', target_Hz=0.5)
        self.param('time to run', T=10)
        self.param('number of neurons', n_neurons=200)


    def model(self, p):
        link = ev3link.EV3Link(p.ev3)
        path0 = '/sys/class/tacho-motor/motor%d/' % p.motor
        link.write(path0 + 'command', 'run-direct')
        self.link = link
        self.path0 = path0

        model = nengo.Network()
        with model:
            def ev3_system(t, x):
                value = int(100 * x[0])
                if value > 100:
                    value = 100
                if value < -100:
                    value = -100
                value = '%d' % value
                link.write(path0 + 'duty_cycle_sp', value)
                try:
                    p = link.read(path0 + 'position')
                    return float(p) / 180 * np.pi
                except:
                    return 0
            
            ev3 = nengo.Node(ev3_system, size_in=1, size_out=1)
            
            pid_c = pid.PID(p.Kp, p.Kd, p.Ki, tau_d=p.tau_d)
            control = nengo.Node(lambda t, x: pid_c.step(x[:1], x[1:]), size_in=2)
            nengo.Connection(ev3, control[:1], synapse=0)
            nengo.Connection(control, ev3, synapse=None)
            
            adapt = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=1,
                                   neuron_type=nengo.LIFRate())
            nengo.Connection(ev3, adapt[0], synapse=None)
            conn = nengo.Connection(adapt, ev3, synapse=0.001, 
                 function=lambda x: 0,
                 learning_rule_type=nengo.PES(learning_rate=p.learning_rate))
            nengo.Connection(control, conn.learning_rule, transform=-1,
                             synapse=None)

            self.start_time = None
            self.elapsed_time = 0
            def real_time_signal(t):
                now = timeit.default_timer()
                if self.start_time is None:
                    self.start_time = now
                real_t = now - self.start_time
                self.elapsed_time = real_t
                return real_t


            def desired_signal(t, real_t):
                return np.sin(real_t*2*np.pi*p.target_Hz)
            
            real_time = nengo.Node(real_time_signal)
            desired = nengo.Node(desired_signal, size_in=1)
            nengo.Connection(real_time, desired, synapse=None)
            nengo.Connection(desired, control[1:], synapse=None)

            self.p_time = nengo.Probe(real_time, synapse=None)
            self.p_desired = nengo.Probe(desired, synapse=None)
            self.p_actual = nengo.Probe(ev3, synapse=None)
        return model

    def evaluate(self, p, sim, plt):
        while self.elapsed_time < p.T:
            sim.step()
        self.link.write(self.path0 + 'duty_cycle_sp', '0')

        if plt:
            trange = sim.data[self.p_time]
            plt.plot(trange, sim.data[self.p_desired], label='desired')
            plt.plot(trange, sim.data[self.p_actual], label='actual')
            plt.legend(loc='best')

        rmse = np.sqrt(np.mean((sim.data[self.p_desired]-
                                sim.data[self.p_actual])**2))
        return dict(rmse=rmse)
            
            
