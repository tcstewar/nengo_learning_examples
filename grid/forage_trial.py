import pytry
import nengo
import numpy as np
import scipy.ndimage

import sys
sys.path.append('.')
import grid

map = '''
#######
#     #
#     #
#     #
#     #
#     #
#######
'''

class Cell(grid.Cell):
    def load(self, c):
        if c == '#':
            self.wall = True
    
    def color(self):
        if self.wall:
            return 'black'
            
            
class Prey(grid.ContinuousAgent):
    color = 'green'
    shape = 'circle'


class ForageTrial(pytry.NengoTrial):
    def params(self):
        self.param('simulation time', T=10.0)

    def model(self, p):
                    
        world = grid.World(Cell, map=map, directions=4)
        body = grid.ContinuousAgent()
        world.add(body, x=2, y=2, dir=1)

        prey = Prey()
        world.add(prey, x=2, y=3)


        class FacingReward(nengo.Node):
            def __init__(self, body, target):
                self.body = body
                self.target = target
                self.last_t = None
                self.last_theta = None
                super(FacingReward, self).__init__(self.update)
            def update(self, t):
                if self.last_t is not None and t < self.last_t:
                    self.last_t = None
                    
                angle = (body.dir-1) * 2 * np.pi / 4
                food_angle = np.arctan2(self.target.y - self.body.y,
                                      self.target.x - self.body.x)
                                      
                theta = angle - food_angle
                if theta > np.pi:
                    theta -= np.pi * 2
                if theta < -np.pi:
                    theta += np.pi * 2
                
                theta = np.abs(theta)

                value = 0
                eps = 1e-5
                if self.last_t is not None:
                    if theta < self.last_theta-eps:
                        value = 1
                    elif theta > self.last_theta-eps:
                        value = -1
                else:
                    value = -1
                self.last_t = t
                self.last_theta = theta
                
                return value
                    
        class State(nengo.Node):
            def __init__(self, body, food):
                self.body = body
                self.food = food
                super(State, self).__init__(self.update)
                
            def update(self, t):
                angle = (body.dir-1) * 2 * np.pi / 4
                dy = self.food.y - self.body.y
                dx = self.food.x - self.body.x
                food_angle = np.arctan2(dy, dx)
                theta = angle - food_angle
                dist2 = dy**2 + dx**2
                
                if dist2 == 0:
                    r = 1.0
                else:
                    r = np.clip(0.5 / dist2, 0.2, 1.0)
                return -np.sin(theta)*r, np.cos(theta)*r
                
        D = 3

        model = nengo.Network()
        with model:
            
            speed = nengo.Node(lambda t, x: body.go_forward(x[0]*0.01), size_in=1)
            turn = nengo.Node(lambda t, x: body.turn(x[0]*0.003), size_in=1)
            ctrl_speed = nengo.Node([0.3])
            nengo.Connection(ctrl_speed, speed)
            
            facing_reward = FacingReward(body, prey)
            sensors = State(body, prey)
            
            state = nengo.Ensemble(n_neurons=200, dimensions=2, 
                                   intercepts=nengo.dists.Uniform(-0.1,1))
            nengo.Connection(sensors, state, synapse=None)

            bg = nengo.networks.BasalGanglia(D)
            thal = nengo.networks.Thalamus(D)
            nengo.Connection(bg.output, thal.input)
            choice = thal.output

            q = nengo.Node(None, size_in=D)
            nengo.Connection(q, bg.input, transform=2)
            bias = nengo.Node(0.5, label='')
            nengo.Connection(bias, bg.input, transform=np.ones((D, 1)))
            
            conn = nengo.Connection(state, q, function=lambda x: [0]*D,
                                    learning_rule_type=nengo.PES(learning_rate=1e-4,
                                                                 pre_tau=0.05))
            nengo.Connection(choice[0], turn, transform=-1, synapse=0.1)
            nengo.Connection(choice[1], turn, transform=1, synapse=0.1)

            def target(t, x):
                index = np.argmax(x[1:])
                r = x[0]
                
                if index in [0,1]:
                    r = r - 0.3
                    
                
                result = np.ones(D) * -r
                result[index] = r
                return result
            target = nengo.Node(target, size_in=D+1)

            nengo.Connection(choice, target[1:])
            nengo.Connection(facing_reward, target[0])
            
            error = nengo.Node(None, size_in=D)
            
            nengo.Connection(q, error)
            nengo.Connection(target, error, transform=-1)

            nengo.Connection(error, conn.learning_rule)

            self.score = 0

            score_node = nengo.Node(lambda t: self.score)
            self.p_score = nengo.Probe(score_node)
            
            
            def move_prey(t):
                dy = prey.y - body.y
                dx = prey.x - body.x
                dist2 = dx**2 + dy**2
                
                while dist2 < 0.25:
                    prey.x = np.random.uniform(1, world.width-2)
                    prey.y = np.random.uniform(1, world.height-2)
                    dy = prey.y - body.y
                    dx = prey.x - body.x
                    dist2 = dx**2 + dy**2
                    self.score += 1
            move_prey = nengo.Node(move_prey)

            if p.gui:
                env = grid.GridNode(world)
                self.locals = locals()
                    
        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T)

        score = sim.data[self.p_score][:,0]
        delta = np.diff(score).astype(float) / p.dt
        indices = np.where(delta>0)[0]
        times = indices.astype(float) * p.dt

        if plt:
            deltaf = scipy.ndimage.filters.gaussian_filter1d(delta, sigma=1000, mode='reflect')
            plt.plot(sim.trange()[1:], deltaf)
            for t in times:
                plt.axvline(t, color='k')

        return dict(
                score_times = times,
                )
