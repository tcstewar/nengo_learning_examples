import nengo
import grid
import nengo_xbox
import numpy as np

map = '''
#####
#   #
#   #
#   #
#####
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
            
world = grid.World(Cell, map=map, directions=4)
body = grid.ContinuousAgent()
world.add(body, x=2, y=2, dir=2)

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
        if self.last_t is not None:
            if theta < self.last_theta:
                value = 1
            elif theta > self.last_theta:
                value = -1
        else:
            value = 0
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
            r = min(0.5 / dist2, 1.0)
        return -np.sin(theta)*r, np.cos(theta)*r
        


        


model = nengo.Network()
with model:
    env = grid.GridNode(world)
    
    speed = nengo.Node(lambda t, x: body.go_forward(x[0]*0.01), size_in=1)
    turn = nengo.Node(lambda t, x: body.turn(x[0]*0.01), size_in=1)
    
    xbox = nengo_xbox.Xbox()
    
    nengo.Connection(xbox.axis[1], speed)
    nengo.Connection(xbox.axis[0], turn)
    
    facing_reward = FacingReward(body, prey)
    
    state = State(body, prey)
    
    
    
    
    