import nengo
import grid

class Cell(grid.Cell):
    def load(self, x):
        self.wall = x == 'X'
    def color(self):
        if self.wall:
            return 'black'
    
map = '''
XXXXX
X   X
X   X
X   X
XXXXX
'''
    
world = grid.World(Cell, map=map, directions=4)

agent = grid.ContinuousAgent()
world.add(agent, x=2, y=2)

model = nengo.Network()
with model:
    env = grid.GridNode(world)
    
