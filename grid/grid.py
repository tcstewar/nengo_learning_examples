import random
import sys

neighbour_synonyms = ('neighbours', 'neighbors', 'neighbour', 'neighbor')


class Cell:
    wall = False

    def __getattr__(self, key):
        if key in neighbour_synonyms:
            pts = [self.world.get_point_in_direction(
                self.x, self.y, dir) for dir in range(self.world.directions)]
            ns = tuple([self.world.grid[y][x] for (x, y) in pts])
            for n in neighbour_synonyms:
                self.__dict__[n] = ns
            return ns
        raise AttributeError(key)


class Agent:
    world = None
    cell = None

    def __setattr__(self, key, val):
        if key == 'cell':
            old = self.__dict__.get(key, None)
            if old is not None:
                old.agents.remove(self)
            if val is not None:
                val.agents.append(self)
        self.__dict__[key] = val

    def __getattr__(self, key):
        if key == 'left_cell':
            return self.get_cell_on_left()
        elif key == 'right_cell':
            return self.get_cell_on_right()
        elif key == 'ahead_cell':
            return self.get_cell_ahead()
        raise AttributeError(key)

    def turn(self, amount):
        self.dir = (self.dir + amount) % self.world.directions

    def turn_left(self):
        self.turn(-1)

    def turn_right(self):
        self.turn(1)

    def turn_around(self):
        self.turn(self.world.directions / 2)

    def go_in_direction(self, dir):
        target = self.cell.neighbour[dir]
        if getattr(target, 'wall', False):
            return False
        self.cell = target
        return True

    def go_forward(self):
        if self.world is None:
            raise CellularException('Agent has not been put in a World')
        return self.go_in_direction(self.dir)

    def go_backward(self):
        self.turn_around()
        r = self.go_forward()
        self.turn_around()
        return r

    def get_cell_ahead(self):
        return self.cell.neighbour[self.dir]

    def get_cell_on_left(self):
        return self.cell.neighbour[(self.dir - 1) % self.world.directions]

    def get_cell_on_right(self):
        return self.cell.neighbour[(self.dir + 1) % self.world.directions]

    def go_towards(self, target, y=None):
        if not isinstance(target, Cell):
            target = self.world.grid[int(y)][int(target)]
        if self.world is None:
            raise CellularException('Agent has not been put in a World')
        if self.cell == target:
            return
        best = None
        for i, n in enumerate(self.cell.neighbours):
            if n == target:
                best = target
                bestDir = i
                break
            if getattr(n, 'wall', False):
                continue
            dist = (n.x - target.x) ** 2 + (n.y - target.y) ** 2
            if best is None or bestDist > dist:
                best = n
                bestDist = dist
                bestDir = i
        if best is not None:
            if getattr(best, 'wall', False):
                return False
            self.cell = best
            self.dir = bestDir
            return True

    def update(self):
        pass


class World:
    def __init__(self, cell=None, width=None, height=None, directions=8, filename=None, map=None):
        if cell is None:
            cell = Cell
        self.Cell = cell
        self.directions = directions
        if filename or map:
            if filename:
                data = file(filename).readlines()
            else:
                data = map.splitlines()
                if len(data[0]) == 0:
                    del data[0]
            if height is None:
                height = len(data)
            if width is None:
                width = max([len(x.rstrip()) for x in data])
        if width is None:
            width = 20
        if height is None:
            height = 20
        self.width = width
        self.height = height
        self.image = None
        self.reset()
        if filename or map:
            self.load(filename=filename, map=map)

    def get_cell(self, x, y):
        return self.grid[y][x]

    def find_cells(self, filter):
        for row in self.grid:
            for cell in row:
                if filter(cell):
                    yield cell

    def reset(self):
        self.grid = [[self._make_cell(
            i, j) for i in range(self.width)] for j in range(self.height)]
        self.dictBackup = [[{} for i in range(self.width)]
                           for j in range(self.height)]
        self.agents = []
        self.age = 0

    def _make_cell(self, x, y):
        c = self.Cell()
        c.x = x
        c.y = y
        c.world = self
        c.agents = []
        return c

    def randomize(self):
        if not hasattr(self.Cell, 'randomize'):
            return
        for row in self.grid:
            for cell in row:
                cell.randomize()

    def save(self, f=None):
        if not hasattr(self.Cell, 'save'):
            return
        if isinstance(f, type('')):
            f = file(f, 'w')

        total = ''
        for j in range(self.height):
            line = ''
            for i in range(self.width):
                line += self.grid[j][i].save()
            total += '%s\n' % line
        if f is not None:
            f.write(total)
            f.close()
        else:
            return total

    def load(self, filename=None, map=None):
        if not hasattr(self.Cell, 'load'):
            return
        if filename:
            if isinstance(filename, type('')):
                filename = file(filename)
            lines = filename.readlines()
        else:
            lines = map.splitlines()
            if len(lines[0]) == 0:
                del lines[0]
        lines = [x.rstrip() for x in lines]
        fh = len(lines)
        fw = max([len(x) for x in lines])
        if fh > self.height:
            fh = self.height
            starty = 0
        else:
            starty = int((self.height - fh) / 2)
        if fw > self.width:
            fw = self.width
            startx = 0
        else:
            startx = int((self.width - fw) / 2)

        self.reset()
        for j in range(fh):
            line = lines[j]
            for i in range(min(fw, len(line))):
                self.grid[starty + j][startx + i].load(line[i])

    def update(self):
        if hasattr(self.Cell, 'update'):
            for j, row in enumerate(self.grid):
                for i, c in enumerate(row):
                    self.dictBackup[j][i].update(c.__dict__)
                    c.update()
                    c.__dict__, self.dictBackup[j][
                        i] = self.dictBackup[j][i], c.__dict__
            for j, row in enumerate(self.grid):
                for i, c in enumerate(row):
                    c.__dict__, self.dictBackup[j][
                        i] = self.dictBackup[j][i], c.__dict__
            for a in self.agents:
                a.update()
        else:
            for a in self.agents:
                oldCell = a.cell
                a.update()
        self.age += 1

    def get_offset_in_direction(self, x, y, dir):
        if self.directions == 8:
            dx, dy = [(0, -1), (1, -1), (
                1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)][dir]
        elif self.directions == 4:
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][dir]
        elif self.directions == 6:
            if y % 2 == 0:
                dx, dy = [(1, 0), (0, 1), (-1, 1), (-1, 0),
                          (-1, -1), (0, -1)][dir]
            else:
                dx, dy = [(1, 0), (1, 1), (0, 1), (-1, 0),
                          (0, -1), (1, -1)][dir]
        return dx, dy


    def get_point_in_direction(self, x, y, dir):
        dx, dy = self.get_offset_in_direction(x, y, dir)

        x2 = x + dx
        y2 = y + dy

        if x2 < 0:
            x2 += self.width
        if y2 < 0:
            y2 += self.height
        if x2 >= self.width:
            x2 -= self.width
        if y2 >= self.height:
            y2 -= self.height

        return (x2, y2)

    def remove(self, agent):
        self.agents.remove(agent)
        agent.world = None
        agent.cell = None

    def add(self, agent, x=None, y=None, cell=None, dir=None):
        self.agents.append(agent)
        if x is not None and y is not None:
            cell = self.grid[y][x]
        if cell is None:
            while True:
                xx = x
                yy = y
                if xx is None:
                    xx = random.randrange(self.width)
                if yy is None:
                    yy = random.randrange(self.height)
                if not getattr(self.grid[yy][xx], 'wall', False):
                    y = yy
                    x = xx
                    break
        else:
            x = cell.x
            y = cell.y

        if dir is None:
            dir = random.randrange(self.directions)

        agent.cell = self.grid[y][x]
        agent.dir = dir
        agent.world = self
        agent.x = x
        agent.y = y


class CellularException(Exception):
    pass

	
class ContinuousAgent(Agent):
    def go_in_direction(self, dir, distance=1, return_obstacle=False):

        dir1=int(dir)
        dir2=(dir1+1)%self.world.directions

        dx1, dy1 = self.world.get_offset_in_direction(self.cell.x, self.cell.y, dir1)
        dx2, dy2 = self.world.get_offset_in_direction(self.cell.x, self.cell.y, dir2)

        scale=dir % 1

        x = self.x + distance*(dx2*scale + dx1*(1 - scale))
        y = self.y + distance*(dy2*scale + dy1*(1 - scale))

        closest = self.cell
        dist = (x-self.cell.x)**2 + (y-self.cell.y)**2
        for n in self.cell.neighbour:
            d = (x-n.x)**2 + (y-n.y)**2
            if d<dist:
                closest = n
                dist = d
        if closest is not self.cell:
            if closest.wall:
                if return_obstacle:
                    return closest
                else:
                    return False
            else:
                self.cell=closest

        self.x=x
        self.y=y

        if return_obstacle:
            return None
        else:
            return True

    def go_forward(self, distance=1):
        return self.go_in_direction(self.dir, distance=distance)

    def go_backward(self, distance=1):
        return self.go_in_direction(self.dir, distance=-distance)

    def detect(self, direction, max_distance=None):
        start_x = self.x
        start_y = self.y
        cell = self.cell
        distance = 0.0
        delta = 1.0
        min_delta = 1.0 / 64
        obstacle = None
        if max_distance is None:
            max_distance = self.world.width + self.world.height

        while distance < max_distance:
            obstacle = self.go_in_direction(direction, delta, return_obstacle=True)
            if obstacle is None:
                distance += delta
            elif delta > min_delta:
                delta = delta / 2
            else:
                distance = math.sqrt((start_x-self.x)**2 + (start_y-self.y)**2)
                break
        self.cell = cell
        self.x = start_x
        self.y = start_y
        return distance, obstacle

    def get_direction_to(self, cell):
        dx = cell.x - self.x
        dy = cell.y - self.y

        theta = math.atan2(dy, dx) + math.pi/2
        theta *= self.world.directions / (2 * math.pi)
        return theta

    def get_distance_to(self, cell):
        dx = cell.x - self.x
        dy = cell.y - self.y
        return math.sqrt(dx**2 + dy**2)

		
import nengo		
# GridNode sets up the pacman world for visualization
class GridNode(nengo.Node):
    def __init__(self, world, dt=0.001):

        # The initalizer sets up the html layout for display
        def svg(t):
            last_t = getattr(svg, '_nengo_html_t_', None)
            if last_t is None or t >= last_t + dt or t <= last_t:
                svg._nengo_html_ = self.generate_svg(world)
                svg._nengo_html_t_ = t
        super(GridNode, self).__init__(svg)

    # This function sets up an SVG (used to embed html code in the environment)
    def generate_svg(self, world):
        cells = []
        # Runs through every cell in the world (walls & food)
        for i in range(world.width):
            for j in range(world.height):
                cell = world.get_cell(i, j)
                color = cell.color
                if callable(color):
                    color = color()

                if color is not None:
                    cells.append('<rect x=%d y=%d width=1 height=1 style="fill:%s"/>' %
                         (i, j, color))

        # Runs through every agent in the world
        agents = []
        for agent in world.agents:

            # sets variables like agent direction, color and size
            direction = agent.dir * 360.0 / world.directions
            color = getattr(agent, 'color', 'blue')
            if callable(color):
                color = color()

            shape = getattr(agent, 'shape', 'triangle')

            if shape == 'triangle':

                agent_poly = ('<polygon points="0.25,0.25 -0.25,0.25 0,-0.5"'
                         ' style="fill:%s" transform="translate(%f,%f) rotate(%f)"/>'
                         % (color, agent.x+0.5, agent.y+0.5, direction))

            elif shape == 'circle':
                agent_poly = ('<circle '
                         ' style="fill:%s" cx="%f" cy="%f" r="0.4"/>'
                         % (color, agent.x+0.5, agent.y+0.5))

            agents.append(agent_poly)

        # Sets up the environment as a HTML SVG
        svg = '''<svg style="background: white" width="100%%" height="100%%" viewbox="0 0 %d %d">
            %s
            %s
            </svg>''' % (world.width, world.height,
                         ''.join(cells), ''.join(agents))
        return svg
	
