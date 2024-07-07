# importing necessary modules
import pygame
from random import choice, shuffle

# setting the window resolution
RES = WIDTH, HEIGHT = 1202, 760
TILE = 50  # width of each tile (square)
cols, rows = WIDTH // TILE, HEIGHT // TILE  # calculating the number of columns and rows based on window size

# initializing a pygame window
pygame.init()
sc = pygame.display.set_mode(RES)
clock = pygame.time.Clock()

# defining a class for each cell in the grid
class Cell:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.walls = {'top': True, 'right': True, 'bottom': True, 'left': True}  # each cell starts with all walls intact
        self.visited = False  # initially marking all cells as unvisited

    # method to highlight the current cell
    def draw_current_cell(self):
        x, y = self.x * TILE, self.y * TILE
        pygame.draw.rect(sc, pygame.Color('yellow'), (x + 2, y + 2, TILE - 2, TILE - 2))

    # method to draw the cell and its walls
    def draw(self):
        x, y = self.x * TILE, self.y * TILE
        if self.visited:
            pygame.draw.rect(sc, pygame.Color('black'), (x, y, TILE, TILE))

        # draw each wall if it exists
        if self.walls['top']:
            pygame.draw.line(sc, pygame.Color('darkorange'), (x, y), (x + TILE, y), 2)
        if self.walls['right']:
            pygame.draw.line(sc, pygame.Color('darkorange'), (x + TILE, y), (x + TILE, y + TILE), 2)
        if self.walls['bottom']:
            pygame.draw.line(sc, pygame.Color('darkorange'), (x + TILE, y + TILE), (x, y + TILE), 2)
        if self.walls['left']:
            pygame.draw.line(sc, pygame.Color('darkorange'), (x, y + TILE), (x, y), 2)

    # method to check if a neighboring cell is within bounds
    def check_cell(self, x, y):
        find_index = lambda x, y: x + y * cols  # calculates index in grid_cells list
        if x < 0 or x > cols - 1 or y < 0 or y > rows - 1:
            return False  # returns False if cell is out of bounds
        return grid_cells[find_index(x, y)]

    # method to find all unvisited neighbors of a cell
    def check_neighbours(self):
        neighbors = []
        top = self.check_cell(self.x, self.y - 1)
        right = self.check_cell(self.x + 1, self.y)
        bottom = self.check_cell(self.x, self.y + 1)
        left = self.check_cell(self.x - 1, self.y)

        # add only unvisited neighbors to the list
        if top and not top.visited:
            neighbors.append(top)
        if right and not right.visited:
            neighbors.append(right)
        if bottom and not bottom.visited:
            neighbors.append(bottom)
        if left and not left.visited:
            neighbors.append(left)
        return neighbors


# function to remove walls between two neighboring cells
def remove_walls(current, next):
    dx = current.x - next.x  # calculate the difference in x coordinates
    if dx == 1:  # next cell is to the left
        current.walls['left'] = False
        next.walls['right'] = False
    elif dx == -1:  # next cell is to the right
        current.walls['right'] = False
        next.walls['left'] = False
    dy = current.y - next.y  # calculate the difference in y coordinates
    if dy == 1:  # next cell is above
        current.walls['top'] = False
        next.walls['bottom'] = False
    elif dy == -1:  # next cell is below
        current.walls['bottom'] = False
        next.walls['top'] = False

# creating a grid of cells
grid_cells = [Cell(col, row) for row in range(rows) for col in range(cols)]
start_cell = grid_cells[0]
maze_generated = False

# function to generate a maze using Prim's algorithm
def generate_maze_prims():
    global maze_generated
    start_cell.visited = True  # mark the starting cell as visited
    walls = []  # list to store the walls

    # function to add neighboring walls of a cell to the wall list
    def add_walls(cell):
        x, y = cell.x, cell.y
        # check and add each neighboring cell's wall if the cell is within bounds and unvisited
        if y > 0:
            top = cell.check_cell(x, y - 1)
            if top and not top.visited:
                walls.append((cell, top))
        if x < cols - 1:
            right = cell.check_cell(x + 1, y)
            if right and not right.visited:
                walls.append((cell, right))
        if y < rows - 1:
            bottom = cell.check_cell(x, y + 1)
            if bottom and not bottom.visited:
                walls.append((cell, bottom))
        if x > 0:
            left = cell.check_cell(x - 1, y)
            if left and not left.visited:
                walls.append((cell, left))

    add_walls(start_cell)

    # main loop for maze generation
    while walls:
        sc.fill(pygame.Color('darkslategray'))  # fill the screen with background color

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        for cell in grid_cells:
            cell.draw()  # draw all cells

        cell1, cell2 = choice(walls)  # choose a random wall
        cell1.draw_current_cell()   # highlight the current cell
        if not cell2.visited:
            remove_walls(cell1, cell2)  # remove the wall between the cells
            cell2.visited = True  # mark the cell as visited
            add_walls(cell2)  # add the walls of the cell to the list
        walls.remove((cell1, cell2))  # remove the wall from the list

        pygame.display.flip()
        clock.tick(60)

    maze_generated = True


# heuristic function for A* (Manhattan distance)
def h(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return abs(x1 - x2) + abs(y1 - y2)


# A* algorithm implementation
def a_star(start, goal):
    open_set = [(h(start, goal), start)]  # initialize the open set with the start node
    came_from = {}  # dictionary to store the path
    g_score = {cell: float('inf') for cell in [(c.x, c.y) for c in grid_cells]}  # initialize g_scores
    g_score[start] = 0
    f_score = {cell: float('inf') for cell in [(c.x, c.y) for c in grid_cells]}  # initialize f_scores
    f_score[start] = h(start, goal)

    while open_set:
        open_set.sort(reverse=True)  # sort the open set to get the node with the lowest f_score
        _, current = open_set.pop()  # get the current node

        if current == goal:
            return reconstruct_path(came_from, current)  # reconstruct the path if goal is reached

        neighbors = get_neighbors(current)  # get the neighbors of the current node
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + 1  # tentative g_score
            if tentative_g_score < g_score[neighbor]:  # if new path is better
                came_from[neighbor] = current  # update the path
                g_score[neighbor] = tentative_g_score  # update g_score
                f_score[neighbor] = tentative_g_score + h(neighbor, goal)  # update f_score
                if neighbor not in [i[1] for i in open_set]:  # if neighbor is not in open set
                    open_set.append((f_score[neighbor], neighbor))  # add neighbor to open set

        # Visualization of A* process
        for cell in grid_cells:
            cell.draw()
        for x, y in came_from:
            pygame.draw.rect(sc, pygame.Color('lightblue'), (x * TILE + 2, y * TILE + 2, TILE - 4, TILE - 4))
        pygame.display.flip()
        clock.tick(60)

    return []


# BFS algorithm implementation
def bfs(start, goal):
    from collections import deque
    queue = deque([start])  # initialize the queue with the start node
    came_from = {}  # dictionary to store the path
    visited = set()  # set to store visited nodes
    visited.add(start)

    while queue:
        current = queue.popleft()  # get the current node

        if current == goal:
            return reconstruct_path(came_from, current)  # reconstruct the path if goal is reached

        neighbors = get_neighbors(current)  # get the neighbors of the current node
        for neighbor in neighbors:
            if neighbor not in visited:  # if neighbor is not visited
                visited.add(neighbor)  # mark neighbor as visited
                came_from[neighbor] = current  # update the path
                queue.append(neighbor)  # add neighbor to the queue

        # Visualization of BFS process
        for cell in grid_cells:
            cell.draw()
        for x, y in visited:
            pygame.draw.rect(sc, pygame.Color('lightblue'), (x * TILE + 2, y * TILE + 2, TILE - 4, TILE - 4))
        pygame.display.flip()
        clock.tick(60)

    return []


# DFS algorithm implementation
def dfs(start, goal):
    stack = [start]  # initialize the stack with the start node
    came_from = {}  # dictionary to store the path
    visited = set()  # set to store visited nodes
    visited.add(start)

    while stack:
        current = stack.pop()  # get the current node

        if current == goal:
            return reconstruct_path(came_from, current)  # reconstruct the path if goal is reached

        neighbors = get_neighbors(current)  # get the neighbors of the current node
        for neighbor in neighbors:
            if neighbor not in visited:  # if neighbor is not visited
                visited.add(neighbor)  # mark neighbor as visited
                came_from[neighbor] = current  # update the path
                stack.append(neighbor)  # add neighbor to the stack

        # Visualization of DFS process
        for cell in grid_cells:
            cell.draw()
        for x, y in visited:
            pygame.draw.rect(sc, pygame.Color('lightblue'), (x * TILE + 2, y * TILE + 2, TILE - 4, TILE - 4))
        pygame.display.flip()
        clock.tick(60)

    return []


# function to reconstruct the path from the came_from dictionary
def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:  # traverse back from the goal to start
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()  # reverse the path to get it from start to goal
    return total_path


# function to get the neighbors of any cell
def get_neighbors(cell):
    x, y = cell
    neighbors = []
    # check each neighbor's wall status and add if there is no wall between
    if y > 0 and not find_cell(x, y).walls['top'] and not find_cell(x, y - 1).walls['bottom']:
        neighbors.append((x, y - 1))
    if x < cols - 1 and not find_cell(x, y).walls['right'] and not find_cell(x + 1, y).walls['left']:
        neighbors.append((x + 1, y))
    if y < rows - 1 and not find_cell(x, y).walls['bottom'] and not find_cell(x, y + 1).walls['top']:
        neighbors.append((x, y + 1))
    if x > 0 and not find_cell(x, y).walls['left'] and not find_cell(x - 1, y).walls['right']:
        neighbors.append((x - 1, y))
    return neighbors

# function to find a cell in the grid
def find_cell(x, y):
    if 0 <= x < cols and 0 <= y < rows:  # check if cell is within bounds
        return grid_cells[x + y * cols]
    return None

# main function
if __name__ == '__main__':
    start = (0, 0)
    goal = (cols - 1, rows - 1)
    path = []
    display_path = False
    maze_generated = False
    start_generation = False

    while True:
        sc.fill(pygame.Color('darkslategray'))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a and maze_generated:
                    path = a_star(start, goal)  # execute A* algorithm
                    display_path = True
                if event.key == pygame.K_b and maze_generated:
                    path = bfs(start, goal)  # execute BFS algorithm
                    display_path = True
                if event.key == pygame.K_d and maze_generated:
                    path = dfs(start, goal)  # execute DFS algorithm
                    display_path = True
                if event.key == pygame.K_s and not start_generation:
                    start_generation = True  # start maze generation

        if start_generation and not maze_generated:
            generate_maze_prims()  # generate the maze if not already generated

        for cell in grid_cells:
            cell.draw()  # draw all cells

        if display_path:
            for x, y in path:
                pygame.draw.rect(sc, pygame.Color('blue'), (x * TILE + 2, y * TILE + 2, TILE - 4, TILE - 4))  # draw the path

        # draw the start and goal cells
        pygame.draw.rect(sc, pygame.Color('red'), (start[0] * TILE + 2, start[1] * TILE + 2, TILE - 4, TILE - 4))
        pygame.draw.rect(sc, pygame.Color('green'), (goal[0] * TILE + 2, goal[1] * TILE + 2, TILE - 4, TILE - 4))

        pygame.display.flip()
        clock.tick(60)
