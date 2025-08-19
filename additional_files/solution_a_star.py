"""
This script should contain a class Wallace, which implements a function `act`, as follows.
Notably, `act` is expected to take 3 parameters (`obs`, `gold`, `done`), and output `None` if `done==True` or an element of `Action` otherwise.

Here, this examples of Wallace are a bit dummy: he repeatedly executes the same sequence of actions.
Your mission is to modify the class so that Wallace has better chances
to gather colossal quantities of gold from the maze.

If you need helper files, please use imports, following this template:
from myhelper import bbb, ccc # use this for custom relative scripts
import numpy as np # this syntax is fine for packages
"""
import enum
import numpy as np
# from myhelper import bbb, ccc
from a_star_utils import a_star, shortest_path
from collections import deque

class Cell(enum.IntEnum):
    EMPTY = 0
    WALL = 1

class Action(enum.IntEnum):
    """See maze.py:Action
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    GATHER = 4

class Wallace:
    """
    Wallace has no clue about the layout of the maze, nor the locations where gold can be gathered.
    Wallace needs to have a clever strategy in order to explore the maze, find the best gold locations
    and maximize its total collected gold in a limited number of timesteps.
    """

    def __init__(self):
        self.sequence = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.map = np.zeros((29, 29), dtype=int)
        self.discovered = np.zeros((29, 29), dtype=bool)
        self.gold_map = np.zeros((29, 29), dtype=float)
        self.total_count = 0
        self.gold_count = np.zeros_like(self.gold_map, dtype=float)
        self.idx = 0
        self.directions = {
            (-1, 0): Action.UP,
            (1, 0): Action.DOWN,
            (0, -1): Action.LEFT,
            (0, 1): Action.RIGHT
        }
        self.ucb_constant = 1.0
    
    def find_nearest_frontier(self, source):
        rows, cols = len(self.map), len(self.map[0])
        q = deque([source])
        seen = {source}
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            cur = q.popleft()
            r,c = cur
            # frontier condition: known free and has at least one unknown neighbor
            if self.map[r][c] == Cell.EMPTY:
                for dr,dc in dirs:
                    nr,nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and not self.discovered[nr][nc]:
                        return cur
            for dr,dc in dirs:
                nxt = (r+dr, c+dc)
                nr,nc = nxt
                if 0 <= nr < rows and 0 <= nc < cols and nxt not in seen and self.discovered[nr][nc]:
                    seen.add(nxt); q.append(nxt)
        return None
    
    def update_map(self, obs):
        y, x, top, left, right, bottom, has_gold = obs
        self.map[y, x] = Cell.EMPTY
        self.discovered[y, x] = True
        directions = [(-1, 0, top), (1, 0, bottom), (0, -1, left), (0, 1, right)]

        for (dy, dx, cell) in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.map.shape[0] and 0 <= nx < self.map.shape[1]:
                self.map[ny, nx] = cell
                self.discovered[ny, nx] = True
    
    def ucb_value(self, mean_estimate, total_count, gold_count):
        return mean_estimate + (self.ucb_constant * np.sqrt(np.log(total_count)/gold_count))
    
    def update_gold_map(self, obs, gold_received_at_previous_step): # UCB to implement here
        y, x, _, _, _, _, _ = obs

        spawn = (3, 3)
        path = a_star(self.map, spawn, (y, x))
        gold_reward = gold_received_at_previous_step/len(path)

        new_gold_mean = ((self.gold_map[y, x] * self.gold_count[y, x]) + gold_reward) / (self.gold_count[y, x] + 1)

        self.total_count += 1
        self.gold_count[y, x] += 1

        self.gold_map[y, x] = self.ucb_value(new_gold_mean, self.total_count, self.gold_count[y, x])
    
    def pick_best_gold(self, obs):
        y, x, _, _, _, _, _ = obs
        max_gold = np.max(self.gold_map)
        coordinates = np.argwhere(self.gold_map == max_gold)
        path = a_star(self.map, (y, x), tuple(coordinates[0]))
        if len(path) > 1:
            direction_to_take = (path[1][0]-y, path[1][1] - x)
            return self.directions.get(direction_to_take, None)
        else:
            return Action.GATHER

    def act(self, obs, gold_received_at_previous_step, done):
        """
        This function gets called alternatively with env.step. At every step, Wallace gets
        some information and must decide which action to perform.

        When Wallace.act returns Action.GATHER, he will then receive
        gold_received_at_previous_step = gold != 0 only if he was standing on gold.
        Whether he was standing on gold or not, he will be sent back to the start, which
        corresponds to done == True; as a result, the action returned when done == True
        does not matter, and is expected to be None.

        Args:
            obs (tuple): The information that Wallace has at this time step: its current
                (y,x) coordinates, the cells surrounding him (ie either Cell.EMPTY or 
                Cell.WALL), and whether he is standing on gold. Ie:
                (y (int), x (int), top (Cell), left (Cell), right (Cell), bottom (Cell),
                has_gold (bool)
            gold_received_at_previous_step (float): The amount of gold that Wallace just received.
                When this is non zero, done == True.
            done (bool): When True, Wallace will be sent back to the start for the next
                step: as a result, in that case, we expect the returned action to be None.

        Returns:
            action (Action): The action that Wallace decides to execute at this time step.
        """
        if done:
            self.update_gold_map(obs, gold_received_at_previous_step)
            return None
        y, x, top, left, right, bottom, has_gold = obs
        self.update_map(obs)
        if has_gold:
            if self.gold_map[y, x] == 0:
                return Action.GATHER

        discovery_goal = self.find_nearest_frontier((y, x))

        if discovery_goal is not None: # Exploration

            path = a_star(self.map, (y, x), discovery_goal)
            direction_to_take = (path[1][0]-y, path[1][1] - x)
            action = self.directions.get(direction_to_take, None)

        else: # Exploitation
            action = self.pick_best_gold(obs)

        self.idx = (self.idx + 1) % len(self.sequence)
        return action

    def get_custom_render_infos(self):
        """If it helps you for debugging your Wallace and/or explaining its behavior, you can modify this function 
        so that it returns a list of tuple of the form (pixel location, RGB color) e.g. [((y1,x1), (r1,g1,b1)), ((y2,x2), (r2,g2,b2))]
        
        These pixel colors will overwrite the default colors used to generate Wallace's gameplay video (except Wallace's current location and fog of war)"""
        return None
        infos = []

        height, width = self.map.shape
        for y in range(height):
            for x in range(width):
                r = g = b = 0
                if self.map[y,x] == Cell.EMPTY:
                    g = 255
                else:
                    r = 255
                if not self.discovered[y,x]:
                    b = 255
                    r = g = 0

                infos.append(((y, x), (r, g, b)))

        return infos