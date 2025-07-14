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
        self.sequence = [Action.UP, Action.RIGHT, Action.RIGHT, Action.GATHER]
        self.idx = 0
        self.init_value = 5
        self.Q_table = np.ones((29,29,5))*self.init_value
        self.lr = 0.25
        self.decay_rate = 0.6
        self.epsilon = 0.5
        self.gamma = 0.99
        self.wall_penalty = 0
        self.wrong_gather = -10
        self.has_gold_reward = 3
    
    def epsilon_greedy(self, y, x):
        """
        Returns an action according to an epsilon-greedy policy.
        With probability epsilon, a random action is chosen.
        Otherwise, the action with the highest Q-value is chosen.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(Action))
        else:
            
            return np.argmax(self.Q_table[y, x, :])
    
    def greedy(self, y, x):
        """
        Returns the action with the highest Q-value.
        This is a deterministic policy.
        """
        return Action(np.argmax(self.Q_table[y, x, :]))

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
        y,x,top,left,right,bottom,has_gold = obs

        gold = gold_received_at_previous_step

        if done:
            if gold > 0:
                self.Q_table[y,x,Action.GATHER] += self.lr * (gold - self.Q_table[y,x,Action.GATHER])
            else:
                self.Q_table[y,x,Action.GATHER] += self.lr * (self.wrong_gather - self.Q_table[y,x,Action.GATHER])
            
            self.idx = 0
            self.epsilon *= self.decay_rate

            if top == Cell.EMPTY:
                self.Q_table[y-1,x,Action.DOWN] += self.lr * (self.gamma*self.Q_table[y,x,Action.GATHER] - self.Q_table[y-1,x,Action.DOWN])

            if left == Cell.EMPTY:
                self.Q_table[y,x-1,Action.RIGHT] += self.lr * (self.gamma*self.Q_table[y,x,Action.GATHER] - self.Q_table[y,x-1,Action.RIGHT])

            if right == Cell.EMPTY:
                self.Q_table[y,x+1,Action.RIGHT] += self.lr * (self.gamma*self.Q_table[y,x,Action.GATHER] - self.Q_table[y,x+1,Action.LEFT])

            if bottom == Cell.EMPTY:
                self.Q_table[y+1,x,Action.UP] += self.lr * (self.gamma*self.Q_table[y,x,Action.GATHER] - self.Q_table[y+1,x,Action.UP])

            return None
        
        self.idx += 1

        if top == Cell.EMPTY:
            self.Q_table[y,x,Action.UP] += self.lr * (self.gamma*np.max(self.Q_table[y-1,x,:]) - self.Q_table[y,x,Action.UP])
        
        else:
            self.Q_table[y,x,Action.UP] = self.wall_penalty

        if left == Cell.EMPTY:
            self.Q_table[y,x,Action.LEFT] += self.lr * (self.gamma*np.max(self.Q_table[y,x-1,:]) - self.Q_table[y,x,Action.LEFT])

        else:
            self.Q_table[y,x,Action.LEFT] = self.wall_penalty

        if right == Cell.EMPTY:
            self.Q_table[y,x,Action.RIGHT] += self.lr * (self.gamma*np.max(self.Q_table[y,x+1,:]) - self.Q_table[y,x,Action.RIGHT])

        else:
            self.Q_table[y,x,Action.RIGHT] = self.wall_penalty

        if bottom == Cell.EMPTY:
            self.Q_table[y,x,Action.DOWN] += self.lr * (self.gamma*np.max(self.Q_table[y+1,x,:]) - self.Q_table[y,x,Action.DOWN])

        else:
            self.Q_table[y,x,Action.DOWN] = self.wall_penalty

        if ~has_gold:
            self.Q_table[y,x,Action.GATHER] = self.wrong_gather
        else:
            self.Q_table[y,x,Action.GATHER] += self.lr * self.has_gold_reward

        action = self.greedy(y, x)
        return action

    def get_custom_render_infos(self):
        """If it helps you for debugging your Wallace and/or explaining its behavior, you can modify this function 
        so that it returns a list of tuple of the form (pixel location, RGB color) e.g. [((y1,x1), (r1,g1,b1)), ((y2,x2), (r2,g2,b2))]
        
        These pixel colors will overwrite the default colors used to generate Wallace's gameplay video (except Wallace's current location and fog of war)
        
        I added two functions: if you want to visualize the min or max just return self.get_..._q_values_info()"""
        return None
    
    def get_max_q_values_infos(self):
        """Visualize the Q-table using RGB colors.
        Cells with higher max Q-values will appear more blue.
        """
        infos = []
        max_q = np.max(self.Q_table)
        min_q = np.min(self.Q_table)

        if max_q == min_q:
            return infos

        norm = lambda q: (q - min_q) / (max_q - min_q + 1e-8)

        height, width, _ = self.Q_table.shape
        for y in range(height):
            for x in range(width):
                max_qval = np.max(self.Q_table[y, x, :])
                value = norm(max_qval)

                # higher value = more blue
                r = 0
                g = int(255 * (1 - value))
                b = int(255 * value)

                infos.append(((y, x), (r, g, b)))

        return infos

    def get_min_q_values_infos(self):
        infos = []

        min_q_values = np.min(self.Q_table, axis=2)

        max_q = np.max(min_q_values)
        min_q = np.min(min_q_values)

        if max_q == min_q:
            return None

        norm = lambda q: (q - min_q) / (max_q - min_q + 1e-8)

        height, width, _ = self.Q_table.shape
        for y in range(height):
            for x in range(width):
                min_qval = min_q_values[y, x]
                value = norm(min_qval)

                r = int(255 * (1 - value))  # more red for lower value
                g = 0
                b = int(255 * value)        # more blue for higher value

                infos.append(((y, x), (r, g, b)))

        return infos
