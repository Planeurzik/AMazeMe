"""
This file contains the classic Q-learning approach.
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
        self.sequence = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.idx = 0
        self.Q_table = np.ones((30,30,5))*5
        self.lr = 0.1
        self.decay_rate = 0.8
        self.epsilon = 0.8
        self.gamma = 0.99
        self.previous_state = None
        self.previous_action = None
    
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
        y, x, top, left, right, bottom, has_gold = obs
        if self.previous_state is not None:
            y_previous, x_previous = self.previous_state
        else:
            action = np.random.choice(list(Action))
            self.previous_state = (y, x)
            self.previous_action = action
            return action
        gold = gold_received_at_previous_step
        self.idx += 1
        reward = 0
        if done and has_gold:
            reward = gold
        elif done and not has_gold:
            reward = -10
        elif has_gold:
            reward = 1
        else:
            reward = 0
        self.Q_table[y_previous, x_previous, self.previous_action] += self.lr * (reward + (self.gamma*np.max(self.Q_table[y,x,:])) - self.Q_table[y_previous, x_previous, self.previous_action])
        action = self.epsilon_greedy(y,x)
        self.previous_state = (y, x)
        self.previous_action = action
        if done:
            self.idx = 0
            self.epsilon *= self.decay_rate
            return None
        return action

    def get_custom_render_infos(self):
        """If it helps you for debugging your Wallace and/or explaining its behavior, you can modify this function 
        so that it returns a list of tuple of the form (pixel location, RGB color) e.g. [((y1,x1), (r1,g1,b1)), ((y2,x2), (r2,g2,b2))]
        
        These pixel colors will overwrite the default colors used to generate Wallace's gameplay video (except Wallace's current location and fog of war)"""
        return None
    
    def get_max_q_values_infos(self):
        """Visualize the Q-table using RGB colors.
        Cells with higher max Q-values will appear more red.
        """
        infos = []
        # Normalize Q-values to [0, 1] for colormap scaling
        max_q = np.max(self.Q_table)
        min_q = np.min(self.Q_table)

        if max_q == min_q:
            return infos  # All Q-values are equal; no point in coloring

        norm = lambda q: (q - min_q) / (max_q - min_q + 1e-8)

        height, width, _ = self.Q_table.shape
        for y in range(height):
            for x in range(width):
                max_qval = np.max(self.Q_table[y, x, :])
                value = norm(max_qval)

                # Map to RGB: higher value = more red
                r = 0
                g = int(255 * (1 - value))
                b = int(255 * value)

                infos.append(((y, x), (r, g, b)))

        return infos

    def get_min_q_values_infos(self):
        infos = []

        # Extract minimum Q-values at each cell
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
