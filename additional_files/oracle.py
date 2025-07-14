from collections import deque
from maze import Maze, procedural_maze
import numpy as np

def shortest_path(grid, start, goal):

    rows, cols = len(grid), len(grid[0])
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    visited = [[False]*cols for _ in range(rows)]
    parent  = [[None]*cols for _ in range(rows)]

    q = deque([start])
    visited[start[0]][start[1]] = True

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            path = []
            while (r, c) != start:
                path.append((r, c))
                r, c = parent[r][c]
            path.append(start)
            return path[::-1] # So the path goes from start to goal

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                not visited[nr][nc] and grid[nr][nc] == 0):
                visited[nr][nc] = True
                parent[nr][nc] = (r, c)
                q.append((nr, nc))

    return []  # No path found

def predict_best_ratio(maze):
    start_pos = (3, 3)

    gold_positions = [(r, c, maze.golds[r, c]) for r, c in np.argwhere(maze.golds > 0)]

    ratio_max = 0
    best_gold = None
    for (r,c,gold) in gold_positions:
        shortest_path_maze = shortest_path(maze.layout, start_pos, (r, c))
        if len(shortest_path_maze) == 0:
            continue
        elif gold/len(shortest_path_maze) > ratio_max:
            ratio_max = gold/len(shortest_path_maze)
            best_gold = (r,c)
        #print(shortest_path(maze.layout, start_pos, (r, c)), gold)
    return best_gold, ratio_max
if __name__ == "__main__":
    sizes = [13,21,29]
    size = np.random.choice(sizes)
    print("Maze size:", size)
    ngolds = 8

    #maze = Maze.load_maze("./super.pkl")
    maze = procedural_maze(size, size, ngolds)
    best_gold, ratio_max = predict_best_ratio(maze)
    print("Best gold position:", best_gold, "with ratio:", ratio_max)