import heapq
import numpy as np
from maze import procedural_maze
from collections import deque

def shortest_path(grid, start, goal):

    rows, cols = len(grid), len(grid[0])
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    visited = [[False]*cols for _ in range(rows)]
    parent  = [[None]*cols for _ in range(rows)]

    q = deque([start])
    visited[start[0]][start[1]] = True
    count = 0
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            path = []
            while (r, c) != start:
                path.append((r, c))
                r, c = parent[r][c]
            path.append(start)
            print("Number of iterations for Dijkstra:",count)
            return path[::-1] # So the path goes from start to goal

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                not visited[nr][nc] and grid[nr][nc] == 0):
                visited[nr][nc] = True
                parent[nr][nc] = (r, c)
                q.append((nr, nc))
        count += 1

    return []  # No path found

def a_star(grid, start, goal):

    def heuristic(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    rows, cols = len(grid), len(grid[0])
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    # Cost from start to each cell (g)
    g_cost = [[float('inf')] * cols for _ in range(rows)]
    g_cost[start[0]][start[1]] = 0

    # Parent for path reconstruction
    parent = [[None] * cols for _ in range(rows)]

    # Priority queue for open set: (f, g, (r,c))
    # f = g + h
    pq = []
    heapq.heappush(pq, (heuristic(start, goal), 0, start))
    count = 0
    while pq:
        f, g, (r,c) = heapq.heappop(pq)
        if (r, c) == goal:
            path = []
            while (r, c) != start:
                path.append((r, c))
                r, c = parent[r][c]
            path.append(start)
            #print("Number of iterations for A*:",count)
            return path[::-1]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                new_g = g + 1
                if new_g < g_cost[nr][nc]:
                    g_cost[nr][nc] = new_g
                    parent[nr][nc] = (r, c)
                    f = new_g + heuristic((nr, nc), goal)
                    heapq.heappush(pq, (f, new_g, (nr, nc)))
        count+=1
        
    return []

if __name__ == "__main__":
    sizes = [13,21,29]
    size = 29 #np.random.choice(sizes)
    print("Maze size:", size)
    ngolds = 8

    #maze = Maze.load_maze("./super.pkl")
    maze = procedural_maze(size, size, ngolds)
    maze.save_maze("./a_star")

    start = (3, 3)
    goal = (27, 3)
    shortest_path_maze = a_star(maze.layout, start, goal)
    test_path = shortest_path(maze.layout, start, goal)

    print(test_path)
    print(shortest_path_maze)