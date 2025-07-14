# AMaze me

- [AMaze me](#amaze-me)
  - [Folder structure](#folder-structure)
  - [Requirements](#requirements)
  - [Problem setup](#problem-setup)
  - [Evaluation](#evaluation)
  - [Your final submission](#your-final-submission)

## Folder structure

Scripts:
- `maze.py`: procedurally generate interactive/playable 2d mazes
- `wrappers.py`: a set of wrappers for 2d mazes, which help to build the present problem setup, and provide useful logging through videos
- `solution.py`: this is expected to contain your solution, provided through the class Wallace
- `test_wallace.py`: spawn a Wallace inside a maze, and let Wallace wander around

Misc:

- `README.md`
- `requirements.txt`

## Requirements

This exercise requires python3. Also, a few packages are required, install them with:

```
pip3 install -r requirements.txt
```

## Problem setup

You are Wallace, a gold seeker in Mazopotamia, a 2d maze that you completely ignore the layout of. The rumour says that every location is either empty, walled, or has some gold to gather. Unfortunately, you can only look around in the four directions; also when you gather gold, you are magically sent back to the start. You definitely want to find the locations which seem to contain the most gold *on average*, even if that costs you a long (eternal?) life of wandering around and starting back again.

A few clarifications, which can also be deduced from the provided code:

- The maze is randomly generated, with a procedural process which is provided to you.
- You may read the code which generates the maze if this can help you to design your Wallace solution.
- Once a 2d maze is generated, the locations of walls and gold are fixed.
- At every timestep, Wallace knows his absolute coordinates and whether he is standing on gold or not. Also, for each of the 4 directions (top, left, right, bottom), he can see whether there is an adjacent wall. In other words, Wallace can see adjacent cells at a distance of 1, but cannot see if the adjacent cells contain gold.
- At every timestep, Wallace interacts with the maze through 5 possible actions: go up 1 cell, go right 1 cell, go down 1 cell, go left 1 cell, or gather gold from the current cell. His decision is returned by function `act`.
- For each cell that contains gold, gold can be gathered an infinite amount of time (in other words, **it does not run out of gold**). For each of these cells, when Wallace gathers gold, the amount that he actually receives is random i.i.d. After gathering gold from any location, Wallace is sent back to the same start.
- When Wallace tries to gather gold, but the cell has no gold, he gains 0 gold. **Still**, he is sent back to the start.
- Because Wallace does not possess the map of the maze, he cannot and must not read attributes of `env` and `maze` (for instance `maze.layout` or `maze.golds`). His only way to get information from the maze is through the provided variables `obs, gold, done` in function `act`!

Your task is to implement an **efficient** and a relatively **fast running** (eg a few minutes at most) Wallace, for any given random procedural maze.

## Evaluation

Your Wallace should seek to find and collect gold from the maze: he wants a high average amount of gold gained per unit of time. We appreciate solutions which achieve that in a low number of total played actions.

The evaluation procedure is not provided to you, because it is part of the exercise that you figure out the metric and procedure to evaluate that your Wallace can perform well, in a variety of mazes; still, your implemented solution can be tested that it works properly (ie it does not crash in corner cases!) by using the script `test_wallace.py`.

On our side, your code will be reviewed, and run against our benchmarks. You can assume that we use exclusively mazes that may come out of `create_maze`.

## Your final submission

First, check that the `test_wallace.py` that we provided loads your Wallace and runs without crashing.

We expect reproducibility and clarity from your submission. Your final submission should include:

- an explanation of your algorithm
- `solution.py` should contain a class `Wallace` which implements a `act` function, similary to the provided example there
- `maze.py`, `wrappers.py`, and `test_wallace.py`; you may tweak these files for the purpose of running your tests, but we ask you to send back unmodified versions from the ones we originally send you
- if relevant, a `requirements.txt` with the packages that you are using and their versions
- any additional helper files that are required to run your Wallace; for your imports, please carefully read the instructions in `solution.py`
