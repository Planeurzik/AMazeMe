import time
import numpy as np

"""
This script is a unit test. It also shows the syntax for instantiating Wallace.

Make sure that this script does not crash. Specifically, this should run in a limited time.

Note: this is not the evaluation script that we will use to evaluate your Wallace.
"""

LIMITED_TIME = 5*60

def test_wallace():
    from custom_maze import create_maze
    from solution_a_star import Wallace
    from oracle import predict_best_ratio

    def run(env, wallace, maze):
        n_played_steps = 1234
        obs = env.reset()
        gold = 0.
        done = False
        info = {}
        oracle_position, oracle_ratio = predict_best_ratio(maze)
        ratios = []
        nb_moves = 0
        start = time.perf_counter()
        for _ in range(n_played_steps):
            if done:
                #print(info)
                ratios.append(gold / nb_moves)
                action = wallace.act(obs, gold, done)
                assert action is None
                obs = env.reset()
                gold = 0.
                done = False
                nb_moves = 0
            action = wallace.act(obs, gold, done)
            obs, gold, done, info = env.step(action, render_infos=wallace.get_custom_render_infos())
            nb_moves += 1
        end = time.perf_counter()
        if (end - start) > LIMITED_TIME:
            raise RuntimeError("This run took too much time! (%.2fsec)"%(end-start))
        mean_ratio = np.mean(ratios)
        #print("Mean ratio:", mean_ratio)
        print("Oracle position:", oracle_position, "with ratio", oracle_ratio)
        print("Score:",mean_ratio/oracle_ratio)
        #print(info["monitor.tot_golds"], info["monitor.tot_steps"])

    for exp_idx in range(2):
        env, maze = create_maze(video_prefix="./video_%d"%exp_idx, overwrite_every_episode=False, fps=4)
        wallace = Wallace()
        run(env, wallace, maze)
        env.close()

if __name__ == "__main__":
    test_wallace()
