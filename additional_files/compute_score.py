import time
import numpy as np
from tqdm import tqdm
import math

LIMITED_TIME = 5*60

def test_wallace():
    from custom_maze import create_maze
    from solution_adapted_q_learning import Wallace # change here if you want solution_classic_q_learning
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
        mean_ratio = np.mean(ratios) if len(ratios)>0 else 0
        if oracle_ratio == 0:
            return 0
        #print("Mean ratio:", mean_ratio)
        #print("Oracle position:", oracle_position, "with ratio", oracle_ratio)
        #print("Score:",mean_ratio/oracle_ratio)
        score = mean_ratio / oracle_ratio
        return score
        #print(info["monitor.tot_golds"], info["monitor.tot_steps"])

    scores = {"13":[],
              "21":[],
              "29":[],
            }
    nb_tries = 100

    for i in tqdm(range(3)):
        if i == 0:
            wanted_size = 13
        elif i == 1:
            wanted_size = 21
        else:
            wanted_size = 29
        for j in range(nb_tries):
            env, maze = create_maze(video_prefix="./video", overwrite_every_episode=False, fps=4, random_size=False, wanted_size=wanted_size)
            wallace = Wallace()
            scores[str(wanted_size)].append(run(env, wallace, maze))
            env.close()
    
    for size, score_list in scores.items():
        mean = np.mean(score_list)
        std = np.std(score_list)
        print(f"Mean score for size {size}: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    test_wallace()
