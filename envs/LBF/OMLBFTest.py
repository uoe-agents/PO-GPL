import argparse
import gym
import random
import lbforaging

if __name__ == '__main__':

    def make_env(env_id, rank,  seed=1285, effective_max_num_players=3, with_shuffle=True):
        def _init():
            env = gym.make(
                env_id, seed=seed + rank,
                effective_max_num_players=effective_max_num_players,
                init_num_players=effective_max_num_players,
                with_shuffle=with_shuffle
            )
            return env

        return _init

    num_players_train = 3
    env = make_env('Adhoc-Foraging-8x8-3f-v0', 3,
        889, num_players_train, True)()

    obs = env.reset()
    actions = []

    for k in range(10000):
        actions = -1
        for ob in obs :
            if obs[ob][-1] == -1 :
                actions = 6
            else :
                actions = random.randint(0,5)
        a, b, c, d = env.step(actions)
        if c:
            break
        if c:
            obs = env.reset()