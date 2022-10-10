import argparse
import gym
import random
import lbforaging
import Wolfpack_gym
from gym.vector import SyncVectorEnv

import gym
import torch
import dgl
import random
import string
import json
import numpy as np
from ExpReplay import ExperienceReplay, SequentialExperienceReplay
from torch.utils.tensorboard import SummaryWriter
import os
from SWBBeliefModel import SWBBeliefModel, DecisionMakingModel
from torch import optim, nn
import torch.distributions as dist
from os import listdir
from os.path import isfile, join

import random

parser = argparse.ArgumentParser()

# Experiment logistics
parser.add_argument('--exp-name', type=str, default="exp1", help="Experiment name.")
parser.add_argument('--logging-dir', type=str, default="logs1", help="Tensorboard logging directory")
parser.add_argument('--saving-dir', type=str, default="decoding_params1", help="Parameter saving directory.")

# Dataset sizes
parser.add_argument('--with-data-collection', type=bool, default=False, help="With data collection process/not.")
parser.add_argument('--target-training-steps', type=int, default=6400000, help="Training data size.")
parser.add_argument('--env-name', type=str, default="Adhoc-wolfpack-v5", help="Environment name.")
parser.add_argument('--data-collection-num-envs', type=int, default=16, help="Training seed.")
parser.add_argument('--num-players-train', type=int, default=3, help="Maximum number of players for training.")
parser.add_argument('--num-players-test', type=int, default=5, help="Maximum number of players for testing.")
parser.add_argument('--num-collection-threads', type=int, default=16, help="Number of threads for data collection.")
parser.add_argument('--eps-length', type=int, default=200, help="Maximum episode length for training.")

# Training details
parser.add_argument('--batch-size', type=int, default=16, help="Batch size per updates.")
parser.add_argument('--use-cuda', type=bool, default=True, help="Use CUDA for training or not")
parser.add_argument('--seed', type=int, default=0, help="Training seed.")
parser.add_argument('--eval-init-seed', type=int, default=2500, help="Evaluation seed")
parser.add_argument('--num-particles', type=int, default=20, help="Number of particles for each sample")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for training.")
parser.add_argument('--num-updates', type=int, default=10000, help="Number of GNN updates")
parser.add_argument('--update-period', type=int, default=4, help="Time between updates.")
parser.add_argument('--max-grad-norm', type=float, default=10.0, help="Maximum gradient magnitude for update.")
parser.add_argument('--model-saving-frequency', type=int, default=2500, help="Number of steps before logging")
parser.add_argument('--init-epsilon', type=float, default=1.0, help="Initial exploration rate.")
parser.add_argument('--final-epsilon', type=float, default=0.05, help="Final exploration rate.")
parser.add_argument('--exploration-percentage', type=float, default=0.7,
                    help="Percentage of experiments where exploration is done.")
parser.add_argument('--gamma', type=float, default=0.99, help="Discount rate.")
parser.add_argument('--tau', type=float, default=0.001, help="Polyak averaging rate for target network update.")
parser.add_argument('--target-network-update-frequency', type=int, default=100,
                    help="Number of updates before target network is updated.")
parser.add_argument('--load-from-checkpoint', type=int, default=-1, help="Checkpoint to load parameters from.")

# Loss weights
parser.add_argument('--encoding-weight', type=float, default=0.01, help="Weight associated to encoding/decoding loss.")
parser.add_argument('--act-reconstruction-weight', type=float, default=0.1,
                    help="Weight associated to action reconstruction loss.")
parser.add_argument('--entropy-weight', type=float, default=0.01, help="Weight associated to entropy loss.")
parser.add_argument('--q-loss-weight', type=float, default=1.0, help="Weight associated to value loss.")

# Model size details
parser.add_argument('--act-encoding-size', type=int, default=16, help="Length of action encoding vector.")
parser.add_argument('--hidden-1', type=int, default=100, help="Encoding hidden units 1.")
parser.add_argument('--hidden-2', type=int, default=70, help="Encoding hidden units 2.")
parser.add_argument('--s-dim', type=int, default=100, help="State embedding size.")
parser.add_argument('--h-dim', type=int, default=100, help="Type embedding size.")
parser.add_argument('--gnn-hid-dims1', type=int, default=50, help="GNN hidden dim 1.")
parser.add_argument('--gnn-hid-dims2', type=int, default=100, help="GNN hidden dim 2.")
parser.add_argument('--gnn-hid-dims3', type=int, default=100, help="GNN hidden dim 3.")
parser.add_argument('--gnn-hid-dims4', type=int, default=50, help="GNN hidden dim 4.")
parser.add_argument('--gnn-decoder-hid-dims1', type=int, default=70, help="GNN obs decoder hidden dim 1.")
parser.add_argument('--gnn-decoder-hid-dims2', type=int, default=100, help="GNN obs decoder hidden dim 2.")
parser.add_argument('--gnn-decoder-hid-dims3', type=int, default=50, help="GNN obs decoder hidden dim 3.")
parser.add_argument('--state-hid-dims', type=int, default=70, help="Obs decoder hidden dims.")
parser.add_argument('--state-msg-dims', type=int, default=70, help="Obs decoder message dims.")
parser.add_argument('--mid-pair', type=int, default=70,
                    help="Hidden layer sizes for CG's pairwise utility computation.")
parser.add_argument('--mid-nodes', type=int, default=70,
                    help="Hidden layer sizes for CG's singular utility computation.")
parser.add_argument('--lrf-rank', type=int, default=6,
                    help="Rank for the low rank factorization trick in the CG's pairwise utility computation.")

# Decision making evaluation parameters
parser.add_argument('--num-eval-episodes', type=int, default=3, help="Number of evaluation episodes")
parser.add_argument('--eval-seed', type=int, default=500, help="Number of evaluation episodes")

args = parser.parse_args()


class ModelTraining(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] and torch.cuda.is_available() else "cpu")
        self.env_name = config["env_name"]
        self.epsilon = config["init_epsilon"]

        self.env_kwargs = {}
        if "Foraging" in self.env_name:
            self.env_kwargs = {
                "players": 5,
                "effective_max_num_players": 3,
                "init_num_players": 3,
                "with_shuffle": False,
                "gnn_input": True,
                "with_gnn_shuffle": True
            }

            self.env_kwargs_eval = {
                "players": 5,
                "effective_max_num_players": 5,
                "init_num_players": 5,
                "with_shuffle": False,
                "gnn_input": True,
                "with_gnn_shuffle": True
            }
        elif "wolfpack" in self.env_name:
            self.env_kwargs = {
                "implicit_max_player_num": 3,
                "num_players": 3,
                'max_player_num': 5,
                "rnn_with_gnn": True,
                "close_penalty": 0.5,
                "main_sight_radius": 3
            }

            self.env_kwargs_eval = {
                "implicit_max_player_num": 5,
                'max_player_num': 5,
                "num_players": 3,
                "rnn_with_gnn": True,
                "close_penalty": 0.5,
                "main_sight_radius": 3
            }

    def preprocess(self, obs):
        final_obs = None
        if "Foraging" in self.env_name:
            batch_size = obs["player_info_shuffled"].shape[0]
            player_obs = np.reshape(obs["player_info_shuffled"], (batch_size, self.env_kwargs["players"], -1))
            other_obs = np.repeat(
                np.reshape(obs["food_info"], (batch_size, 1, -1)),
                self.env_kwargs["players"],
                axis=1
            )
            added_data = np.zeros((batch_size, self.env_kwargs["players"], 2))

        elif "wolfpack" in self.env_name:
            batch_size = obs["teammate_location_shuffled"].shape[0]
            player_obs = np.reshape(
                obs["teammate_location_shuffled"], (batch_size, self.env_kwargs["max_player_num"], -1)
            )
            other_obs = np.repeat(
                np.reshape(obs["opponent_info"], (batch_size, 1, -1)),
                self.env_kwargs["max_player_num"],
                axis=1
            )
            added_data = np.zeros((batch_size, self.env_kwargs["max_player_num"], 2))

        all_obs = np.concatenate([player_obs, other_obs], axis=-1)

        # Add feature to distinguish player from others
        added_data[:, 0, 0] = 1

        # Add feature that denotes agent existence
        agent_exists = all_obs[:, :, 0] != -1
        added_data[agent_exists, 1] = 1

        final_obs = np.concatenate([added_data, all_obs], axis=-1)

        return final_obs

    def preprocess_complete(self, obs):
        final_obs = None
        if "Foraging" in self.env_name:
            batch_size = obs["complete_player_info_shuffled"].shape[0]
            player_obs = np.reshape(obs["complete_player_info_shuffled"], (batch_size, self.env_kwargs["players"], -1))
            other_obs = np.repeat(
                np.reshape(obs["food_info_complete"], (batch_size, 1, -1)),
                self.env_kwargs["players"],
                axis=1
            )
            added_data = np.zeros((batch_size, self.env_kwargs["players"], 2))
        elif "wolfpack" in self.env_name:
            batch_size = obs["teammate_location_shuffled_complete"].shape[0]
            player_obs = np.reshape(
                obs["teammate_location_shuffled_complete"], (batch_size, self.env_kwargs["max_player_num"], -1)
            )
            other_obs = np.repeat(
                np.reshape(obs["opponent_info_complete"], (batch_size, 1, -1)),
                self.env_kwargs["max_player_num"],
                axis=1
            )
            added_data = np.zeros((batch_size, self.env_kwargs["max_player_num"], 2))

        all_obs = np.concatenate([player_obs, other_obs], axis=-1)

        # Add feature to distinguish player from others
        added_data[:, 0, 0] = 1

        # Add feature that denotes agent existence
        agent_exists = all_obs[:, :, 0] != -1
        added_data[agent_exists, 1] = 1

        final_obs = np.concatenate([added_data, all_obs], axis=-1)

        return final_obs

    def get_obs_sizes(self, obs_space):
        obs_sizes = None
        agent_existence_offset = 1
        agent_learner_offset = 1

        if "Foraging" in self.env_name:
            obs_sizes = (
                self.env_kwargs["players"],
                (obs_space["player_info"].shape[-1] // self.env_kwargs["players"]) +
                obs_space["food_info"].shape[-1] +
                agent_existence_offset + agent_learner_offset
            )
            return obs_sizes, \
                   (obs_space["player_info"].shape[-1] // self.env_kwargs["players"]) + \
                   agent_existence_offset + agent_learner_offset
        elif "wolfpack" in self.env_name:
            obs_sizes = (
                self.env_kwargs["max_player_num"],
                (obs_space["teammate_location_shuffled"].shape[-1] // self.env_kwargs["max_player_num"]) +
                obs_space["opponent_info"].shape[-1] +
                agent_existence_offset + agent_learner_offset
            )
            return obs_sizes, \
                   (obs_space["teammate_location"].shape[-1] // self.env_kwargs["max_player_num"]) + \
                   agent_existence_offset + agent_learner_offset

    def to_one_hot(self, actions, num_acts):
        one_hot_acts = np.zeros([actions.shape[0], actions.shape[1], num_acts])
        non_zero_entries = (actions != -1)
        indices = np.asarray(actions[non_zero_entries]).astype(int)
        one_hot_acts[non_zero_entries] = np.eye(num_acts)[indices]
        return one_hot_acts

    def log_values(self, writer, log_dict, update_id):
        for key, value in log_dict.items():
            writer.add_scalar(
                key, value, update_id
            )

    def create_directories(self, random_experiment_name):
        if not os.path.exists(self.config["logging_dir"]):
            os.makedirs(self.config["logging_dir"])

        if not os.path.exists(self.config["logging_dir"] + "_grad"):
            os.makedirs(self.config["logging_dir"] + "_grad")

        if not os.path.exists(self.config["saving_dir"]):
            os.makedirs(self.config["saving_dir"])

        directory = os.path.join(self.config['saving_dir'], random_experiment_name)

        if not os.path.exists(self.config["logging_dir"] + "/" + random_experiment_name):
            os.makedirs(self.config["logging_dir"] + "/" + random_experiment_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, 'params.json'), 'w') as json_file:
            json.dump(self.config, json_file)

        with open(
                os.path.join(self.config['logging_dir'], random_experiment_name, 'params.json'), 'w'
        ) as json_file:
            json.dump(self.config, json_file)

    def decide_acts(self, q_values, log_weights, eval=False):
        q_values = q_values.view(self.config["num_collection_threads"], self.config["num_particles"], -1)
        particle_dist = dist.Categorical(logits=log_weights)
        weights = particle_dist.probs.unsqueeze(-1)

        weighted_q_values = (q_values * weights).sum(dim=1)
        acts = torch.argmax(weighted_q_values, dim=-1).tolist()

        if not eval:
            acts = [
                a if random.random() > self.epsilon else random.randint(0, weighted_q_values.shape[-1] - 1) for a in
                acts
            ]

        return acts

    def eval_policy_performance(self, action_shape, model, cg_model, logger, logging_id):

        # Create env for policy eval
        def make_env(env_name, seed, env_kwargs):
            def _make():
                env = gym.make(
                    env_name, seed=seed, **env_kwargs
                )
                return env

            return _make

        env_train = gym.vector.SyncVectorEnv([
            make_env(
                self.config["env_name"], seed=100 * idx + self.config["eval_seed"], env_kwargs=self.env_kwargs
            ) for idx in range(self.config["num_collection_threads"])
        ])

        device = torch.device("cuda" if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
        num_dones = [0] * self.config["num_collection_threads"]
        per_worker_rew = [0.0] * self.config["num_collection_threads"]

        # Initialize initial obs and states for model
        try:
            raw_obs = env_train.reset().item(0)
        except AttributeError:
            # there is a weird bug in gym in which in some cases it returns the orderder dictionary
            raw_obs = env_train.reset()

        test_obses = self.preprocess(raw_obs)
        test_states = self.preprocess_complete(raw_obs)
        initial_states = model.new_latent_state()
        avgs = []

        batch_size, agent_nums = test_obses.shape[0], test_obses.shape[1]

        # Push initial particles to memory
        current_memory = {
            "states": initial_states.to(device),
            "current_obs": torch.tensor(test_obses).double().to(device),
            "actions": torch.zeros([batch_size, action_shape]).double().to(device),
            "rewards": torch.zeros([batch_size, 1]).double().to(device),
            "dones": torch.zeros([batch_size, 1]).double().to(device),
            "other_actions": torch.zeros([batch_size, agent_nums - 1, action_shape]).double().to(device),
            "current_state": torch.tensor(test_states).double().to(device)
        }

        while (any([k < self.config["num_eval_episodes"] for k in num_dones])):
            # Decide agent's action based on model
            out = model(current_memory, with_resample=True, eval=True)
            action_dist = model.predict_action(
                out["others"]["graph"], out["latent_state"]
            )
            q_vals = cg_model(
                out["others"]["graph"], out["latent_state"], "inference", action_dist, None
            )

            old_logits = -out["encoding_losses"]
            acts = self.decide_acts(q_vals, old_logits, eval=True)
            n_obs_raw, rews, dones, infos = env_train.step(acts)

            per_worker_rew = [k + l for k, l in zip(per_worker_rew, rews)]
            act = None
            if "Foraging" in self.config["env_name"]:
                act = np.concatenate(
                    [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["prev_actions_shuffled"]], axis=-1
                )
            elif "wolfpack" in self.config["env_name"]:
                act = np.concatenate(
                    [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["oppo_actions_shuffled"]], axis=-1
                )

            one_hot_acts = self.to_one_hot(act, action_shape)

            # Update observations with next observation
            current_memory["states"] = out["latent_state"].multiply_each(
                (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                    'all_actions', 'all_encoded_actions', 'i', 's', 'theta', 'log_weight', 'cell1', 'cell2', 'recently_seen', 'since_last_seen'
                ]).detach()

            masked_actions = torch.tensor(one_hot_acts).double().to(device) * (
                    1 - torch.Tensor(dones).double().to(device).unsqueeze(-1).unsqueeze(-1).repeat(
                1, one_hot_acts.shape[1], one_hot_acts.shape[2]
            )).detach()

            nob = self.preprocess(n_obs_raw)
            nob_complete = self.preprocess_complete(n_obs_raw)
            reward_tensor = torch.tensor(rews).double().to(device)
            reward_tensor = reward_tensor.view(reward_tensor.shape[-1], -1)

            done_tensor = torch.tensor(dones).double().to(device).double()
            done_tensor = done_tensor.view(done_tensor.shape[-1], -1)

            current_memory["current_obs"] = torch.tensor(nob).double().to(device)
            current_memory["actions"] = masked_actions[:, 0, :]
            current_memory["rewards"] = reward_tensor
            current_memory["other_actions"] = masked_actions[:, 1:, :]
            current_memory["dones"] = done_tensor
            current_memory["current_state"] = torch.tensor(nob_complete).double().to(device)

            for idx, flag in enumerate(dones):
                if flag:
                    if num_dones[idx] < self.config['num_eval_episodes']:
                        num_dones[idx] += 1
                        avgs.append(per_worker_rew[idx])
                    per_worker_rew[idx] = 0

        avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
        print("Finished train with rewards " + str(avg_total_rewards))
        env_train.close()
        logger.add_scalar('Rewards/train_set', sum(avgs) / len(avgs), logging_id)

        env_eval = gym.vector.SyncVectorEnv([
            make_env(
                self.config["env_name"], seed=100 * idx + self.config["eval_seed"], env_kwargs=self.env_kwargs_eval
            ) for idx in range(self.config["num_collection_threads"])
        ])

        device = torch.device("cuda" if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
        num_dones = [0] * self.config["num_collection_threads"]
        per_worker_rew = [0.0] * self.config["num_collection_threads"]
        avgs = []

        raw_obs = env_eval.reset().item(0)
        test_obses = self.preprocess(raw_obs)
        test_states = self.preprocess_complete(raw_obs)
        initial_states = model.new_latent_state()

        batch_size, agent_nums = test_obses.shape[0], test_obses.shape[1]

        # Push initial particles to memory
        current_memory = {
            "states": initial_states.to(device),
            "current_obs": torch.tensor(test_obses).double().to(device),
            "actions": torch.zeros([batch_size, action_shape]).double().to(device),
            "rewards": torch.zeros([batch_size, 1]).double().to(device),
            "dones": torch.zeros([batch_size, 1]).double().to(device),
            "other_actions": torch.zeros([batch_size, agent_nums - 1, action_shape]).double().to(device),
            "current_state": torch.tensor(test_states).double().to(device)
        }

        # Initialize initial obs and states for model
        while (any([k < self.config["num_eval_episodes"] for k in num_dones])):

            # Decide agent's action based on model
            out = model(current_memory, with_resample=True, eval=True)
            action_dist = model.predict_action(
                out["others"]["graph"], out["latent_state"]
            )
            q_vals = cg_model(
                out["others"]["graph"], out["latent_state"], "inference", action_dist, None
            )

            old_logits = -out["encoding_losses"]
            acts = self.decide_acts(q_vals, old_logits, eval=True)
            n_obs_raw, rews, dones, infos = env_eval.step(acts)

            per_worker_rew = [k + l for k, l in zip(per_worker_rew, rews)]
            act = None
            if "Foraging" in self.config["env_name"]:
                act = np.concatenate(
                    [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["prev_actions_shuffled"]], axis=-1
                )
            elif "wolfpack" in self.config["env_name"]:
                act = np.concatenate(
                    [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["oppo_actions_shuffled"]], axis=-1
                )

            one_hot_acts = self.to_one_hot(act, action_shape)

            # Update observations with next observation
            current_memory["states"] = out["latent_state"].multiply_each(
                (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                    'all_actions', 'all_encoded_actions', 'i', 's', 'theta', 'log_weight', 'cell1', 'cell2', 'recently_seen', 'since_last_seen'
                ]).detach()

            masked_actions = torch.tensor(one_hot_acts).double().to(device) * (
                    1 - torch.Tensor(dones).double().to(device).unsqueeze(-1).unsqueeze(-1).repeat(
                1, one_hot_acts.shape[1], one_hot_acts.shape[2]
            )).detach()

            nob = self.preprocess(n_obs_raw)
            nob_complete = self.preprocess_complete(n_obs_raw)
            reward_tensor = torch.tensor(rews).double().to(device)
            reward_tensor = reward_tensor.view(reward_tensor.shape[-1], -1)

            done_tensor = torch.tensor(dones).double().to(device).double()
            done_tensor = done_tensor.view(done_tensor.shape[-1], -1)

            current_memory["current_obs"] = torch.tensor(nob).double().to(device)
            current_memory["actions"] = masked_actions[:, 0, :]
            current_memory["rewards"] = reward_tensor
            current_memory["other_actions"] = masked_actions[:, 1:, :]
            current_memory["dones"] = done_tensor
            current_memory["current_state"] = torch.tensor(nob_complete).double().to(device)

            for idx, flag in enumerate(dones):
                if flag:
                    if num_dones[idx] < self.config['num_eval_episodes']:
                        num_dones[idx] += 1
                        avgs.append(per_worker_rew[idx])
                    per_worker_rew[idx] = 0

        avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
        print("Finished eval with rewards " + str(avg_total_rewards))
        env_eval.close()
        logger.add_scalar('Rewards/eval_set', sum(avgs) / len(avgs), logging_id)

    def run(self):
        def randomString(stringLength=10):
            letters = string.ascii_lowercase
            return ''.join(random.choice(letters) for i in range(stringLength))

        random_experiment_name = self.config["exp_name"]
        if random_experiment_name == None:
            random_experiment_name = randomString(10)

        self.create_directories(random_experiment_name)
        writer = SummaryWriter(log_dir=self.config["logging_dir"] + "/" + random_experiment_name)
        writer2 = SummaryWriter(log_dir=self.config["logging_dir"] + "_grad" + "/" + random_experiment_name)

        env1 = gym.make(
            self.config["env_name"], **self.env_kwargs
        )

        obs_sizes, agent_o_size = self.get_obs_sizes(env1.observation_space)
        act_sizes = [env1.action_space.n]

        def make_env(env_name, seed, env_kwargs):
            def _make():
                env = gym.make(
                    env_name, seed=seed, **env_kwargs
                )
                return env

            return _make

        env = gym.vector.SyncVectorEnv([
            make_env(
                self.config["env_name"], seed=100 * idx, env_kwargs=self.env_kwargs
            ) for idx in range(self.config["num_collection_threads"])
        ])

        num_steps = self.config["target_training_steps"] // self.config["num_collection_threads"]
        num_episodes = num_steps // self.config["eps_length"]

        try:
            test_obses = self.preprocess(env.reset().item(0))
        except AttributeError:
            # there is a weird bug in gym in which in some cases it returns the orderder dictionary
            test_obses = self.preprocess(env.reset())

        device = torch.device("cuda" if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
        obs_size = test_obses.shape[-1]

        action_shape = None
        action_type = None
        if env1.action_space.__class__.__name__ == "Discrete":
            action_shape = env1.action_space.n
            action_type = "discrete"
        else:
            action_shape = env1.shape[0]
            action_type = "continuous"

        num_agents = None
        if "Foraging" in self.config["env_name"]:
            num_agents = self.env_kwargs["players"]
        elif "wolfpack" in self.config["env_name"]:
            num_agents = self.env_kwargs["max_player_num"]

        # Initialize belief prediction model
        model = SWBBeliefModel(
            action_space=env1.action_space,
            nr_inputs=obs_size,
            agent_inputs=agent_o_size,
            u_inputs=obs_size - agent_o_size,
            action_encoding=self.config["act_encoding_size"],
            cnn_channels=[self.config["hidden_1"], self.config["hidden_2"]],
            s_dim=self.config["s_dim"],
            theta_dim=self.config["h_dim"],
            gnn_act_pred_dims=[
                self.config["gnn_hid_dims1"],
                self.config["gnn_hid_dims2"],
                self.config["gnn_hid_dims3"],
                self.config["gnn_hid_dims4"]
            ],
            gnn_state_hid_dims=[
                self.config["gnn_hid_dims1"],
                self.config["gnn_hid_dims2"],
                self.config["gnn_hid_dims3"],
                self.config["gnn_hid_dims4"]
            ],
            gnn_type_update_hid_dims=[
                self.config["gnn_hid_dims1"],
                self.config["gnn_hid_dims2"],
                self.config["gnn_hid_dims3"],
                self.config["gnn_hid_dims4"]
            ],
            gnn_decoder_hid_dims=[
                self.config["gnn_decoder_hid_dims1"],
                self.config["gnn_decoder_hid_dims1"],
                self.config["gnn_decoder_hid_dims1"]
            ],
            state_hid_dims=self.config["state_hid_dims"],
            state_message_dims=self.config["state_msg_dims"],
            encoder_batch_norm=False,
            batch_size=self.config["num_collection_threads"],
            num_particles=self.config["num_particles"],
            num_agents=num_agents,
            resample=True,
            device=device
        )
        cg_model = DecisionMakingModel(
            action_space=env1.action_space,
            s_dim=self.config["s_dim"],
            theta_dim=self.config["h_dim"],
            device=device,
            mid_pair=self.config["mid_pair"],
            mid_nodes=self.config["mid_nodes"],
            mid_pair_out=self.config["lrf_rank"]
        )

        target_cg_model = DecisionMakingModel(
            action_space=env1.action_space,
            s_dim=self.config["s_dim"],
            theta_dim=self.config["h_dim"],
            device=device,
            mid_pair=self.config["mid_pair"],
            mid_nodes=self.config["mid_nodes"],
            mid_pair_out=self.config["lrf_rank"]
        )

        if self.config["load_from_checkpoint"] == -1:
            hard_copy(target_cg_model, cg_model)

            # TODO
            model, cg_model, target_cg_model = model.double(), cg_model.double(), target_cg_model.double()
            model_optimizer = optim.Adam(list(model.parameters()) + list(cg_model.parameters()), lr=self.config["lr"])

            torch.save(model.state_dict(),
                       self.config["saving_dir"] + "/" + random_experiment_name + "/" + "0.pt")

            torch.save(cg_model.state_dict(),
                       self.config["saving_dir"] + "/" + random_experiment_name + "/" + "0-cg.pt")

            torch.save(target_cg_model.state_dict(),
                       self.config["saving_dir"] + "/" + random_experiment_name + "/" + "0-tar-cg.pt")

            torch.save(model_optimizer.state_dict(),
                       self.config["saving_dir"] + "/" + random_experiment_name + "/" + "0-optim.pt")

            self.eval_policy_performance(action_shape, model, cg_model, writer, 0)
        else:
            model.load_state_dict(
                torch.load(self.config["saving_dir"] + "/" + random_experiment_name + "/" + str(
                    self.config["load_from_checkpoint"]
                ) + ".pt")
            )
            cg_model.load_state_dict(
                torch.load(self.config["saving_dir"] + "/" + random_experiment_name + "/" + str(
                    self.config["load_from_checkpoint"]
                ) + "-cg.pt")
            )
            target_cg_model.load_state_dict(
                torch.load(self.config["saving_dir"] + "/" + random_experiment_name + "/" + str(
                    self.config["load_from_checkpoint"]
                ) + "-tar-cg.pt")
            )

            model, cg_model, target_cg_model = model.double(), cg_model.double(), target_cg_model.double()
            model_optimizer = optim.Adam(list(model.parameters()) + list(cg_model.parameters()), lr=self.config["lr"])

            model_optimizer.load_state_dict(
                torch.load(self.config["saving_dir"] + "/" + random_experiment_name + "/" + str(
                    self.config["load_from_checkpoint"]
                ) + "-optim.pt")
            )

        updates_per_episode = self.config["eps_length"] // self.config["update_period"]

        start_point = self.config["load_from_checkpoint"]
        if self.config["load_from_checkpoint"] == -1:
            start_point = 0
        total_updates = start_point * self.config["model_saving_frequency"]
        start_ep_id = total_updates // updates_per_episode
        all_num_steps = 0

        # Make updates based on sequential data
        for ep_id in range(start_ep_id, num_episodes):
            # Preprocess sequential data into input for network.
            raw_obs = env.reset().item(0)
            test_obses = self.preprocess(raw_obs)
            test_states = self.preprocess_complete(raw_obs)
            initial_states = model.new_latent_state()

            batch_size, agent_nums = test_obses.shape[0], test_obses.shape[1]

            # Push initial particles to memory
            current_memory = {
                "states": initial_states.to(device),
                "current_obs": torch.tensor(test_obses).double().to(device),
                "actions": torch.zeros([batch_size, action_shape]).double().to(device),
                "rewards": torch.zeros([batch_size, 1]).double().to(device),
                "dones": torch.zeros([batch_size, 1]).double().to(device),
                "other_actions": torch.zeros([batch_size, agent_nums - 1, action_shape]).double().to(device),
                "current_state": torch.tensor(test_states).double().to(device)
            }

            idx = 0

            # For optimization
            total_loss = 0
            total_q_loss = 0

            # For logging
            total_action_prediction_log_prob = 0
            total_particle_entropy = 0
            total_state_log_prob = 0
            total_proposed_state_log_prob = 0
            total_agent_obs_visibility_log_prob = 0
            total_agent_features_log_prob = 0
            total_u_features_log_prob = 0
            total_num_killed_particles = 0
            total_num_existing_agents = 0
            total_state_teammate_log_prob = 0
            total_weighted_rmse = 0

            steps_since_log = 0
            self.epsilon = 1.0 - (
                min((ep_id + 0.0) / (self.config["exploration_percentage"] * num_episodes), 1.0)
            ) * 0.95

            # Compute per step computations
            while idx < self.config["eps_length"]:
                all_num_steps += 1
                # print(all_num_steps)
                idx += 1
                steps_since_log += 1
                out = model(current_memory, with_resample=True)
                action_dist = model.predict_action(
                    out["others"]["graph"], out["latent_state"].detach()
                )
                q_vals = cg_model(
                    out["others"]["graph"], out["latent_state"], "inference", action_dist, None
                )

                old_logits = -out["encoding_losses"]
                acts = self.decide_acts(q_vals, -out["encoding_losses"])
                n_obs_raw, rews, dones, infos = env.step(acts)

                act = None
                if "Foraging" in self.config["env_name"]:
                    act = np.concatenate(
                        [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["prev_actions_shuffled"]], axis=-1
                    )
                elif "wolfpack" in self.config["env_name"]:
                    act = np.concatenate(
                        [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["oppo_actions_shuffled"]], axis=-1
                    )

                one_hot_acts = self.to_one_hot(act, action_shape)
                recently_executed_actions = torch.tensor(one_hot_acts).double().to(device)

                cg_predicted_values = cg_model(
                    out["others"]["graph"], out["latent_state"],
                    "train", action_dist,
                    recently_executed_actions
                )

                predicted_values = cg_predicted_values.view(
                    self.config["num_collection_threads"], self.config["num_particles"]
                )

                total_num_killed_particles += out["num_killed_particles"]
                total_num_existing_agents += out["others"]["num_existing_agents"]

                # Compute per batch negative log likelihood of the data and average it out
                avg_encoding_loss = out["total_encoding_loss"].mean()

                # Compute average logits for edge prediction
                # This is important since gumbel softmax, much like the usual logits for policy gradient
                # Has a tendency to collapse to a Categorical distribution that is deterministic (probs
                # close to 0 to a certain choice).
                # Configure the weights of this loss in case logits collapse to a deterministic value
                # (e.g. large norm_edge logits)

                # Encoding/Decoding loss
                total_loss += self.config["encoding_weight"] * avg_encoding_loss
                print("Encoding element  : ", self.config["encoding_weight"] * avg_encoding_loss)

                avg_act_log_prob_loss = out["others"]["action_reconstruction_log_prob"]
                # Add action reconstruction losses
                total_loss += self.config["act_reconstruction_weight"] * -avg_act_log_prob_loss
                print(
                    "Action prediction loss  : ",
                    self.config["act_reconstruction_weight"] * -avg_act_log_prob_loss
                )

                # Add log probability of action reconstruction to logging
                total_action_prediction_log_prob += avg_act_log_prob_loss

                print(avg_encoding_loss, avg_act_log_prob_loss)

                # Get states that should be reconstructed
                prev_state = current_memory["current_obs"]
                expanded_state = prev_state.unsqueeze(1).repeat(1, self.config["num_particles"], 1, 1)
                reconstructed_state = torch.cat([expanded_state[:, :, :, :1], expanded_state[:, :, :, 2:]], dim=-1)

                # Get weights of each particle in a batch
                particle_dist = dist.Categorical(logits=-out["encoding_losses"])
                weights = particle_dist.probs
                entropy = particle_dist.entropy()

                # Log entropy following Categorical distribution from particle weight
                total_particle_entropy += entropy.mean()

                # Calculate weighted obs reconstruction error
                mean_obs_feat = out["others"]["obs_feature_dist_mean"]
                rmse = (((mean_obs_feat - reconstructed_state) ** 2).sum(dim=-1)).mean(dim=-1)
                total_weighted_rmse += (rmse * weights).sum(dim=-1).mean()

                # Log encoding loss components
                total_state_log_prob += out["others"]["state_log_prob"].mean(dim=-1).mean(dim=-1).mean()
                total_proposed_state_log_prob += out["others"]["proposed_state_log_prob"].mean(dim=-1).mean(
                    dim=-1).mean()
                total_agent_obs_visibility_log_prob += out["others"]["agent_obs_visibility_log_prob"].mean(dim=-1).mean(
                    dim=-1).mean()
                total_agent_features_log_prob += out["others"]["agent_features_log_prob"].mean(dim=-1).mean(
                    dim=-1).mean()
                total_u_features_log_prob += out["others"]["u_features_log_prob"].sum(dim=-1).mean()

                # Update observation with next observation
                current_memory["states"] = out["latent_state"].multiply_each(
                    (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                        'all_actions', 'all_encoded_actions', 'i', 's', 'theta', 'log_weight', 'cell1', 'cell2', 'recently_seen', 'since_last_seen'
                    ])

                masked_actions = torch.tensor(one_hot_acts).double().to(device) * (
                        1 - torch.Tensor(dones).double().to(device).unsqueeze(-1).unsqueeze(-1).repeat(
                    1, one_hot_acts.shape[1], one_hot_acts.shape[2]
                ))

                nob = self.preprocess(n_obs_raw)
                nob_complete = self.preprocess_complete(n_obs_raw)
                reward_tensor = torch.tensor(rews).double().to(device)
                reward_tensor = reward_tensor.view(reward_tensor.shape[-1], -1)

                done_tensor = torch.tensor(dones).double().to(device).double()
                done_tensor = done_tensor.view(done_tensor.shape[-1], -1)

                current_memory["current_obs"] = torch.tensor(nob).double().to(device)
                current_memory["actions"] = masked_actions[:, 0, :]
                current_memory["rewards"] = reward_tensor
                current_memory["other_actions"] = masked_actions[:, 1:, :]
                current_memory["dones"] = done_tensor
                current_memory["current_state"] = torch.tensor(nob_complete).double().to(device)

                out = model(current_memory, with_resample=False)
                action_dist = model.predict_action(
                    out["others"]["graph"], out["latent_state"].detach()
                )
                target_q_vals = target_cg_model(
                    out["others"]["graph"], out["latent_state"].detach(),
                    "inference", action_dist, None
                )

                # Process target q values by weighting them based on particle weights
                target_q_vals = target_q_vals.view(
                    self.config["num_collection_threads"], self.config["num_particles"], -1
                )

                weighted_target_q_values = (target_q_vals.max(dim=-1)[0])
                rews = reward_tensor.repeat([1, self.config["num_particles"]])
                dons = done_tensor.repeat([1, self.config["num_particles"]])
                all_target_values = rews + self.config["gamma"] * (1 - dons) * weighted_target_q_values

                old_weights = dist.Categorical(logits=old_logits).probs
                q_loss = (old_weights.detach() * ((predicted_values - all_target_values.detach()) ** 2)).sum(
                    dim=-1).mean()

                total_loss += self.config["q_loss_weight"] * q_loss
                total_q_loss += q_loss
                print(
                    "Value loss : ",
                    self.config["q_loss_weight"] * q_loss
                )

                if idx % self.config["update_period"] == 0:
                    model_optimizer.zero_grad()
                    total_loss.backward()
                    total_updates += 1
                    # Clip grads if necessary
                    if self.config['max_grad_norm'] > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), self.config['max_grad_norm'])
                        nn.utils.clip_grad_norm_(cg_model.parameters(), self.config['max_grad_norm'])

                    for name, param in model.named_parameters():
                        if not param.grad is None:
                            writer2.add_scalar(
                                name, torch.abs(param.grad).mean(), total_updates
                            )
                    model_optimizer.step()

                    # Prevents gradients from propagating further than self.config["update_period"] steps
                    current_memory['states'] = current_memory['states'].detach()

                    log_dict = {
                        "total_loss": total_loss,
                        "total_q_loss": (total_q_loss + 0.0) / steps_since_log,
                        "total_action_prediction_log_prob": (total_action_prediction_log_prob + 0.0) / steps_since_log,
                        "total_particle_entropy": (total_particle_entropy + 0.0) / steps_since_log,
                        "total_weighted_rmse": (total_weighted_rmse + 0.0) / steps_since_log,
                        "total_state_log_prob": (total_state_log_prob + 0.0) / steps_since_log,
                        "total_proposed_state_log_prob": (total_proposed_state_log_prob + 0.0) / steps_since_log,
                        "total_agent_obs_visibility_log_prob": (
                                                                           total_agent_obs_visibility_log_prob + 0.0) / steps_since_log,
                        "total_agent_features_log_prob": (total_agent_features_log_prob + 0.0) / steps_since_log,
                        "total_u_features_log_prob": (total_u_features_log_prob + 0.0) / steps_since_log,
                        "total_num_killed_particles": (total_num_killed_particles + 0.0) / steps_since_log,
                        "total_num_existing_agents": (total_num_existing_agents + 0.0) / steps_since_log,
                        "total_state_teammate_log_prob": (total_state_teammate_log_prob + 0.0) / steps_since_log
                    }
                    self.log_values(writer, log_dict, total_updates)

                    # Reset values
                    # For optimization
                    total_loss = 0
                    total_q_loss = 0

                    # For logging
                    total_action_prediction_log_prob = 0
                    total_particle_entropy = 0
                    total_weighted_rmse = 0
                    total_state_log_prob = 0
                    total_proposed_state_log_prob = 0
                    total_agent_obs_visibility_log_prob = 0
                    total_agent_features_log_prob = 0
                    total_u_features_log_prob = 0
                    total_num_killed_particles = 0
                    total_num_existing_agents = 0
                    total_state_teammate_log_prob = 0

                    steps_since_log = 0
                    soft_copy(target_cg_model, cg_model, tau=self.config["tau"])

                    # Save model every once in a while
                    if total_updates % self.config["model_saving_frequency"] == 0:
                        torch.save(model.state_dict(),
                                   self.config["saving_dir"] + "/" + random_experiment_name + "/" + str(
                                       total_updates // self.config["model_saving_frequency"]) + ".pt")

                        torch.save(cg_model.state_dict(),
                                   self.config["saving_dir"] + "/" + random_experiment_name + "/" + str(
                                       total_updates // self.config["model_saving_frequency"]) + "-cg.pt")

                        torch.save(target_cg_model.state_dict(),
                                   self.config["saving_dir"] + "/" + random_experiment_name + "/" + str(
                                       total_updates // self.config["model_saving_frequency"]) + "-tar-cg.pt")

                        torch.save(model_optimizer.state_dict(),
                                   self.config["saving_dir"] + "/" + random_experiment_name + "/" + str(
                                       total_updates // self.config["model_saving_frequency"]) + "-optim.pt")

                        self.eval_policy_performance(
                            action_shape, model, cg_model, writer,
                            total_updates // self.config["model_saving_frequency"]
                        )


def hard_copy(target_cg, cg):
    for target_param, param in zip(target_cg.parameters(), cg.parameters()):
        target_param.data.copy_(param.data)


def soft_copy(target_net, source_net, tau=0.001):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param + (1 - tau) * target_param)


if __name__ == '__main__':
    args = vars(args)
    model_trainer = ModelTraining(args)
    model_trainer.run()
