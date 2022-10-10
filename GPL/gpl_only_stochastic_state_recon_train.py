import argparse
import gym
import random
import lbforaging
import Wolfpack_gym
import coopnavigation
import FortAttack_gym
from gym.vector import AsyncVectorEnv as VectorEnv

import math
import gym
import torch
import dgl
import random
import string
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from arguments import get_args
import os
from RNNTypeModel import GPLTypeInferenceModelStateReconsStochastic, ParticledGPLDecisionMakingModel
from torch import optim, nn
import torch.distributions as dist

import random

parser = argparse.ArgumentParser()

# Experiment logistics
parser.add_argument('--exp-name', type=str, default="exp1", help="Experiment name.")
parser.add_argument('--logging-dir', type=str, default="logs1", help="Tensorboard logging directory")
parser.add_argument('--saving-dir', type=str, default="decoding_params1", help="Parameter saving directory.")

# Dataset sizes
parser.add_argument('--target-training-steps', type=int, default=16000000, help="Number of experiences for training.")
parser.add_argument('--env-name', type=str, default="Adhoc-wolfpack-v5", help="Environment name.")
parser.add_argument('--num-collection-threads', type=int, default=16, help="Number of parallel threads for data collection during training.")
parser.add_argument('--num-players-train', type=int, default=3, help="Maximum number of players for training.")
parser.add_argument('--num-players-test', type=int, default=5, help="Maximum number of players for testing.")
parser.add_argument('--eps-length', type=int, default=200, help="Maximum episode length for training.")

# Training details
parser.add_argument('--batch-size', type=int, default=16, help="Batch size per updates.")
parser.add_argument('--use-cuda', type=bool, default=True, help="Use CUDA for training or not")
parser.add_argument('--seed', type=int, default=0, help="Training seed.")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for training.")
parser.add_argument('--update-period', type=int, default=4, help="Time between updates.")
parser.add_argument('--max-grad-norm', type=float, default=10.0, help="Maximum gradient magnitude for update.")
parser.add_argument('--model-saving-frequency', type=int, default=2500, help="Number of steps before logging")
parser.add_argument('--init-epsilon', type=float, default=1.0, help="Initial exploration rate.")
parser.add_argument('--final-epsilon', type=float, default=0.05, help="Final exploration rate.")
parser.add_argument('--exploration-percentage', type=float, default=0.7, help="Percentage of experiment where epsilon annealing is done.")
parser.add_argument('--gamma', type=float, default=0.99, help="Discount rate.")
parser.add_argument('--tau', type=float, default=0.001, help="Polyak averaging rate for target network update.")
parser.add_argument('--target-network-update-frequency', type=int, default=100, help="Number of updates before target network is updated (applied when using hard target network updates).")
parser.add_argument('--load-from-checkpoint', type=int, default=-1, help="Checkpoint to load parameters from.")
parser.add_argument('--num-particles', type=int, default=10, help="Num of particles for action value computation")

# Loss weights
parser.add_argument('--act-reconstruction-weight', type=float, default=0.1, help="Weight associated to action reconstruction loss.")
parser.add_argument('--states-reconstruction-weight', type=float, default=0.01, help="Weight associated to state reconstruction loss.")
parser.add_argument('--agent-existence-reconstruction-weight', type=float, default=0.2, help="Weight associated to agent existence prediction loss.")
parser.add_argument('--q-loss-weight', type=float, default=1.0, help="Weight associated to value loss.")
parser.add_argument('--kl-div-loss-weight', type=float, default=0.1, help="Weight associated to KL Loss.")

# Model size details
parser.add_argument('--act-encoding-size', type=int, default=16, help="Length of action encoding vector.")
parser.add_argument('--hidden-1', type=int, default=100, help="Encoding hidden units 1.")
parser.add_argument('--hidden-2', type=int, default=70, help="Encoding hidden units 2.")
parser.add_argument('--s-dim', type=int, default=128, help="State embedding size.")
parser.add_argument('--h-dim', type=int, default=128, help="Type embedding size.")
parser.add_argument('--gnn-hid-dims1', type=int, default=64, help="GNN hidden dim 1.")
parser.add_argument('--gnn-hid-dims2', type=int, default=128, help="GNN hidden dim 2.")
parser.add_argument('--gnn-hid-dims3', type=int, default=128, help="GNN hidden dim 3.")
parser.add_argument('--gnn-hid-dims4', type=int, default=64, help="GNN hidden dim 4.")
parser.add_argument('--gnn-decoder-hid-dims1', type=int, default=70, help="GNN obs decoder hidden dim 1.")
parser.add_argument('--gnn-decoder-hid-dims2', type=int, default=128, help="GNN obs decoder hidden dim 2.")
parser.add_argument('--gnn-decoder-hid-dims3', type=int, default=64, help="GNN obs decoder hidden dim 3.")
parser.add_argument('--state-hid-dims', type=int, default=70, help="Obs decoder hidden dims.")
parser.add_argument('--mid-pair', type=int, default=70, help="Hidden layer sizes for CG's pairwise utility computation.")
parser.add_argument('--mid-nodes', type=int, default=70, help="Hidden layer sizes for CG's singular utility computation.")
parser.add_argument('--lrf-rank', type=int, default=10, help="Rank for the low rank factorization trick in the CG's pairwise utility computation.")
parser.add_argument('--separate-types', type=bool, default=False, help="Whether to use separate types/not.")

# Decision making evaluation parameters
parser.add_argument('--num-eval-episodes', type=int, default=3, help="Number of evaluation episodes")
parser.add_argument('--eval-seed', type=int, default=500, help="Seed for evaluation.")

# Additional arguments for FAttack
parser.add_argument('--obs-mode', type=str, default="conic", help="Type of observation function for FAttack (either conic/circular).")
parser.add_argument('--cone-angle', type=float, default=math.pi, help="Obs cone angle for FAttack")
parser.add_argument('--vision-radius', type=float, default=2.0, help="Vis radius for FAttack")

args = parser.parse_args()

class GPLOnlyModelTraining(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] and torch.cuda.is_available() else "cpu")
        self.env_name = config["env_name"]
        self.epsilon = config["init_epsilon"]

        self.env_kwargs = {}
        self.env_kwargs_eval = {}
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
        if "Navigation" in self.env_name:
            self.env_kwargs = {
                "players": 5,
                "effective_max_num_players": 3,
                "init_num_players": 3,
                "with_shuffle": False,
                "gnn_input": True,
                "with_gnn_shuffle": True,
                "designated_device":"cpu",
                "disappearance_prob": 0.,
                "perturbation_prob": [1.,0.,0.]
            }

            self.env_kwargs_eval = {
                "players": 5,
                "effective_max_num_players": 5,
                "init_num_players": 5,
                "with_shuffle": False,
                "gnn_input": True,
                "with_gnn_shuffle": True,
                "designated_device":"cpu",
                "disappearance_prob": 0.,
                "perturbation_prob": [1.,0.,0.]
            }
        elif "wolfpack" in self.env_name:
            self.env_kwargs = {
                "implicit_max_player_num": 3,
                "num_players": 3,
                'max_player_num': 5,
                "rnn_with_gnn": True,
                "close_penalty": 0.5,
                "main_sight_radius": 20,
                "disappearance_prob": 0.,
                "perturbation_probs": [1., 0., 0.]
            }

            self.env_kwargs_eval = {
                "implicit_max_player_num": 5,
                'max_player_num': 5,
                "num_players": 3,
                "rnn_with_gnn": True,
                "close_penalty": 0.5,
                "main_sight_radius": 20,
                "disappearance_prob": 0.,
                "perturbation_probs": [1., 0., 0.]
            }

        elif "fortattack" in self.env_name:
            self.env_kwargs = {
                "max_timesteps": 100,
                "num_guards": 5,
                "num_attackers": 5,
                "active_agents": 3,
                "num_freeze_steps": 80,
                "reward_mode": "sparse",
                "arguments": get_args(),
                "with_oppo_modelling": True,
                "team_mode": "guard",
                "agent_type": -1,
                "obs_mode": config["obs_mode"],
                "cone_angle": config["cone_angle"],
                "vision_radius": config["vision_radius"]
            }

            self.env_kwargs_eval = {
                "max_timesteps": 100,
                "num_guards": 5,
                "num_attackers": 5,
                "active_agents": 5,
                "num_freeze_steps": 80,
                "reward_mode": "sparse",
                "arguments": get_args(),
                "with_oppo_modelling": True,
                "team_mode": "guard",
                "agent_type": -1,
                "obs_mode": config["obs_mode"],
                "cone_angle": config["cone_angle"],
                "vision_radius": config["vision_radius"]
            }


    def preprocess(self, obs):
        final_obs = None
        if "Foraging" in self.env_name:
            try :
                batch_size = obs["player_info_shuffled"].shape[0]
            except IndexError:
                obs = obs.item(0)
                batch_size = obs["player_info_shuffled"].shape[0]
            player_obs = np.reshape(obs["player_info_shuffled"], (batch_size, self.env_kwargs["players"], -1))
            other_obs = np.repeat(
                np.reshape(obs["food_info"], (batch_size, 1, -1)),
                self.env_kwargs["players"],
                axis=1
            )
            added_data = np.zeros((batch_size, self.env_kwargs["players"], 2))
        elif "Navigation" in self.env_name:
            try :
                batch_size = obs["player_info_shuffled"].shape[0]
            except IndexError:
                obs = obs.item(0)
                batch_size = obs["player_info_shuffled"].shape[0]
            player_obs = np.reshape(obs["player_info_shuffled"], (batch_size, self.env_kwargs["players"], -1))
            other_obs = np.repeat(
                np.reshape(obs["dest_info"], (batch_size, 1, -1)),
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

        elif "fortattack" in self.env_name:
            try :
                batch_size = obs["player_obs"].shape[0]
            except IndexError:
                obs = obs.item(0)
                batch_size = obs["player_obs"].shape[0]
            player_obs = np.reshape(
                obs["player_obs"], (batch_size, self.env_kwargs["num_guards"]+self.env_kwargs["num_attackers"], -1)
            )
            other_obs = np.zeros([batch_size, self.env_kwargs["num_guards"]+self.env_kwargs["num_attackers"], 0])
            added_data = np.zeros((batch_size, self.env_kwargs["num_guards"] + self.env_kwargs["num_attackers"], 1))

        all_obs = np.concatenate([player_obs, other_obs], axis=-1)

        # Add feature to distinguish player from others
        added_data[:, 0, 0] = 1

        # Add feature that denotes agent existence
        if not "fortattack" in self.env_name:
            agent_exists = all_obs[:, :, 0] != -1
            added_data[agent_exists, 1] = 1

        final_obs = np.concatenate([added_data, all_obs], axis=-1)

        return final_obs

    def preprocess_complete(self, obs):
        final_obs = None
        if "Foraging" in self.env_name:
            try :
                batch_size = obs["player_info_shuffled"].shape[0]
            except IndexError:
                obs = obs.item(0)
                batch_size = obs["player_info_shuffled"].shape[0]
            player_obs = np.reshape(obs["complete_player_info_shuffled"], (batch_size, self.env_kwargs["players"], -1))
            other_obs = np.repeat(
                np.reshape(obs["food_info_complete"], (batch_size, 1, -1)),
                self.env_kwargs["players"],
                axis=1
            )
            added_data = np.zeros((batch_size, self.env_kwargs["players"], 2))
        elif "Navigation" in self.env_name:
            try :
                batch_size = obs["complete_player_info_shuffled"].shape[0]
            except IndexError:
                obs = obs.item(0)
                batch_size = obs["complete_player_info_shuffled"].shape[0]
            player_obs = np.reshape(obs["complete_player_info_shuffled"], (batch_size, self.env_kwargs["players"], -1))
            other_obs = np.repeat(
                np.reshape(obs["dest_info_complete"], (batch_size, 1, -1)),
                self.env_kwargs["players"],
                axis=1
            )
            added_data = np.zeros((batch_size, self.env_kwargs["players"], 2))

        elif "wolfpack" in self.env_name:
            try :
                batch_size = obs["teammate_location_shuffled"].shape[0]
            except IndexError:
                obs = obs.item(0)
                batch_size = obs["teammate_location_shuffled"].shape[0]
            player_obs = np.reshape(
                obs["teammate_location_shuffled_complete"], (batch_size, self.env_kwargs["max_player_num"], -1)
            )
            other_obs = np.repeat(
                np.reshape(obs["opponent_info_complete"], (batch_size, 1, -1)),
                self.env_kwargs["max_player_num"],
                axis=1
            )
            added_data = np.zeros((batch_size, self.env_kwargs["max_player_num"], 2))
        elif "fortattack" in self.env_name:
            try :
                batch_size = obs["complete_player_obs"].shape[0]
            except IndexError:
                obs = obs.item(0)
                batch_size = obs["complete_player_obs"].shape[0]
            player_obs = np.reshape(
                obs["complete_player_obs"], (batch_size, self.env_kwargs["num_guards"]+self.env_kwargs["num_attackers"], -1)
            )
            other_obs = np.zeros([batch_size, self.env_kwargs["num_guards"]+self.env_kwargs["num_attackers"], 0])
            added_data = np.zeros((batch_size, self.env_kwargs["num_guards"] + self.env_kwargs["num_attackers"], 1))

        all_obs = np.concatenate([player_obs, other_obs], axis=-1)

        # Add feature to distinguish player from others
        added_data[:, 0, 0] = 1

        # Add feature that denotes agent existence
        if not "fortattack" in self.env_name:
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
                (obs_space["player_info"].shape[-1]//self.env_kwargs["players"])+
                obs_space["food_info"].shape[-1]+
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
        elif "Navigation" in self.env_name:
            obs_sizes = (
                self.env_kwargs["players"],
                (obs_space["player_info"].shape[-1] // self.env_kwargs["players"]) +
                obs_space["dest_info"].shape[-1] +
                agent_existence_offset + agent_learner_offset
            )
            return obs_sizes, \
                   (obs_space["player_info"].shape[-1] // self.env_kwargs["players"]) + \
                   agent_existence_offset + agent_learner_offset
        elif "fortattack" in self.env_name:
            obs_sizes = (
                self.env_kwargs["num_guards"] + self.env_kwargs["num_attackers"],
                obs_space["player_obs"].shape[-1] + agent_learner_offset
            )
            return obs_sizes, obs_space["player_obs"].shape[-1] + agent_learner_offset

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

        if not os.path.exists(self.config["logging_dir"]+"_grad"):
            os.makedirs(self.config["logging_dir"]+"_grad")

        if not os.path.exists(self.config["saving_dir"]):
            os.makedirs(self.config["saving_dir"])

        directory = os.path.join(self.config['saving_dir'], random_experiment_name)

        if not os.path.exists(self.config["logging_dir"]+"/"+random_experiment_name):
            os.makedirs(self.config["logging_dir"]+"/"+random_experiment_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, 'params.json'), 'w') as json_file:
            json.dump(self.config, json_file)

        with open(
                os.path.join(self.config['logging_dir'], random_experiment_name, 'params.json'), 'w'
        ) as json_file:
            json.dump(self.config, json_file)

    def decide_acts(self, q_values, particle_logs, eval=False):

        particle_dist_prob = dist.Categorical(logits=particle_logs.mean(dim=-1).mean(dim=-1)).probs
        q_values = torch.sum(q_values * (particle_dist_prob.unsqueeze(-1).detach()), dim=1)

        acts = torch.argmax(q_values, dim=-1).tolist()
        if not eval:
            acts = [
                a if random.random() > self.epsilon else random.randint(0, q_values.shape[-1] - 1) for a in
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

        env_train = VectorEnv([
            make_env(
                self.config["env_name"], seed=100 * idx + self.config["eval_seed"], env_kwargs=self.env_kwargs
            ) for idx in range(self.config["num_collection_threads"])
        ])

        device = torch.device("cuda" if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
        num_dones = [0] * self.config["num_collection_threads"]
        per_worker_rew = [0.0] * self.config["num_collection_threads"]

        # Initialize initial obs and states for model
        raw_obs = None
        try:
            raw_obs = env_train.reset().item(0)
        except AttributeError:
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
            out = model(current_memory, eval=True)
            action_dist = model.predict_action(
                out["others"]["graph"], out["latent_state"]
            )
            q_vals = cg_model(
                out["others"]["graph_particles"], out["latent_state"],
                torch.ones([
                    current_memory["current_obs"].size()[0], self.config["num_particles"], current_memory["current_obs"].size()[1]
                ]).to(device).double(),
                "inference", action_dist, None
            )

            acts = self.decide_acts(q_vals, out["latent_state"].theta_many_samples_likelihood.detach(), eval=True)
            n_obs_raw, rews, dones, infos = env_train.step(acts)

            per_worker_rew = [k + l for k, l in zip(per_worker_rew, rews)]
            act = None
            if "Foraging" in self.config["env_name"] or "Navigation" in self.config["env_name"]:
                act = np.concatenate(
                    [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["prev_actions_shuffled"]], axis=-1
                )
            elif "wolfpack" in self.config["env_name"]:
                act = np.concatenate(
                    [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["oppo_actions_shuffled"]], axis=-1
                )
            elif "fortattack" in self.config["env_name"]:
                act = np.concatenate(
                    [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["prev_actions"].squeeze(-1)[:, 1:]], axis=-1
                )

            one_hot_acts = self.to_one_hot(act, action_shape)

            # Update observations with next observation
            if not self.config["separate_types"]:
                current_memory["states"] = out["latent_state"].multiply_each(
                    (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                        'theta','cell1', 'cell2', 'theta_var', 'cell1_var', 'cell2_var', 'theta_sample',
                        'theta_many_samples', 'theta_many_samples_likelihood'
                    ]).detach()
            else:
                current_memory["states"] = out["latent_state"].multiply_each(
                    (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                        'theta', 'cell1', 'cell2', 'theta_var', 'cell1_var', 'cell2_var', 'theta_sample',
                        'theta2', 'cell12', 'cell22', 'theta2_var', 'cell12_var', 'cell22_var', 'theta2_sample',
                        'theta_many_samples', 'theta_many_samples_likelihood'
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

        env_eval = VectorEnv([
            make_env(
                self.config["env_name"], seed=100 * idx + self.config["eval_seed"], env_kwargs=self.env_kwargs_eval
            ) for idx in range(self.config["num_collection_threads"])
        ])

        device = torch.device("cuda" if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
        num_dones = [0] * self.config["num_collection_threads"]
        per_worker_rew = [0.0] * self.config["num_collection_threads"]
        avgs = []

        raw_obs = None
        try:
            raw_obs = env_eval.reset().item(0)
        except AttributeError:
            raw_obs = env_eval.reset()

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
            out = model(current_memory, eval=True)
            action_dist = model.predict_action(
                out["others"]["graph"], out["latent_state"]
            )
            q_vals = cg_model(
                out["others"]["graph_particles"], out["latent_state"],
                torch.ones([
                    current_memory["current_obs"].size()[0], self.config["num_particles"], current_memory["current_obs"].size()[1]
                ]).to(device).double(),
                "inference", action_dist, None
            )

            acts = self.decide_acts(q_vals, out["latent_state"].theta_many_samples_likelihood.detach(), eval=True)
            n_obs_raw, rews, dones, infos = env_eval.step(acts)

            per_worker_rew = [k + l for k, l in zip(per_worker_rew, rews)]
            act = None
            if "Foraging" in self.config["env_name"] or "Navigation" in self.config["env_name"]:
                act = np.concatenate(
                    [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["prev_actions_shuffled"]], axis=-1
                )
            elif "wolfpack" in self.config["env_name"]:
                act = np.concatenate(
                    [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["oppo_actions_shuffled"]], axis=-1
                )
            elif "fortattack" in self.config["env_name"]:
                act = np.concatenate(
                    [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["prev_actions"].squeeze(-1)[:, 1:]], axis=-1
                )

            one_hot_acts = self.to_one_hot(act, action_shape)

            # Update observations with next observation
            if not self.config["separate_types"]:
                current_memory["states"] = out["latent_state"].multiply_each(
                    (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                        'theta','cell1', 'cell2', 'theta_var', 'cell1_var', 'cell2_var', 'theta_sample',
                        'theta_many_samples', 'theta_many_samples_likelihood'
                    ]).detach()
            else:
                current_memory["states"] = out["latent_state"].multiply_each(
                    (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                        'theta', 'cell1', 'cell2', 'theta_var', 'cell1_var', 'cell2_var', 'theta_sample',
                        'theta2', 'cell12', 'cell22', 'theta2_var', 'cell12_var', 'cell22_var', 'theta2_sample',
                        'theta_many_samples', 'theta_many_samples_likelihood'
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
        writer2 = SummaryWriter(log_dir=self.config["logging_dir"]+"_grad" + "/" + random_experiment_name)

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

        env = VectorEnv([
            make_env(
                self.config["env_name"], seed=100 * idx, env_kwargs=self.env_kwargs
            ) for idx in range(self.config["num_collection_threads"])
        ])

        num_steps = self.config["target_training_steps"] // self.config["num_collection_threads"]
        num_episodes = num_steps // self.config["eps_length"]

        try:
            test_obses = self.preprocess(env.reset().item(0))
        except AttributeError:
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
        if "Foraging" in self.config["env_name"] or "Navigation" in self.config["env_name"]:
            num_agents = self.env_kwargs["players"]
        elif "wolfpack" in self.config["env_name"]:
            num_agents = self.env_kwargs["max_player_num"]
        elif "fortattack" in self.config["env_name"]:
            num_agents = self.env_kwargs["num_guards"] + self.env_kwargs["num_attackers"]

        # Initialize belief prediction model
        model = GPLTypeInferenceModelStateReconsStochastic(
            action_space=env1.action_space,
            nr_inputs=obs_size,
            agent_inputs=agent_o_size,
            u_inputs=obs_size - agent_o_size,
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
            gnn_decoder_hid_dims=[
                self.config["gnn_decoder_hid_dims1"],
                self.config["gnn_decoder_hid_dims1"],
                self.config["gnn_decoder_hid_dims1"]
            ],
            state_hid_dims=self.config["state_hid_dims"],
            batch_size=self.config["num_collection_threads"],
            num_agents=num_agents,
            device=device,
            num_particles=self.config["num_particles"],
            separate_types=self.config["separate_types"],
            encoder_batchnorm=False,
            with_global_features=False if "fortattack" in self.config["env_name"] else True
        )

        cg_model = ParticledGPLDecisionMakingModel(
            action_space=env1.action_space,
            s_dim=self.config["s_dim"],
            theta_dim=self.config["h_dim"],
            device=device,
            mid_pair=self.config["mid_pair"],
            mid_nodes=self.config["mid_nodes"],
            mid_pair_out=self.config["lrf_rank"]
        )

        target_cg_model = ParticledGPLDecisionMakingModel(
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
                       self.config["saving_dir"] + "/" + random_experiment_name + "/"+ "0-cg.pt")

            torch.save(target_cg_model.state_dict(),
                       self.config["saving_dir"] + "/" + random_experiment_name + "/"+ "0-tar-cg.pt")

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

        updates_per_episode = self.config["eps_length"]//self.config["update_period"]

        start_point = self.config["load_from_checkpoint"]
        if self.config["load_from_checkpoint"] == -1:
            start_point = 0
        total_updates = start_point * self.config["model_saving_frequency"]
        start_ep_id = total_updates//updates_per_episode
        all_num_steps = 0

        # Make updates based on sequential data
        for ep_id in range(start_ep_id, num_episodes):
            # Preprocess sequential data into input for network.
            raw_obs = None
            try:
                raw_obs = env.reset().item(0)
            except AttributeError:
                raw_obs = env.reset()
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
            total_obs_reconstruction_log_prob = 0

            steps_since_log = 0
            self.epsilon = 1.0 - (
                min((ep_id + 0.0) / (self.config["exploration_percentage"] * num_episodes), 1.0)
            ) * 0.95

            # Compute per step computations
            while idx < self.config["eps_length"]:
                all_num_steps += 1
                idx += 1
                steps_since_log += 1
                out = model(current_memory)
                action_dist = model.predict_action(
                    out["others"]["graph"], out["latent_state"].detach()
                )
                q_vals = cg_model(
                    out["others"]["graph_particles"], out["latent_state"],
                    torch.ones([
                        current_memory["current_obs"].size()[0], self.config["num_particles"], current_memory["current_obs"].size()[1]
                    ]).to(device).double(),
                    "inference", action_dist, None
                )

                acts = self.decide_acts(q_vals, out["latent_state"].theta_many_samples_likelihood.detach())
                n_obs_raw, rews, dones, infos = env.step(acts)

                act = None
                if "Foraging" in self.config["env_name"] or "Navigation" in self.config["env_name"]:
                    act = np.concatenate(
                        [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["prev_actions_shuffled"]], axis=-1
                    )
                elif "wolfpack" in self.config["env_name"]:
                    act = np.concatenate(
                        [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["oppo_actions_shuffled"]], axis=-1
                    )
                elif "fortattack" in self.config["env_name"]:
                    act = np.concatenate(
                        [np.reshape(np.asarray(acts), (-1, 1)), n_obs_raw["prev_actions"].squeeze(-1)[:, 1:]], axis=-1
                    )

                one_hot_acts = self.to_one_hot(act, action_shape)
                recently_executed_actions = torch.tensor(one_hot_acts).double().to(device)

                cg_predicted_values = cg_model(
                    out["others"]["graph_particles"], out["latent_state"],
                    torch.ones([
                        current_memory["current_obs"].size()[0], self.config["num_particles"], current_memory["current_obs"].size()[1]
                    ]).to(device).double(),
                    "train", action_dist,
                    recently_executed_actions
                )

                particle_dist_prob = dist.Categorical(
                    logits=out["latent_state"].theta_many_samples_likelihood.mean(dim=-1).mean(dim=-1)
                ).probs
                cg_predicted_values = torch.sum(cg_predicted_values * (particle_dist_prob.unsqueeze(-1).detach()), dim=1)

                predicted_values = cg_predicted_values.view(
                    self.config["num_collection_threads"], -1
                )

                # Add action reconstruction losses
                avg_act_log_prob_loss = out["others"]["action_reconstruction_log_prob"]
                avg_state_log_prob_loss = out["others"]["state_reconstruction_log_prob"]
                avg_agent_existence_log_prob_loss = out["others"]["agent_existence_log_prob"]
                kl_div_term = out["others"]["KL_div_type1"].mean()

                total_loss += self.config["states_reconstruction_weight"] * -avg_state_log_prob_loss
                total_loss += self.config["act_reconstruction_weight"] * -avg_act_log_prob_loss
                total_loss += self.config["kl_div_loss_weight"] * kl_div_term
                #total_loss += self.config["agent_existence_reconstruction_weight"] * -avg_agent_existence_log_prob_loss

                print(
                    "States prediction loss  : ",
                    self.config["states_reconstruction_weight"] * -avg_state_log_prob_loss
                )

                print(
                    "Agent existence prediction loss  : ",
                    self.config["agent_existence_reconstruction_weight"] * -avg_agent_existence_log_prob_loss
                )

                print(
                    "Action prediction loss  : ",
                    self.config["act_reconstruction_weight"] * -avg_act_log_prob_loss
                )

                # Add log probability of action reconstruction to logging
                total_action_prediction_log_prob += avg_act_log_prob_loss
                total_obs_reconstruction_log_prob += avg_state_log_prob_loss

                # Update observation with next observation
                if not self.config["separate_types"]:
                    current_memory["states"] = out["latent_state"].multiply_each(
                        (1-torch.Tensor(dones).double().to(device).view(-1,1)),[
                            'theta', 'cell1', 'cell2', 'theta_var', 'cell1_var', 'cell2_var', 'theta_sample',
                            'theta_many_samples', 'theta_many_samples_likelihood'
                        ])
                else:
                    current_memory["states"] = out["latent_state"].multiply_each(
                        (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                            'theta', 'cell1', 'cell2', 'theta_var', 'cell1_var', 'cell2_var', 'theta_sample',
                            'theta2', 'cell12', 'cell22', 'theta2_var', 'cell12_var', 'cell22_var', 'theta2_sample',
                            'theta_many_samples', 'theta_many_samples_likelihood'
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
                current_memory["actions"] = masked_actions[:,0,:]
                current_memory["rewards"] = reward_tensor
                current_memory["other_actions"] = masked_actions[:, 1:, :]
                current_memory["dones"] = done_tensor
                current_memory["current_state"] = torch.tensor(nob_complete).double().to(device)

                # Compute target values for value loss computation
                out = model(current_memory)
                action_dist = model.predict_action(
                    out["others"]["graph"], out["latent_state"].detach()
                )
                target_q_vals = target_cg_model(
                    out["others"]["graph_particles"], out["latent_state"].detach(),
                    torch.ones([
                        current_memory["current_obs"].size()[0], self.config["num_particles"], current_memory["current_obs"].size()[1]
                    ]).to(device).double(),
                    "inference", action_dist, None
                )

                target_particle_dist_prob = dist.Categorical(
                    logits=out["latent_state"].theta_many_samples_likelihood.mean(dim=-1).mean(dim=-1)
                ).probs
                target_q_vals = torch.sum(target_q_vals * (target_particle_dist_prob.unsqueeze(-1).detach()), dim=1)

                # Process target q values by weighting them based on particle weights
                target_q_vals = target_q_vals.view(
                    self.config["num_collection_threads"], -1
                )

                target_q_values = (target_q_vals.max(dim=-1)[0].unsqueeze(-1))
                rews = reward_tensor
                dons = done_tensor
                all_target_values = rews + self.config["gamma"] * (1 - dons) * target_q_values
                q_loss = ((predicted_values - all_target_values.detach()) ** 2).mean()

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

                    log_dict ={
                        "total_loss": total_loss,
                        "total_q_loss": (total_q_loss+0.0)/steps_since_log,
                        "total_action_prediction_log_prob": (total_action_prediction_log_prob+0.0)/steps_since_log,
                        "total_obs_reconstruction_log_prob": (total_obs_reconstruction_log_prob+0.0)/steps_since_log
                    }
                    self.log_values(writer,log_dict, total_updates)

                    # Reset values
                    # For optimization
                    total_loss = 0
                    total_q_loss = 0

                    # For logging
                    total_action_prediction_log_prob = 0
                    total_obs_reconstruction_log_prob = 0

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
        target_param.data.copy_(tau*param + (1-tau)*target_param)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = vars(args)
    model_trainer = GPLOnlyModelTraining(args)
    model_trainer.run()
