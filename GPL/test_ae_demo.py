import argparse
import gym
import random
import lbforaging
import Wolfpack_gym
import coopnavigation
import FortAttack_gym
from gym.vector import SyncVectorEnv as VectorEnv

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
import os
from RNNTypeModel import GPLTypeInferenceModelStateRecons, GPLDecisionMakingModel
from torch import optim, nn
import torch.distributions as dist
from arguments import get_args
import random
from render import Visualizer
import imageio

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
parser.add_argument('--num-collection-threads', type=int, default=1, help="Number of threads for data collection.")
parser.add_argument('--num-players-train', type=int, default=3, help="Maximum number of players for training.")
parser.add_argument('--num-players-test', type=int, default=5, help="Maximum number of players for testing.")
parser.add_argument('--eps-length', type=int, default=200, help="Maximum episode length for training.")

# Training details
parser.add_argument('--batch-size', type=int, default=16, help="Batch size per updates.")
parser.add_argument('--use-cuda', type=bool, default=True, help="Use CUDA for training or not")
parser.add_argument('--google-cloud', type=str, default="True", help="If multiple GPUs we can choose which one to use")
parser.add_argument('--designated-cuda', type=str, default="cuda:0", help="Designated cuda")
parser.add_argument('--seed', type=int, default=0, help="Training seed.")
parser.add_argument('--eval-init-seed', type=int, default=2500, help="Evaluation seed")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for training.")
parser.add_argument('--update-period', type=int, default=4, help="Time between updates.")
parser.add_argument('--max-grad-norm', type=float, default=10.0, help="Maximum gradient magnitude for update.")
parser.add_argument('--model-saving-frequency', type=int, default=2500, help="Number of steps before logging")
parser.add_argument('--init-epsilon', type=float, default=1.0, help="Initial exploration rate.")
parser.add_argument('--final-epsilon', type=float, default=0.05, help="Final exploration rate.")
parser.add_argument('--exploration-percentage', type=float, default=0.7, help="Percentage of experiments where exploration is done.")
parser.add_argument('--gamma', type=float, default=0.99, help="Discount rate.")
parser.add_argument('--tau', type=float, default=0.001, help="Polyak averaging rate for target network update.")
parser.add_argument('--target-network-update-frequency', type=int, default=100, help="Number of updates before target network is updated.")
parser.add_argument('--load-from-checkpoint', type=int, default=-1, help="Checkpoint to load parameters from.")

# Loss weights
parser.add_argument('--act-reconstruction-weight', type=float, default=0.1, help="Weight associated to action reconstruction loss.")
parser.add_argument('--states-reconstruction-weight', type=float, default=0.01, help="Weight associated to state reconstruction loss.")
parser.add_argument('--agent-existence-reconstruction-weight', type=float, default=0.2, help="Weight associated to agent existence prediction loss.")
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
parser.add_argument('--mid-pair', type=int, default=70, help="Hidden layer sizes for CG's pairwise utility computation.")
parser.add_argument('--mid-nodes', type=int, default=70, help="Hidden layer sizes for CG's singular utility computation.")
parser.add_argument('--lrf-rank', type=int, default=6, help="Rank for the low rank factorization trick in the CG's pairwise utility computation.")
parser.add_argument('--separate-types', type=bool, default=False, help="Whether to use separate types/not.")

# Decision making evaluation parameters
parser.add_argument('--num-eval-episodes', type=int, default=3, help="Number of evaluation episodes")
parser.add_argument('--eval-seed', type=int, default=500, help="Number of evaluation episodes")

# Additional arguments for FAttack
parser.add_argument('--obs-mode', type=str, default="conic", help="Type of observation function for FAttack (either conic/circular).")
parser.add_argument('--cone-angle', type=float, default=math.pi, help="Obs cone angle for FAttack")
parser.add_argument('--vision-radius', type=float, default=2.0, help="Vis radius for FAttack")
# Testing details
parser.add_argument('--render-env', type=bool, default=False, help="Render environment")
parser.add_argument('--plot-graph', type=bool, default=False, help="plot graphs")
parser.add_argument('--do-analysis', type=bool, default=False, help="do analysis of environment")
parser.add_argument('--buffer-size', type=int, default=50000, help="buffer size for decoder net")
parser.add_argument('--collect-data', type=bool, default=False, help="if we are collecting data or not")
parser.add_argument('--decode-existence', type=int, default=1, help="if 2 the state decoder does not reconstruct existence. if 1 it does")
parser.add_argument('--collect_data_n_episodes', type=int, default=50, help="number of episodes to collect data from")
parser.add_argument('--rmse-loss', type=bool, default=False, help="uses rms loss for the decoder network")
parser.add_argument('--wolfpack-constant', type=int, default=8, help="Removes the rotation from the global states")
parser.add_argument('--main-sight-radius', type=int, default=3, help="Sight radious of woflpack")
parser.add_argument('--verbose', type=bool, default=False, help="Print in screen or not")


args = parser.parse_args()

class GPLOnlyModelTraining(object):
    def __init__(self, config):
        self.config = config
        if self.config['google_cloud'] == "True":
            self.device = torch.device(self.config['designated_cuda'] if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
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
                "main_sight_radius": 3,
                "disappearance_prob": 0.,
                "perturbation_probs": [1., 0., 0.]
            }

            self.env_kwargs_eval = {
                "implicit_max_player_num": 5,
                'max_player_num': 5,
                "num_players": 3,
                "rnn_with_gnn": True,
                "close_penalty": 0.5,
                "main_sight_radius": 3,
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

    def decide_acts(self, q_values, eval=False):

        acts = torch.argmax(q_values, dim=-1).tolist()
        if not eval:
            acts = [
                a if random.random() > self.epsilon else random.randint(0, q_values.shape[-1] - 1) for a in
                acts
            ]

        return acts

    def evaluate_demo_episode(self, action_shape, model, cg_model, true_state_directory):
        """
        This function loads an episode and evaluates the inference model over that recorded episode. 

        It also generates the renders for each particle  
    
        """

        # Create env for policy eval
        def make_env(env_name, seed, env_kwargs):
            def _make():
                env = gym.make(
                    env_name, seed=seed, **env_kwargs
                )
                return env
            return _make

        env_test = gym.vector.SyncVectorEnv([
            make_env(
                self.config["env_name"], seed=100 * idx + self.config["eval_seed"], env_kwargs=self.env_kwargs
            ) for idx in range(self.config["num_collection_threads"])
        ])
        device = torch.device("cuda" if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
        num_dones = [0] * self.config["num_collection_threads"]
        per_worker_rew = [0.0] * self.config["num_collection_threads"]
        
        if "Foraging" in self.config["env_name"]: 
            env_name = "LBF"
        if "Navigation" in self.config["env_name"]:
            env_name = "Navigation"
        elif "wolfpack" in self.config["env_name"]:
            env_name = "wolfpack"

        # Get basic info of env
        env1 = gym.make(
            self.config["env_name"], **self.env_kwargs
        )
        obs_sizes, agent_o_size = self.get_obs_sizes(env1.observation_space)
        

        # load episode 
       
        raw_obs_buff  = np.load(true_state_directory + '/raw_obs_buff.npy', allow_pickle=True)
        test_obses_buff  = np.load(true_state_directory +'/test_obses_buff.npy', allow_pickle=True)
        test_states_buff = np.load(true_state_directory + '/test_states_buff.npy', allow_pickle=True)
        dones_buff = np.load(true_state_directory + '/dones_buff.npy', allow_pickle=True)
        rews_buff  = np.load(true_state_directory + '/rews_buff.npy', allow_pickle=True)
        if "Foraging" in self.config["env_name"] or "Navigation" in self.config["env_name"]:
            # print("Will continue without saving images because env is:", self.config["env_name"])
            pass
        elif "wolfpack" in self.config["env_name"] and self.config["render_env"]:
            # TODO I have an error with this when the env 
            visualizer = Visualizer(10,10,main_sight=self.config["main_sight_radius"])
            ## print('no')


        dim_test = len(test_obses_buff)
        n_updates_buff = np.zeros(self.config["num_eval_episodes"])
        n_updates = 0 
        # Buffers for saving data and analysis
        buff_action_reconstruction_log_prob = np.zeros((self.config["num_eval_episodes"], dim_test)) 
        buff_state_reconstruction_log_prob = np.zeros((self.config["num_eval_episodes"], dim_test)) 
        buff_agent_existence_log_prob = np.zeros((self.config["num_eval_episodes"], dim_test)) 
        buff_agent_existence_squarred = np.zeros((self.config["num_eval_episodes"], dim_test)) 

        for j in range(self.config["num_eval_episodes"]):
            
            # print('Episode number', j)
            #init state
            n_obs_raw = raw_obs_buff[0]
            test_obses = test_obses_buff[0]
            test_states = test_states_buff[0]

            dim_test = len(test_obses_buff)

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


            
            for k in range(1,dim_test): 
                # print("Total n eval episodes", self.config["num_eval_episodes"],"Current eval episode", j,"episode step", k)
                # Decide agent's action based on model
                out = model(current_memory, eval=True)

                action_dist = model.predict_action(
                    out["others"]["graph"], out["latent_state"]
                )
                q_vals = cg_model(
                    out["others"]["graph"], out["latent_state"], (out["latent_state"].i[:,:,0]).double(), "inference", action_dist, None
                )
                
                
                n_updates += 1
                

                acts = self.decide_acts(q_vals, eval=True)

                # Demo environment loaded so I do not need to do a step             
                # n_obs_raw, rews, dones, infos = env_test.step(acts)
                # I only need to load it 
                dones = dones_buff[k]
                rews = rews_buff[k]
                n_obs_raw = raw_obs_buff[k]
                nob = test_obses_buff[k]
                nob_complete = test_states_buff[k]
                
                ########################### Image reconstruction
                # Get obs vectors from current vector
                state_dist, u_dist = model.decoder_network(out["latent_state"].detach())
                

                

                # true existence 
                existence_state = current_memory["current_state"][:,1:5,1:2]
                predicted_existence = out["latent_state"].i[:,1:5]
                existence_difference = (predicted_existence.flatten().sum(-1) - existence_state.flatten().sum(-1))**2
                
                # Get losses and log them 
                # state recons
                buff_state_reconstruction_log_prob[num_dones, n_updates]  = out["others"]["state_reconstruction_log_prob"].detach().numpy()
                # existence squared 
                buff_agent_existence_squarred[num_dones, n_updates] = existence_difference.detach().numpy()
                # existence log prob
                # not calculated during inference
                # buff_agent_existence_log_prob = out["others"]["agent_existence_log_prob"].detach().numpy()

                if type(out["others"]["action_reconstruction_log_prob"])==float:
                    buff_action_reconstruction_log_prob[num_dones, n_updates] = out["others"]["action_reconstruction_log_prob"]
                else:
                    buff_action_reconstruction_log_prob[num_dones, n_updates] = out["others"]["action_reconstruction_log_prob"].detach().numpy()
                # print()

                # print(out["others"]["action_reconstruction_log_prob"])
                # print(out["others"]["state_reconstruction_log_prob"].detach().numpy(), existence_difference.detach().numpy())






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

                one_hot_acts = self.to_one_hot(act, action_shape)

                # Update observations with next observation
                current_memory["states"] = out["latent_state"].multiply_each(
                        (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                            'all_actions', 'all_encoded_actions', 'i', 's', 'theta', 'log_weight', 'cell1', 'cell2'
                        ]).detach()

                masked_actions = torch.tensor(one_hot_acts).double().to(device) * (
                            1 - torch.Tensor(dones).double().to(device).unsqueeze(-1).unsqueeze(-1).repeat(
                        1, one_hot_acts.shape[1], one_hot_acts.shape[2]
                    )).detach()


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
                        
                        if self.config["do_analysis"]:
                            self.do_analysis()
                        if self.config["plot_graph"]:
                            self.plot()
                        if num_dones[idx] < self.config['num_eval_episodes']:
                            n_updates_buff[num_dones] = n_updates
                            n_updates = 0
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

        # Save buffers 
        np.save(self.config["saving_dir"] + "/" + self.config["exp_name"] + 
            '/buff_action_reconstruction_log_prob_' + str(self.config["load_from_checkpoint"])+ '.npy',buff_action_reconstruction_log_prob)
        np.save(self.config["saving_dir"] + "/" + self.config["exp_name"] + 
            '/buff_state_reconstruction_log_prob_' + str(self.config["load_from_checkpoint"])+ '.npy',buff_state_reconstruction_log_prob)
        np.save(self.config["saving_dir"] + "/" + self.config["exp_name"] + 
            '/buff_agent_existence_log_prob_' + str(self.config["load_from_checkpoint"])+ '.npy',buff_agent_existence_log_prob)
        np.save(self.config["saving_dir"] + "/" + self.config["exp_name"] + 
            '/buff_agent_existence_squarred_' + str(self.config["load_from_checkpoint"])+ '.npy',buff_agent_existence_squarred)





        return 

    def create_demo_episode(self, action_shape, model, cg_model, directory):
        """
        This function takes the policy and runs it over one single episode. This episode is stored for doing the state reconstruction analysis. 
        Aditionally, a folder with the jpgs of the episode is generated which then can be used for reconstruction analysis. 

        TODO: The main issue with this function is that only generates 1 single episode. 
        This is fine for state reconstruction, but not if you want to perform multiple analysis it would be nice to store multiple episodes like this.  
        """

        # Create env for policy eval
        def make_env(env_name, seed, env_kwargs):
            def _make():
                env = gym.make(
                    env_name, seed=seed, **env_kwargs
                )
                return env
            return _make

        env_test = gym.vector.SyncVectorEnv([
            make_env(
                self.config["env_name"], seed=100 * idx + self.config["eval_seed"], env_kwargs=self.env_kwargs
            ) for idx in range(self.config["num_collection_threads"])
        ])
        device = torch.device("cuda" if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
        num_dones = [0] * self.config["num_collection_threads"]
        per_worker_rew = [0.0] * self.config["num_collection_threads"]
        
        if "Foraging" in self.config["env_name"]: 
            env_name = "LBF"
        if "Navigation" in self.config["env_name"]:
            env_name = "Navigation"
        elif "wolfpack" in self.config["env_name"]:
            env_name = "wolfpack"
            visualizer = Visualizer(10,10,main_sight=self.config["main_sight_radius"])
        
        # Get basic info of env
        env1 = gym.make(
            self.config["env_name"], **self.env_kwargs
        )
        obs_sizes, agent_o_size = self.get_obs_sizes(env1.observation_space)

        # Initialize initial obs and states for model
        raw_obs_buff = []
        test_obses_buff = []
        test_states_buff = []
        dones_buff = []
        rews_buff = []

        try: 
            raw_obs = env_test.reset().item(0)
        except AttributeError:
            # there is a weird bug in gym in which in some cases it returns the orderder dictionary
            raw_obs = env_test.reset()
        # print(self.config["render_env"])
        if self.config["render_env"]:
            env_test.envs[0].render()
        test_obses = self.preprocess(raw_obs)
        test_states = self.preprocess_complete(raw_obs)
        nob_complete = test_states
        raw_obs_buff.append(raw_obs)
        test_obses_buff.append(test_obses)
        test_states_buff.append(test_states)
        dones_buff.append(False)
        rews_buff.append(0)


        initial_states = model.new_latent_state()
        avgs = []

        batch_size, agent_nums = test_obses.shape[0], test_obses.shape[1]
        # print('batch_size, agent_nums', batch_size, agent_nums)
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
        num_demo_env = 1
        # print('obs_sizes', obs_sizes, agent_o_size)
        n_updates_buff = np.zeros(num_demo_env) # self.config["num_eval_episodes"]
        n_updates = 0 
        # print("creating demo episode")

        while (any([k < num_demo_env for k in num_dones])):


            if "wolfpack" in self.config["env_name"]:
                # render if it is wolfpack only
                visualizer.render_true_state(nob_complete[:,:,2:4], # single estate estimation
                                        nob_complete[:,:,4:8], # single u estimation
                                        nob_complete[:,:,1], # true existence ,
                                        n_updates, 
                                        directory,
                                        particle_num_str='true_state',
                                        verbose=self.config["verbose"])
           
            # Decide agent's action based on model
            out = model(current_memory, with_resample=True, eval=True)
            action_dist = model.predict_action(
                out["others"]["graph"], out["latent_state"]
            )
            q_vals = cg_model(
                out["others"]["graph"], out["latent_state"], "inference", action_dist, None
            )
            n_updates += 1
            
            old_logits = -out["encoding_losses"]
            acts = self.decide_acts(q_vals, old_logits, eval=True)
            n_obs_raw, rews, dones, infos = env_test.step(acts)
            




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

            one_hot_acts = self.to_one_hot(act, action_shape)

            # Update observations with next observation
            current_memory["states"] = out["latent_state"].multiply_each(
                    (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                        'all_actions', 'all_encoded_actions', 'i', 's', 'theta', 'log_weight', 'cell1', 'cell2'
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
            
            
            raw_obs_buff.append(n_obs_raw)
            test_obses_buff.append(nob)
            test_states_buff.append(nob_complete)
            dones_buff.append(dones)
            rews_buff.append(rews)

            current_memory["current_obs"] = torch.tensor(nob).double().to(device)
            current_memory["actions"] = masked_actions[:, 0, :]
            current_memory["rewards"] = reward_tensor
            current_memory["other_actions"] = masked_actions[:, 1:, :]
            current_memory["dones"] = done_tensor
            current_memory["current_state"] = torch.tensor(nob_complete).double().to(device)

            for idx, flag in enumerate(dones):
                if flag:
                    
                    if self.config["do_analysis"]:
                        self.do_analysis()
                    if self.config["plot_graph"]:
                        self.plot()
                    if num_dones[idx] < self.config['num_eval_episodes']:
                        n_updates_buff[num_dones] = n_updates
                        n_updates = 0
                        num_dones[idx] += 1
                        avgs.append(per_worker_rew[idx])
                    per_worker_rew[idx] = 0

        # print('len of buffer', len(raw_obs_buff))
        input()

        np.save(directory + '/raw_obs_buff.npy',raw_obs_buff)
        np.save(directory + '/test_obses_buff.npy',test_obses_buff)
        np.save(directory + '/test_states_buff.npy',test_states_buff)
        np.save(directory + '/rews_buff.npy',rews_buff)
        np.save(directory + '/dones_buff.npy',dones_buff)
        # print("Finished creating demo episode")

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

        if self.config['google_cloud'] == "True":
            device = torch.device(self.config['designated_cuda'] if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
        else:
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
                out["others"]["graph"], out["latent_state"],
                (out["latent_state"].i[:,:,0]).double(),
                "inference", action_dist, None
            )

            acts = self.decide_acts(q_vals, eval=True)
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
                        'theta','cell1', 'cell2', 'i'
                    ]).detach()
            else:
                current_memory["states"] = out["latent_state"].multiply_each(
                    (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                        'theta', 'cell1', 'cell2', 'theta2', 'cell12', 'cell22', 'i'
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
        # print("Finished train with rewards " + str(avg_total_rewards))
        env_train.close()
        logger.add_scalar('Rewards/train_set', sum(avgs) / len(avgs), logging_id)

        env_eval = VectorEnv([
            make_env(
                self.config["env_name"], seed=100 * idx + self.config["eval_seed"], env_kwargs=self.env_kwargs_eval
            ) for idx in range(self.config["num_collection_threads"])
        ])

        if self.config['google_cloud'] == "True":
            device = torch.device(self.config['designated_cuda'] if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
        else:
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
                out["others"]["graph"], out["latent_state"],
                (out["latent_state"].i[:,:,0]).double(),
                "inference", action_dist, None
            )

            acts = self.decide_acts(q_vals, eval=True)
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
                        'theta', 'cell1', 'cell2', 'i'
                    ]).detach()
            else:
                current_memory["states"] = out["latent_state"].multiply_each(
                    (1 - torch.Tensor(dones).double().to(device).view(-1, 1)), [
                        'theta', 'cell1', 'cell2',  'theta2', 'cell12', 'cell22', 'i'
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
        # print("Finished eval with rewards " + str(avg_total_rewards))
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

        if self.config['google_cloud'] == "True":
            device = torch.device(self.config['designated_cuda'] if self.config['use_cuda'] and torch.cuda.is_available() else "cpu")
        else:
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
        model = GPLTypeInferenceModelStateRecons(
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
            separate_types=self.config["separate_types"],
            encoder_batchnorm=False,
            with_global_features=False if "fortattack" in self.config["env_name"] else True
        )

        cg_model = GPLDecisionMakingModel(
            action_space=env1.action_space,
            s_dim=self.config["s_dim"],
            theta_dim=self.config["h_dim"],
            device=device,
            mid_pair=self.config["mid_pair"],
            mid_nodes=self.config["mid_nodes"],
            mid_pair_out=self.config["lrf_rank"]
        )

        target_cg_model = GPLDecisionMakingModel(
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
                ) + ".pt", map_location=self.device)
            )
            cg_model.load_state_dict(
                torch.load(self.config["saving_dir"] + "/" + random_experiment_name + "/" + str(
                    self.config["load_from_checkpoint"]
                ) + "-cg.pt", map_location=self.device)
            )
            target_cg_model.load_state_dict(
                torch.load(self.config["saving_dir"] + "/" + random_experiment_name + "/" + str(
                    self.config["load_from_checkpoint"]
                ) + "-tar-cg.pt", map_location=self.device)
            )

            model, cg_model, target_cg_model = model.double(), cg_model.double(), target_cg_model.double()
            model_optimizer = optim.Adam(list(model.parameters()) + list(cg_model.parameters()), lr=self.config["lr"])

            model_optimizer.load_state_dict(
                torch.load(self.config["saving_dir"] + "/" + random_experiment_name + "/" + str(
                    self.config["load_from_checkpoint"]
                ) + "-optim.pt", map_location=self.device)
            )

                        
        import os
        if "wolfpack" in self.config["env_name"]:
            env_name = "wolfpack"
        elif "Foraging" in self.config["env_name"]:
            env_name = "lbf" 
        elif "Navigation" in self.config["env_name"]:
            env_name = "coop"
        else:
            # print('xxxxxxxx')
            raise NotImplementedError


        # check if directory of demo episode exists
        directory = 'episodes/'+ env_name + '/true_state'
        # print(directory)

        # Try to load the demo episode
        try: 
            raw_obs_buff  = np.load(directory + '/raw_obs_buff.npy', allow_pickle=True)
        except FileNotFoundError:
            # if not create directory
            print("Demo episode not found. Please run an episode generator first ")
            #os.makedirs(directory)
            # and call function that creates demo episode
            # this part of the code is like this because you only want to have 1 demo episode 
            # to evaluate everything in there
            #self.create_demo_episode(action_shape, model, cg_model, directory)
            #exit()


        # print("evaluating episode")
        # Then evaluate policy in the demo episode
        self.evaluate_demo_episode(action_shape, model, cg_model, directory)

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
