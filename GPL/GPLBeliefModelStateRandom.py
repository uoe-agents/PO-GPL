import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import aesmc.random_variable as rv
import aesmc.state as st
import aesmc.util as ae_util
import aesmc.statistics as stats
import aesmc.math as math
import aesmc.test_utils as tu
from aesmc.inference import sample_ancestral_index, sample_ancestral_index_ratio
import encoder_decoder
import numpy as np
from BeliefModel import BeliefModelPriv
from Network import GNNBlock, RFMBlock, Encoder, Decoder, JointUtilLayer
from functools import reduce
import torch.nn.functional as F
import torch.distributions as dist
import dgl
from utils import binary_concrete as BinaryConcrete


# Resample will only happen if weight variance is over some threshold
RESAMPLE_THRESHOLD = 0.5
class PF_State():
    """
        A class that contains information about the states of particles in a set of particles for particle filtering.
    """
    def __init__(self, particle_state, particle_log_weights):
        """
            Constructor for PF_State.
                Args:
                    particle_state : State of particles used for particle filtering
                    particle_log_weights : Log weights of particles used for particle filtering.

        """
        self.particle_state = particle_state
        self.particle_log_weights = particle_log_weights

    def detach(self):
        """
            Method to detach (stopping gradient flow) of the particle states and particle log weights.
        """
        return PF_State(
            self.particle_state.detach(),
            self.particle_log_weights.detach())

    def cuda(self):
        """
            Method to move particle states & weights to cuda if using GPU.
        """
        return PF_State(
            self.particle_state.cuda(),
            self.particle_log_weights.cuda())

class GPLBeliefModelStateRandom(BeliefModelPriv):
    def __init__(self,
                 action_space,
                 nr_inputs,
                 agent_inputs,
                 u_inputs,
                 action_encoding,
                 cnn_channels,
                 s_dim,
                 theta_dim,
                 gnn_act_pred_dims,
                 gnn_state_hid_dims,
                 gnn_type_update_hid_dims,
                 gnn_decoder_hid_dims,
                 state_hid_dims,
                 state_message_dims,
                 encoder_batch_norm,
                 batch_size,
                 num_particles,
                 num_agents,
                 resample,
                 device,
                 agent_existence_temperature=1.0,
                 with_global_features = True
                 ):
        super().__init__(action_space, encoding_dimension=theta_dim)
        #self.init_function = init_function
        self.num_particles = num_particles
        self.batch_size = batch_size
        self.encoder_batch_norm = encoder_batch_norm
        self.s_dim = s_dim
        self.theta_dim = theta_dim
        self.resample = resample
        self.device = device
        self.num_agents = num_agents
        self.action_type = None
        self.action_shape = 0
        self.agent_inputs = agent_inputs
        self.u_inputs = u_inputs
        self.agent_existence_temperature=agent_existence_temperature
        self.resample_ratio_treshold = RESAMPLE_THRESHOLD
        # All encoder/decoders are defined in the encoder_decoder.py file
        def mul(num1, num2):
            return num1 * num2

        self.output_dimension = [cnn_channels[-1]]
        self.cnn_output_number = reduce(mul, self.output_dimension, 1)
        # Naming conventions
        phi_x_dim = self.cnn_output_number

        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
            self.action_type = "discrete"
        else:
            action_shape = action_space.shape[0]
            self.action_type = "continuous"

        self.action_shape = action_shape

        ## Create all relevant networks

        # Encodes learner's actions and observations of all agent into a latent state
        self.encoding_network = EncoderNet(
            nr_actions=action_shape,
            action_encoding=action_encoding,
            nr_inputs=nr_inputs,
            layer_dims=cnn_channels,
            encoder_batch_norm=encoder_batch_norm
        ).to(self.device)

        # Removed for current assumptions
        # Combines all agents' observation and learner's action into a single vector
        # self.obs_act_encoder_net = ObservationLearnerActionEncoderNet(
        #     phi_x_dim, action_encoding,
        #     o_gnn_dims[0], o_gnn_dims[1], o_gnn_dims[2],
        #     o_hid_dims[0], o_hid_dims[1], o_hid_dims[2]
        # )

        # Add network that proposes teammate actions that are taken at previous timestep
        self.proposal_act_predictor_net = ProposalActionPredictionNet(
            phi_x_dim, action_encoding, s_dim, theta_dim,
            gnn_act_pred_dims[0], gnn_act_pred_dims[1], gnn_act_pred_dims[2], gnn_act_pred_dims[3],
            action_shape, self.action_type, device=self.device
        ).to(self.device)

        # Add network that predicts the likelihood of sampled actions from proposal_act_predictor_net
        self.act_predictor_net = ActionPredictionNet(
            s_dim, theta_dim,
            gnn_act_pred_dims[0], gnn_act_pred_dims[1], gnn_act_pred_dims[2], gnn_act_pred_dims[3],
            action_shape, self.action_type, device=self.device
        ).to(self.device)

        # Add network that proposes state of the environment (teammate state, s_{t}, and teammate existence, i_{t})
        self.proposal_state_prediction_network = ProposalStatePredictionNetwork(
            phi_x_dim, action_encoding, s_dim, theta_dim,
            gnn_state_hid_dims[0], gnn_state_hid_dims[1], gnn_state_hid_dims[2], gnn_state_hid_dims[1],
            state_hid_dims, state_message_dims, device=self.device
        ).to(self.device)

        # Add network that predicts the likelihood of the sampled state of the environment
        # from proposal_state_prediction_network

        self.state_prediction_network = StatePredictionNetwork(
            action_encoding, s_dim, theta_dim,
            gnn_state_hid_dims[0], gnn_state_hid_dims[1], gnn_state_hid_dims[2], gnn_state_hid_dims[1],
            state_hid_dims, state_message_dims, device=self.device
        ).to(self.device)

        # Add network that updates the type of the agents in the environment
        self.type_updating_network = TypeUpdateNet(
            phi_x_dim, action_encoding, s_dim, theta_dim
        ).to(self.device)

        # Add network that decodes the observation given the information inside the particles
        self.decoder_network = ObsDecoderNet(
            action_encoding, s_dim, theta_dim,
            gnn_decoder_hid_dims[0],
            gnn_decoder_hid_dims[1],
            gnn_decoder_hid_dims[2],
            cnn_channels, agent_inputs-1,
            u_inputs, device=self.device,
            with_global_feature=with_global_features
        ).to(self.device)

        self.state_decoder_network = StateDecoderNet(
            action_encoding, s_dim, theta_dim,
            gnn_decoder_hid_dims[0],
            gnn_decoder_hid_dims[1],
            gnn_decoder_hid_dims[2],
            cnn_channels, agent_inputs - 1,
            u_inputs, device=self.device,
            with_global_feature=with_global_features
        ).to(self.device)

        self.with_global_feature = with_global_features

    def create_graph_structure(self, obs):
        """
        This function creates a graph structure that will be used for the GNNs for particle updates.
        """
        obs_size = obs.all_phi_x.size()
        num_graphs = obs_size[1] * obs_size[2]
        num_agents = obs_size[3]

        all_graphs = []
        for _ in range(num_graphs):
            graph = dgl.DGLGraph()
            graph.add_nodes(num_agents)
            src, dst = zip(*[(i, j) for i in range(num_agents) for j in range(num_agents) if i != j])
            graph.add_edges(src, dst)
            all_graphs.append(graph)

        batched_graphs = dgl.batch(all_graphs)
        batched_graphs.to(self.device)
        return batched_graphs

    def new_latent_state(self):
        """
            This is a function to create new latent states for particles.
            For each particle, we will create the following tensors :
                theta : The theta component (type vector) of the particle following the write up.
                s : The s component (state representation) of the particle following the write up.
                i : The flag indicating the existence of the agent in the particle.
                a : The teammate action component in the particle.
        """

        device = self.device
        mean = 0.
        var = 0.5
        # Initialize
        initial_state = st.State(
                theta=torch.normal(mean,var,
                    size=(self.batch_size, self.num_particles, self.num_agents, self.theta_dim)
                ).to(device).double()
        )

        log_random = torch.rand(size=(self.batch_size, self.num_particles)).to(device).double()
        log_weight_sum = log_random.sum(dim=-1).unsqueeze(dim=-1)
        initial_state.log_weight = log_random/log_weight_sum
        initial_state.s = torch.normal(mean,var,
            size=(self.batch_size, self.num_particles, self.num_agents, self.s_dim)
        ).to(device).double()

        # Initialize the state as if no agents exist beforehand
        initial_state.i = torch.randint(0 , 2 , size=(self.batch_size, self.num_particles, self.num_agents, 1)).to(device).double()

        initial_state.cell1 = torch.normal(mean,var,
             size=(self.batch_size, self.num_particles, self.num_agents, self.theta_dim)
        ).to(device).double()

        initial_state.cell2 = torch.normal(mean,var,
            size=(self.batch_size, self.num_particles, self.num_agents, self.theta_dim)
        ).to(device).double()

        return initial_state

    def vec_conditional_new_latent_state(self, latent_state, mask):
        """
        Set latent state to 0-tensors when new episode begins.
        Args:
            latent_state (`State`): latent_state
            mask: binary tensor with 0 whenever a new episode begins.

        """
        # Multiply log_weight, h, z with mask
        return latent_state.multiply_each(mask, only=['log_weight', 'h', 'z'])

    def sample_from(self, state_random_variable):
        """
        Helper function, legacy code.
        """
        return state_random_variable.sample_reparameterized(
            self.batch_size, self.num_particles
        )

    def propagate(self, observation, reward, actions, previous_latent_state, other_actions, state, state_agent_existence, with_resample=True, eval=False, with_noise=False):
        """
        This is where the core of the particle update is happening.

        Args:
            observation, reward: Last observation and reward recieved from all n_e environments
            actions: Action vector (oneHot for discrete actions)
            previous_latent_state: previous latent state of type state.State

        Returns:
            latent_state: New latent state
            - encoding_logli = encoding_loss: Reconstruction loss when prediction current observation X obs_loss_coef
            - transition_logpdf + proposal_logpdf: KL divergence loss
            - emission_logpdf: Reconstruction loss
            avg_num_killed_particles: Average numer of killed particles in particle filter
            others : other tensors for logging purposes

        """
        batch_size, *rest = observation.size()

        # Needed for legacy AESMC code
        ae_util.init(observation.is_cuda)

        # Legacy code: We need to pass in a (time) sequence of observations
        # With dim=0 for time
        img_observation = observation.unsqueeze(0)
        img_state = state.unsqueeze(0)
        actions = actions.unsqueeze(0)
        if not eval:
            state_existence = state_agent_existence.unsqueeze(0)
        reward = reward.unsqueeze(0)

        # Legacy code: All values are wrapped in state.State (which can contain more than one value)
        observation_states = st.State(
            all_x=img_observation.contiguous(),
            all_x_state=img_state.contiguous(),
            all_a=actions.contiguous(),
        )

        if not eval:
            observation_states.all_state_existence = state_existence.contiguous()

        old_log_weight = previous_latent_state.log_weight

        # Create graph structures for latent variable proposal & transition

        # Encode the actions and observations (nothing recurrent yet)
        observation_states = self.encoding_network(observation_states)
        observation_states.unsequeeze_and_expand_all_(dim=2, size=self.num_particles)
        graph_batch = self.create_graph_structure(observation_states)

        # Removed : Not used under current assumption
        # Get single vectors describing obs related to all agents & the learner's action
        # observation_states.processed_obs = self.obs_act_encoder_net(
        #     obs_graph_batch, current_observation, previous_latent_state, message_passing_steps=1
        # ).unsqueeze(0)
       
        if with_noise:
            old_prob = dist.Categorical(logits=old_log_weight).probs
            tradeoff = 0.6
            old_probs_noisy = tradeoff * old_prob + (1.0 - tradeoff) * torch.ones_like(old_prob) / self.num_particles
            old_log_weight_noisy = dist.Categorical(old_probs_noisy).logits
            ancestral_indices = sample_ancestral_index(old_log_weight_noisy)
        else: 
            # OLD WAY: resample with systematic resample
            ancestral_indices = sample_ancestral_index(old_log_weight)

        # Count number of killed particles per batch entry for logging purposes
        num_killed_particles = list(tu.num_killed_particles(ancestral_indices.data.cpu()))
        if with_resample:
            previous_latent_state = previous_latent_state.resample(ancestral_indices)
        else:
            num_killed_particles = [0] * batch_size

        avg_num_killed_particles = sum(num_killed_particles) / len(num_killed_particles)

        # Get 0th element (first element of time sequence) as input
        current_observation = observation_states.index_elements(0)

        # Sample teammate actions based on action proposal network
        action_parameters = self.proposal_act_predictor_net(
            current_observation, previous_latent_state
        )

        # Postprocess action parameters to get teammate actions at prev timestep
        teammate_action_distribution = None
        teammate_action_sample = None
        active_agent_samples = previous_latent_state.i[:, :, 1:, :]
        teammate_is_active = (active_agent_samples[:, :, :, 0] == 1)

        if self.action_type == "continuous":
            action_distribution = dist.Normal(action_parameters[0], action_parameters[1])
            teammate_action_sample = action_distribution.rsample()

        else:
            action_distribution = dist.OneHotCategorical(logits=action_parameters)
            teammate_action_sample = action_distribution.sample()

        # Make such that non-existent agents will have a default previous action of all zeros.
        existing_teammate_flags = previous_latent_state.i
        teammate_action_samples = teammate_action_sample * existing_teammate_flags
        teammate_action_samples = teammate_action_samples[:,:,1:,:]

        # Use observed learner's action as action for learner's node in particle
        current_batch_size = teammate_action_sample.size()[0]
        learners_action = current_observation.all_a.unsqueeze(dim=-2)
        all_actions = torch.cat([learners_action, teammate_action_samples], dim=-2)

        # Get the likelihood of sampled teammate actions based on action prediction network
        predictor_action_parameters = self.act_predictor_net(
            graph_batch, previous_latent_state
        )

        active_agent_actions = teammate_action_samples[teammate_is_active]
        predicted_action_distribution = None
        all_agent_distribution = None

        if not active_agent_actions.nelement() == 0:
            if self.action_type == "continuous":
                predicted_action_distribution = dist.Normal(
                    predictor_action_parameters[0][:,:,1:,:][teammate_is_active],
                    predictor_action_parameters[1][:,:,1:,:][teammate_is_active]
                )
                all_agent_distribution = dist.Normal(
                    predictor_action_parameters[0][:, :, 1:, :],
                    predictor_action_parameters[1][:, :, 1:, :]
                )
            else:
                predicted_action_distribution = dist.OneHotCategorical(
                    logits=predictor_action_parameters[:,:,1:,:][teammate_is_active]
                )
                all_agent_distribution = dist.OneHotCategorical(
                    logits=predictor_action_parameters[:,:,1:,:]
                )

        # Postprocess agent action log likelihood for existing agents by combining it with
        # log likelihood of actions from non-existing agents (set to zeros by default)

        eval_action_distribution = None
        if not active_agent_actions.nelement() == 0:
            if self.action_type == "continuous":
                eval_action_distribution = dist.Normal(
                    action_parameters[0][:,:,1:,:][teammate_is_active],
                    action_parameters[1][:,:,1:,:][teammate_is_active]
                )

            else:
                eval_action_distribution = dist.OneHotCategorical(
                    logits=action_parameters[:,:,1:,:][teammate_is_active]
                )

        # Initialize default actions of zeros for non-existent agents
        if self.action_type == "continuous":
            all_agents_proposed_action_log_prob = torch.zeros(
                current_batch_size, self.num_particles, self.num_agents-1, self.action_shape
            ).double().to(self.device)

            all_agents_predicted_action_log_prob = torch.zeros(
                current_batch_size, self.num_particles, self.num_agents-1, self.action_shape
            ).double().to(self.device)

            all_agents_proposed_action_entropy = torch.zeros(
                current_batch_size, self.num_particles, self.num_agents-1, self.action_shape
            ).double().to(self.device)

        else:
            all_agents_proposed_action_log_prob = torch.zeros(
                current_batch_size, self.num_particles, self.num_agents-1, 1
            ).double().to(self.device)

            all_agents_predicted_action_log_prob = torch.zeros(
                current_batch_size, self.num_particles, self.num_agents-1, 1
            ).double().to(self.device)

            all_agents_proposed_action_entropy = torch.zeros(
                current_batch_size, self.num_particles, self.num_agents-1, 1
            ).double().to(self.device)


        # Codes to compute action reconstruction loss for all agents

        all_seen_actions = other_actions.unsqueeze(1).repeat(1, self.num_particles, 1, 1)
        valid_seen_actions = (all_seen_actions.sum(dim=-1) != 0)
        all_seen_valid_actions = all_seen_actions[valid_seen_actions]

        seen_action_distribution = None
        if not all_seen_valid_actions.nelement() == 0:
            if self.action_type == "continuous":
                seen_action_distribution = dist.Normal(
                    predictor_action_parameters[0][:, :, 1:, :][valid_seen_actions],
                    predictor_action_parameters[1][:, :, 1:, :][valid_seen_actions]
                )
            else:
                seen_action_distribution = dist.OneHotCategorical(
                    logits=predictor_action_parameters[:, :, 1:, :][valid_seen_actions]
                )

        # Safeguard against empty tensors.
        if not active_agent_actions.nelement() == 0:
            proposed_agent_log_probs = eval_action_distribution.log_prob(active_agent_actions)
            existing_agent_log_probs = predicted_action_distribution.log_prob(active_agent_actions)
            proposed_agent_dist_entropy = eval_action_distribution.entropy()


            if self.action_type == "discrete":
                proposed_agent_log_probs = proposed_agent_log_probs.unsqueeze(-1)
                existing_agent_log_probs = existing_agent_log_probs.unsqueeze(-1)
                proposed_agent_dist_entropy = proposed_agent_dist_entropy.unsqueeze(-1)

            all_agents_proposed_action_log_prob[teammate_is_active] = proposed_agent_log_probs
            all_agents_predicted_action_log_prob[teammate_is_active] = existing_agent_log_probs
            all_agents_proposed_action_entropy[teammate_is_active] = proposed_agent_dist_entropy

        # Safeguard against empty tensors.
        action_reconstruction_log_prob = 0.0
        if not all_seen_valid_actions.nelement() == 0:
            seen_action_log_probs = seen_action_distribution.log_prob(all_seen_valid_actions)
            if self.action_type == "discrete":
                seen_action_log_probs = seen_action_log_probs.unsqueeze(-1)
            action_reconstruction_log_prob = seen_action_log_probs.mean()

        # Create new particles for next timestep
        current_latent_state = st.State()
        setattr(current_latent_state, "all_actions", all_actions)

        # Store encodings of predicted actions in particle
        new_action_encoding = self.encoding_network(all_actions, action_only=True)
        setattr(current_latent_state, "all_encoded_actions", new_action_encoding)

        # Get updated agent existence flag (i_{t+1}) and state representation (s_{t+1})
        updated_agent_flags, proposed_agent_log_existence, agent_existence_logits, proposed_agent_existence_entropy, s_rep_sample, s_rep_dist, agent_state_existence_logs = \
            self.proposal_state_prediction_network(
                current_observation, previous_latent_state, current_latent_state, temperature=self.agent_existence_temperature, eval=eval
            )

        # Set the sampled i and s from the network as i and s in the new particle
        setattr(current_latent_state, "i", updated_agent_flags)
        setattr(current_latent_state, "s", s_rep_sample)

        agent_log_existence, s_distrib_params = self.state_prediction_network(
            previous_latent_state, current_latent_state, temperature=self.agent_existence_temperature
        )

        # Filter out agents that exist
        next_step_existing_agents = (current_latent_state.i[:,:,:,0] == 1)

        proposed_agent_state_log_probs = s_rep_dist.log_prob(s_rep_sample)

        existing_agent_mean = s_distrib_params[0][next_step_existing_agents]
        existing_agent_var = s_distrib_params[1][next_step_existing_agents]

        existing_agents_s_dist = dist.Normal(existing_agent_mean, existing_agent_var)
        existing_agents_s_sample = current_latent_state.s[next_step_existing_agents]

        # Initialize state logprob tensors for both proposal and transition network
        proposed_state_log_prob = torch.zeros_like(s_distrib_params[0]).to(self.device).double()
        state_logprob = torch.zeros_like(s_distrib_params[0]).to(self.device).double()

        # Compute log_prob for both proposed state and transition network
        proposed_state_log_prob[next_step_existing_agents] = proposed_agent_state_log_probs[next_step_existing_agents]
        state_logprob[next_step_existing_agents] = existing_agents_s_dist.log_prob(existing_agents_s_sample)

        # Compute updated type vectors for the agents in the particles
        updated_types, updated_c1, updated_c2 = self.type_updating_network(
            current_observation, previous_latent_state, current_latent_state
        )

        # Set updated type vectors as type vectors of the current particle
        setattr(current_latent_state, "theta", updated_types)
        setattr(current_latent_state,"cell1", updated_c1)
        setattr(current_latent_state, "cell2", updated_c2)

        # Get agent visibility distribution and observation vectors from current vector
        agent_visibility_dist, obs_feature_dist, u_feature_dist = self.decoder_network(
            current_latent_state
        )

        # Get obs vectors from current vector
        state_feature_dist, u_state_feature_dist = self.state_decoder_network(
            current_latent_state.detach()
        )

        # Compute likelihood of observed observation here
        agent_vis = current_observation.all_x[:,:,:,1].unsqueeze(-1)
        obs_vis_log_prob = agent_visibility_dist.log_prob(agent_vis)

        all_obs = torch.cat(
            [current_observation.all_x[:,:,:,:1],current_observation.all_x[:,:,:,2:]], dim=-1
        )

        all_states=torch.cat(
            [current_observation.all_x_state[:,:,:,:1],current_observation.all_x_state[:,:,:,2:]], dim=-1
        )

        agent_inputs = all_obs[:,:,:,:self.agent_inputs-1]
        u_obs = all_obs[:,:,0,self.agent_inputs-1:]

        a_obs_log_prob = obs_feature_dist.log_prob(
            agent_inputs
        )

        u_obs_log_prob = 0
        if self.with_global_feature:
            u_obs_log_prob = u_feature_dist.log_prob(
                u_obs
            )

        all_agent_states = all_states[:,:,:,:self.agent_inputs-1]
        u_state = all_states[:, :, 0, self.agent_inputs - 1:]

        a_state_log_prob = state_feature_dist.log_prob(
            all_agent_states
        )

        u_state_log_prob = 0
        if self.with_global_feature:
            u_state_log_prob = u_state_feature_dist.log_prob(
                u_state
            )

        # Add agent existence loss component
        states_reconstruction_component = None
        if not eval:
            states_reconstruction_component = agent_state_existence_logs.mean(dim=-1).mean(dim=-1)

        # Compute log weight
        # Use REINFORCE trick in SWB to train distributions that output discrete random variables

        if self.action_type == "discrete":
            # Use REINFORCE trick
            action_component_log_prob = all_agents_predicted_action_log_prob.detach() + \
                                        all_agents_proposed_action_log_prob - \
                                        all_agents_proposed_action_log_prob.detach()
        else:
            action_component_log_prob = all_agents_predicted_action_log_prob.detach() - all_agents_proposed_action_log_prob
        action_component_log_prob = action_component_log_prob.mean(dim=-1).mean(dim=-1)

        # Use REINFORCE trick for agent existence
        agent_existence_component = agent_log_existence - \
                                    (proposed_agent_log_existence.detach())

        agent_existence_component = agent_existence_component.mean(dim=-1).mean(dim=-1)

        state_component = state_logprob - proposed_state_log_prob
        # old way was~
        # state_component = state_component.mean(dim=-1).sum(dim=-1)
        # new way is: 
        state_component = state_component.sum(dim=-1).mean(dim=-1)
        

        obs_reconstruction_component = obs_vis_log_prob + a_obs_log_prob
        # same here 
        # old way
        # obs_reconstruction_component = obs_reconstruction_component.mean(dim=-1).mean(dim=-1)
        # new way 
        obs_reconstruction_component = obs_reconstruction_component.sum(dim=-1).mean(dim=-1)

        if self.with_global_feature:
            obs_reconstruction_component = obs_reconstruction_component + (u_obs_log_prob.mean(dim=-1))

        new_log_weight = action_component_log_prob + \
                         agent_existence_component + \
                         state_component + \
                         obs_reconstruction_component

        state_reconstruction_log_prob = a_state_log_prob.mean(dim=-1).mean(dim=-1) + u_state_log_prob.mean(dim=-1)

        # Set the updated log weight in current particle as new_log_weight
        setattr(current_latent_state, "log_weight", new_log_weight)

        # TODO
        # Average (in log space) over particles
        encoding_logli = torch.logsumexp(
            # torch.stack(log_weights, dim=0), dim=2
            new_log_weight, dim=1
        ) - np.log(self.num_particles)

        # inference_result.latent_states = latent_states

        ae_util.init(False)

        if self.with_global_feature:
            predicted_obs = torch.cat([
                obs_feature_dist.mean,
                u_feature_dist.mean.unsqueeze(-2).repeat([1,1,self.num_agents,1])
            ], dim=-1)
        else:
            predicted_obs = obs_feature_dist.mean

        # Initialize dictionaries to store other values for logging
        others = {}
        others["predicted_action_parameters"] = predictor_action_parameters
        others["agent_predicted_action_log_prob"] = all_agents_predicted_action_log_prob
        others["agent_proposed_action_log_prob"] = all_agents_proposed_action_log_prob
        others["agent_existence_log_prob"] = agent_log_existence
        others["proposed_agent_existence_log_prob"] = proposed_agent_log_existence
        others["state_log_prob"] = state_logprob
        others["proposed_state_log_prob"] = proposed_state_log_prob
        others["agent_obs_visibility_log_prob"] = obs_vis_log_prob
        others["agent_features_log_prob"] = a_obs_log_prob
        others["u_features_log_prob"] = u_obs_log_prob
        others["proposed_action_entropy"] = all_agents_proposed_action_entropy
        others["proposed_agent_existence_entropy"] = proposed_agent_existence_entropy
        others["obs_feature_dist_mean"] = predicted_obs
        others["num_existing_agents"] = self.num_agents*(current_latent_state.i==1).float().mean(dim=-2).mean()
        others["action_reconstruction_log_prob"] = action_reconstruction_log_prob
        others["graph"] = graph_batch
        others["agent_existence_logits"] = agent_existence_logits
        others["all_other_agent_distribution"] = all_agent_distribution
        others["states_reconstruction_component"] = states_reconstruction_component
        others["proposed_state_agent_existence"] = agent_state_existence_logs.mean() if not agent_state_existence_logs is None else None
        others["state_reconstruction_log_prob"] = state_reconstruction_log_prob.mean()

        return current_latent_state, \
               -encoding_logli, \
               -new_log_weight, \
               avg_num_killed_particles, \
               others

    def predict_action(self, graph_batch, previous_latent_state):
        predictor_action_parameters = self.act_predictor_net(
            graph_batch, previous_latent_state
        )
        all_agent_distribution = None
        if self.action_type == "continuous":
            all_agent_distribution = dist.Normal(
                predictor_action_parameters[0],
                predictor_action_parameters[1]
            )
        else:
            all_agent_distribution = dist.OneHotCategorical(
                logits=predictor_action_parameters
            )

        return all_agent_distribution

class DecisionMakingModel(nn.Module):
    def __init__(self,
                 action_space,
                 s_dim,
                 theta_dim,
                 device,
                 mid_pair,
                 mid_nodes,
                 mid_pair_out=8,
                 ):
        super(DecisionMakingModel,self).__init__()
        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
            self.action_type = "discrete"
        else:
            action_shape = action_space.shape[0]
            self.action_type = "continuous"

        self.device = device
        self.cg = JointUtilLayer(s_dim+theta_dim,mid_pair, mid_nodes, action_shape, mid_pair_out, device=self.device).to(self.device)

    def cg_input_preprocess(self, graph_batch, latent_states):
        # Get i, s, theta components of latent states and use it as input to CG
        i, s, theta = latent_states.i, latent_states.s, latent_states.theta
        num_agents = s.size()[-2]
        node_representations = torch.cat([s, theta], dim=-1)
        zero_node_representations = node_representations[:, :, 0, :].unsqueeze(2).repeat([1, 1, num_agents, 1])
        zero_edge_representations = node_representations[:, :, 0, :].unsqueeze(2).repeat(
            [1, 1, num_agents*(num_agents-1), 1]
        )

        final_node_representations = torch.cat([zero_node_representations, node_representations], dim=-1)
        final_node_representations = final_node_representations.view(-1, final_node_representations.size()[-1])

        edges = graph_batch.edges()
        edge_zero_node_data = zero_edge_representations.view(-1, zero_edge_representations.size()[-1])
        resized_node_data = node_representations.view(-1, node_representations.size()[-1])

        edge_representation_src = resized_node_data[edges[0]]
        edge_representation_dst = resized_node_data[edges[1]]

        final_edge_representation = torch.cat([
            edge_zero_node_data, edge_representation_src, edge_representation_dst
        ], dim=-1)
        final_edge_representation_flipped = torch.cat([
            edge_zero_node_data, edge_representation_dst, edge_representation_src
        ], dim=-1)

        existence_flags = i.view(-1, i.size()[-1])

        return final_node_representations, final_edge_representation, final_edge_representation_flipped, existence_flags

    def cg_joint_actions_preprocess(self, latent_states, all_actions):

        num_particles = latent_states.s.size()[1]
        all_particle_actions = all_actions.unsqueeze(1).repeat(1, num_particles, 1, 1)
        all_particle_actions = all_particle_actions.view(-1, all_particle_actions.size()[-1])

        # Replace sampled action with real agent action in case agent is visible in obs
        valid_indices = (all_particle_actions.sum(dim=-1) != 0)
        joint_acts = torch.zeros_like(valid_indices).long().to(self.device)
        joint_acts[valid_indices] = torch.argmax(all_particle_actions[valid_indices], dim=-1)

        return joint_acts.tolist(), valid_indices.unsqueeze(-1).double()


    def forward(
            self, graph_batch, latent_states, mode="train", action_dist = None, all_actions=None
    ):

        # Get latent state after filtering in inference mode
        node_probability = action_dist.probs.detach()

        # Preprocess CG input
        node, edge, edge_flipped, existence = self.cg_input_preprocess(graph_batch, latent_states)

        # Prepare joint action if call not in inference mode
        joint_acts, valid_indices = None, None
        if not "inference" in mode:
            joint_acts, valid_indices = self.cg_joint_actions_preprocess(latent_states, all_actions)

        node_probability = node_probability.view(-1, node_probability.shape[-1])

        q_vals = self.cg(
            graph_batch, node, edge, edge_flipped,
            existence, mode=mode, node_probability=node_probability,
            joint_acts=joint_acts, visibility_flags=valid_indices
        )

        return q_vals


class EncoderNet(nn.Module):
    """
        A class that encapsulates the model used to encode learner's observations & actions.
    """
    def __init__(self, nr_actions, action_encoding, nr_inputs, layer_dims, encoder_batch_norm):
        """
        Args:
            nr_actions : Action vector length
            action_encoding : Action vector representation length
            nr_input : Obs vector length
            layer_dims : Encoder layers dimensions
            encoder_batch_norm : Flag on whether to use batch norm or not for encoder
        """
        super().__init__()
        # We will use fc obs type
        self.action_encoding = action_encoding
        assert(action_encoding > 0)

        self.phi_x = Encoder(
            nr_inputs,
            layer_dims,
            batch_norm=encoder_batch_norm
        )

        def mul(num1, num2):
            return num1 * num2

        self.output_dimension = [layer_dims[-1]]
        self.cnn_output_number = reduce(mul, self.output_dimension, 1)

        if encoder_batch_norm:
            self.action_encoder = nn.Sequential(
                nn.Linear(nr_actions, action_encoding),
                nn.BatchNorm1d(action_encoding),
                nn.ReLU()
            )
        else:
            self.action_encoder = nn.Sequential(
                nn.Linear(nr_actions, action_encoding),
                nn.ReLU()
            )
        self.nr_actions = nr_actions

    def forward(self, observation_states, action_only=False):
        """
        Method that encodes the observation & action experienced by agent

        Input:
        - Observations_states containing `all_x`    [seq_len, batch_size, num_agents, obs_feature_size]

        Output:
        - Observations_states with additional entry `all_phi_x` (Encoding of observation)
          [seq_len, batch_size, num_agents, encoding_dim]
        - Observations_states with additional entry `encoded_action` (Encoding of action)
          [seq_len, batch_size, num_agents, encoding_dim]
        """

        if not action_only:
            seq_len, batch_size, *obs_dim = observation_states.all_x.size()
            num_agents, input_dim = obs_dim[0], obs_dim[1]

            # Encode the observations and expand
            all_phi_x = self.phi_x(
                observation_states.all_x.view(-1, *obs_dim)  # Collapse particles
                ).view(-1, self.cnn_output_number)  # Flatten CNN output

            all_phi_x = all_phi_x.view(seq_len, batch_size, num_agents, -1)
            observation_states.all_phi_x = all_phi_x

            if self.action_encoding > 0:
                encoded_action = self.action_encoder(
                    observation_states.all_a.view(-1, self.nr_actions)
                )
                encoded_action = encoded_action.view(seq_len, batch_size, -1)
                encoded_action = encoded_action.unsqueeze(2).repeat([1,1,num_agents,1])

                observation_states.encoded_action = encoded_action

            return observation_states

        batch_size, num_particles, num_agents, act_dim = observation_states.size()

        encoded_action = self.action_encoder(
            observation_states.view(-1, act_dim)
        ).view(batch_size, num_particles, num_agents, -1)
        return encoded_action


# Removed under current assumptions
# class ObservationLearnerActionEncoderNet(nn.Module):
#     """
#         A class that encodes observation-learner actions into a fixed length vector for further processing.
#     """
#
#     def __init__(
#             self, o_dim, action_encoding,
#             o_edge_dim, o_gnn_hdim1, o_gnn_hdim_2,
#             o_out_hdim1, o_out_hdim2, o_out_dim
#     ):
#         super().__init__()
#
#         self.gnn_obs_processing = GNNBlock(
#             o_dim + action_encoding, 0, o_edge_dim,
#             o_gnn_hdim1, o_gnn_hdim_2, o_out_hdim1,
#             False
#         )
#
#         self.obs_fc = nn.Sequential(
#             nn.Linear(2 * o_out_hdim1, o_out_hdim2),
#             nn.ReLU(),
#             nn.Linear(o_out_hdim2, o_out_dim)
#         )
#
#     def forward(self, graph_obs, observation_states, message_passing_steps=1):
#         obs_sizes = observation_states.all_phi_x.size()
#         batch_size, phi_x_dim = obs_sizes[0], obs_sizes[-1]
#
#         if self.action_encoding > 0:
#             input = torch.cat([
#                 observation_states.all_phi_x,
#                 observation_states.encoded_action
#             ], -1).view(-1, phi_x_dim + self.action_encoding)
#         else:
#             input = torch.cat([
#                 observation_states.all_phi_x
#             ], -1).view(-1, phi_x_dim)
#
#         # Get node & edge representation from GNN
#         n_out, e_out = self.gnn_obs_processing(graph_obs, input, message_passing_steps, None, with_aggregation=True)
#
#         # Aggregate representations for each graph in batch using summation
#         graph_obs.ndata["final_node_readout"] = n_out
#         graph_obs.edata["final_edge_readout"] = e_out
#
#         n_out_readout = dgl.readout_nodes(graph_obs, "final_node_readout", op="sum")
#         e_out_readout = dgl.readout_nodes(graph_obs, "final_edge_readout", op="sum")
#         input_obs = self.obs_fc(torch.cat([n_out_readout, e_out_readout], dim=-1))
#
#         h_sizes = previous_latent_state.theta.size()
#         batch_size, num_particles, num_agent, h_dim = h_sizes[0], h_sizes[1], h_sizes[2], h_sizes[-1]
#         obs_rep = input_obs.view([batch_size, num_particles, 1, -1]).repeat([1, 1, num_agent, 1])
#
#         return obs_rep



class ActionPredictionNet(nn.Module):
    """
        A class that predicts the parameters of teammates' action distributions given their type vectors,
        inferred state, and existence at previous timestep.
    """
    def __init__(
            self, s_dim, h_dim, mid_dim, gnn_hdim1, gnn_hdim2, gnn_out_dim, act_dim, action_type, device
        ):
        """
            Args:
                s_dim : The length of state vectors in a particle.
                h_dim : The length of agent type vectors in a particle.
                edge_dim : The length of message vectors exchanged between nodes in the GNN.
                gnn_hdim1 : The size of the first hidden layer of the GNN used in this model.
                gnn_hdim2 : The size of the second hidden layer of the GNN used in this model.
                gnn_out_dim : The size of the output vector of the GNN used in this model.
                act_dim : The action space size for the model.
                action_type : The type of action space (continuous/discrete)
        """
        super().__init__()

        self.inp_processor = nn.Sequential(
            nn.Linear(s_dim + h_dim + 1, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim)
        )

        self.device = device

        self.gnn = RFMBlock(
            mid_dim, 0, 0, gnn_hdim1, gnn_hdim2, gnn_out_dim
        )
        self.action_type = action_type

        if self.action_type == "continuous":
            self.var_act_net = nn.Sequential(
                nn.Linear(gnn_out_dim, gnn_out_dim),
                nn.ReLU(),
                nn.Linear(gnn_out_dim, act_dim),
                nn.Softplus()
            )
            self.mean_act_net = nn.Sequential(
                nn.Linear(gnn_out_dim, gnn_out_dim),
                nn.ReLU(),
                nn.Linear(gnn_out_dim, act_dim)
            )
        else:
            self.logit_net = nn.Sequential(
                nn.Linear(gnn_out_dim, gnn_out_dim),
                nn.ReLU(),
                nn.Linear(gnn_out_dim, act_dim)
            )


    def forward(self, graph, previous_latent_state):
        """
        Outputs the logits of p(a_t|s_{t},\theta_{t}, i_{t}).

        Inputs:
            - previous_latent_state containing at least
                `theta`     of dimensions [batch, particles, num_agents, type_vector_dim]
                `s`     of dimensions [batch, particles, num_agents, state_vector_dim]
                `i`     of dimensions [batch, particles, num_agents, 1]
        """

        input_h = previous_latent_state.theta
        input_s = previous_latent_state.s
        input_i = previous_latent_state.i

        h_sizes = input_h.size()
        s_sizes = input_s.size()

        batch_size, num_particles, num_agents, h_dim = h_sizes[0], h_sizes[1], h_sizes[2], h_sizes[-1]
        s_dim = s_sizes[-1]

        input = torch.cat([
            input_h,
            input_s,
            input_i
        ], -1).view(-1, h_dim + s_dim + 1)

        n_inp = self.inp_processor(input)

        _, n_out, _ = self.gnn(
            graph,
            torch.zeros(graph.number_of_edges(),0).to(self.device).double(),
            n_inp,
            torch.zeros(batch_size*num_particles,0).to(self.device).double()
        )

        if self.action_type == "continuous":
            act_mean, act_var = self.mean_act_net(n_out), self.var_act_net(n_out)
            return act_mean.view(batch_size, num_particles, num_agents, -1), \
                   act_var.view(batch_size, num_particles, num_agents, -1)
        else:
            act_logits = self.logit_net(n_out).view(batch_size, num_particles, num_agents, -1)
            return act_logits

class ProposalActionPredictionNet(nn.Module):
    """
        A class that proposes the previous actions of other agents for a particle.
    """

    def __init__(
            self, o_enc_dim, action_encoding, s_dim, h_dim,
            mid_dim, gnn_hdim1, gnn_hdim2, gnn_out_dim,
            act_dim, action_type, device
    ):
        """
            Args:
                o_enc_dim : The length of observation encodings in the current approach.
                action_encoding : The length of action encodings in the current approach.
                s_dim : The length of state vectors in a particle.
                h_dim : The length of agent type vectors in a particle.
                edge_dim : The length of message vectors exchanged between nodes in the GNN.
                gnn_hdim1 : The size of the first hidden layer of the GNN used in this model.
                gnn_hdim2 : The size of the second hidden layer of the GNN used in this model.
                gnn_out_dim : The size of the output vector of the GNN used in this model.
                act_dim : The action space size for the model.
                action_type : The type of action space (continuous/discrete)
        """
        super().__init__()
        self.device = device
        self.inp_processor = nn.Sequential(
            nn.Linear(s_dim + h_dim + o_enc_dim + action_encoding + 1, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim)
        )

        self.state_processor = nn.Sequential(
            nn.Linear(mid_dim, gnn_hdim1),
            nn.ReLU(),
            nn.Linear(gnn_hdim1, gnn_out_dim)
        )

        self.action_type = action_type

        if self.action_type == "continuous":
            self.var_act_net = nn.Sequential(
                nn.Linear(gnn_out_dim, gnn_out_dim),
                nn.ReLU(),
                nn.Linear(gnn_out_dim, act_dim),
                nn.Softplus()
            )
            self.mean_act_net = nn.Sequential(
                nn.Linear(gnn_out_dim, gnn_out_dim),
                nn.ReLU(),
                nn.Linear(gnn_out_dim, act_dim)
            )
        else:
            self.logit_net = nn.Sequential(
                nn.Linear(gnn_out_dim, gnn_out_dim),
                nn.ReLU(),
                nn.Linear(gnn_out_dim, act_dim)
            )

    def forward(self, observation_states, previous_latent_state):
        """
        Outputs the logits of p(a_t|s_{t},\theta_{t},i_{t}, o_{t+1}).

        Inputs:
            - previous_latent_state containing at least
                `theta`     of dimensions [batch, particles, num_agents, type_vector_dim]
                `s`     of dimensions [batch, particles, num_agents, state_vector_din]
                `i`     of dimensions [batch, particles, num_agents, 1]
            - observations containing at least
                `all_phi_x`     of dimensions [batch, particles, num_agents, obs_encoding_dim]
                `encoded_action` of dimension [batch, particles, num_agents, action_encoding_dim]
        """

        input_h = previous_latent_state.theta
        input_s = previous_latent_state.s
        input_i = previous_latent_state.i
        input_obs = observation_states.all_phi_x
        input_act = observation_states.encoded_action

        h_sizes = input_h.size()
        s_sizes = input_s.size()

        batch_size, num_particles, num_agent, h_dim = h_sizes[0], h_sizes[1], h_sizes[2], h_sizes[-1]
        s_dim = s_sizes[-1]

        #input_obs = input_obs.view([batch_size, num_particles, 1, -1]).repeat([1,1, num_agent,1])
        input_obs_dim = input_obs.size()[-1]
        acts_dim = input_act.size()[-1]

        input = torch.cat([
            input_h,
            input_s,
            input_i,
            input_obs,
            input_act
        ], -1).view(-1, h_dim + s_dim + 1 + input_obs_dim + acts_dim)

        n_out = self.state_processor(self.inp_processor(input))

        if self.action_type == "continuous":
            act_mean, act_var = self.mean_act_net(n_out), self.var_act_net(n_out)
            return act_mean.view(batch_size, num_particles, num_agent, -1), \
                   act_var.view(batch_size, num_particles, num_agent, -1)
        else:
            act_logits = self.logit_net(n_out)
            return act_logits.view(batch_size, num_particles, num_agent, -1)

class StatePredictionNetwork(nn.Module):
    """
        A class that predicts the existence of other agents and agents' state representation at next timestep
         given their type vectors, state vectors, and previous actions.
    """
    def __init__(
            self, action_encoding, s_dim, h_dim, edge_dim,
            gnn_hdim1, gnn_hdim2, gnn_out_dim, s_projection_size,
            autoreg_m_size, device
        ):
        """
            Args:
                action_encoding : The length of action encodings in the current approach.
                s_dim : The length of state vectors in a particle.
                h_dim : The length of agent type vectors in a particle.
                edge_dim : The length of message vectors exchanged between nodes in the GNN.
                gnn_hdim1 : The size of the first hidden layer of the GNN used in this model.
                gnn_hdim2 : The size of the second hidden layer of the GNN used in this model.
                gnn_out_dim : The size of the output vector of the GNN used in this model.
                s_projection_size : First hidden layer size for state vector computation.
                autoreg_m_size : Second hidden layer size when computing messages for autoregressive reasoning.
        """
        super().__init__()

        self.state_network = nn.Sequential(
            nn.Linear(s_dim + h_dim + action_encoding + 1, gnn_hdim1),
            nn.ReLU(),
            nn.Linear(gnn_hdim1, gnn_hdim2),
            nn.ReLU(),
            nn.Linear(gnn_hdim2, gnn_out_dim)
        )

        self.device = device

        self.s_update_net = nn.Sequential(
            nn.Linear(gnn_out_dim, s_projection_size),
            nn.ReLU()
        )

        self.s_update_net_mean = nn.Linear(s_projection_size, s_dim)
        self.s_update_net_var = nn.Sequential(
            nn.Linear(s_projection_size, s_dim),
            nn.Softplus()
        )

        self.edge_net = nn.Linear(2*gnn_out_dim+1, autoreg_m_size)
        self.logit_net = nn.Sequential(
            nn.Linear(gnn_out_dim+autoreg_m_size+1, 1)
        )

        self.action_encoding = action_encoding

    def forward(self, previous_latent_state, current_latent_state, temperature=1.0):
        """
            Outputs the logits of p(s_{t+1}, i_{t+1}|s_{t},\theta_{t},i_{t}).

        Inputs:
            - Input graph structure for GNN
            - previous_latent_state containing at least
                `theta`     of dimensions [batch, particles, num_agents, type_vector_dim]
                `s`     of dimensions [batch, particles, num_agents, s_dim]
                `i`     of dimensions [batch, particles, num_agents, 1]
            - current_latent_state containing at least
                `all_encoded_actions`     of dimensions [batch, particles, num_agents, act_encoding_dim]
                `i` of dimensions [batch, particles, num_agents, 1]
        """

        input_h = previous_latent_state.theta
        input_s = previous_latent_state.s
        prev_flag = previous_latent_state.i
        prev_acts = current_latent_state.all_encoded_actions
        sampled_i = current_latent_state.i

        h_sizes = input_h.size()
        s_sizes = input_s.size()

        batch_size, num_particles, num_agent, h_dim = h_sizes[0], h_sizes[1], h_sizes[2], h_sizes[-1]
        s_dim = s_sizes[-1]

        input = torch.cat([
                input_h,
                input_s,
                prev_acts,
                prev_flag
        ], -1).view(-1, h_dim + s_dim + self.action_encoding + 1)

        n_out= self.state_network(input)
        reshaped_n_out = n_out.view(batch_size, num_particles, num_agent, -1)

        # Assume agent index 0 for the learner always exists
        agent_flags = [sampled_i[:,:,0,:].unsqueeze(-2)]
        node_logs = [torch.zeros(batch_size, num_particles, 1, 1).to(self.device).double()]

        # Implement autoregressive reasoning
        for agent_offset in range(1, num_agent):
            # Get node representation of other agents that have been added to particle
            other_agent_data = reshaped_n_out[:, :, :agent_offset, :]

            # Get node representation of the agent that is to be added to the particle
            added_agent_data = reshaped_n_out[:, :, agent_offset, :]
            added_agent_prev_flag = prev_flag[:, :, agent_offset, :]


            added_agent_data_reshaped = added_agent_data.unsqueeze(-2).repeat([1, 1, agent_offset, 1])
            concatenated_agent_flags = torch.cat(agent_flags, dim=-2)

            # Autoregressive input will be representation of agents that already exist
            # Representation of agent to be added to the environment
            # and data on the agents' existence at the previous timestep
            edge_data = torch.cat([other_agent_data, added_agent_data_reshaped, concatenated_agent_flags], dim=-1)

            # Sum message from all existing nodes for deciding whether new agent is added or not
            edge_msg = self.edge_net(edge_data).sum(dim=-2)

            # Compute p(add new agent | prev agents data)
            input_logit_data = torch.cat([added_agent_data, edge_msg, added_agent_prev_flag], dim=-1)
            added_probs_logits = self.logit_net(input_logit_data).view(batch_size, num_particles, 1, -1)/temperature
            added_probs = F.sigmoid(added_probs_logits)
            node_dist = dist.Bernoulli(probs=added_probs)
            node_sample = sampled_i[:,:,agent_offset,:].unsqueeze(-2)
            add_logs = node_dist.log_prob(node_sample)

            # Output lsample existence (0/1) and log prob of sample
            new_flags = node_sample.view(batch_size, num_particles, 1, -1)
            new_logs = add_logs.view(batch_size, num_particles, 1, -1)

            agent_flags.append(new_flags)
            node_logs.append(new_logs)

        agent_existence_logs = torch.cat(node_logs, dim=-2)

        # Compute the updated state representations
        updated_agent_rep = self.s_update_net(reshaped_n_out)
        s_rep_mean, s_rep_var = self.s_update_net_mean(updated_agent_rep), self.s_update_net_var(updated_agent_rep)

        return agent_existence_logs, (s_rep_mean, s_rep_var)

class ProposalStatePredictionNetwork(nn.Module):
    """
        A class that proposes the current state representation for a particle (s_{t})
        and the current existence of agents (i_{t}).
    """

    def __init__(
            self, o_enc_dim, action_encoding, s_dim, h_dim,
            edge_dim, gnn_hdim1, gnn_hdim2, gnn_out_dim,
            s_projection_size, autoreg_m_size, device
    ):
        """
            Args:
                o_enc_dim : The length of observation encodings in the current approach.
                action_encoding : The length of action encodings in the current approach.
                s_dim : The length of state vectors in a particle.
                h_dim : The length of agent type vectors in a particle.
                edge_dim : The length of message vectors exchanged between nodes in the GNN.
                gnn_hdim1 : The size of the first hidden layer of the GNN used in this model.
                gnn_hdim2 : The size of the second hidden layer of the GNN used in this model.
                gnn_out_dim : The size of the output vector of the GNN used in this model.
                s_projection_size : First hidden layer size for state vector computation.
                autoreg_m_size : Second hidden layer size when computing messages for autoregressive reasoning.
        """
        super().__init__()
        self.device = device
        self.gnn = nn.Sequential(
            nn.Linear(s_dim + h_dim + 2*action_encoding+ 1 + o_enc_dim, gnn_hdim1),
            nn.ReLU(),
            nn.Linear(gnn_hdim1, gnn_hdim2),
            nn.ReLU(),
            nn.Linear(gnn_hdim2,gnn_out_dim)
        )

        self.s_update_net = nn.Sequential(
            nn.Linear(gnn_out_dim, s_projection_size),
            nn.ReLU()
        )

        self.s_update_net_mean = nn.Linear(s_projection_size, s_dim)
        self.s_update_net_var = nn.Sequential(
            nn.Linear(s_projection_size, s_dim),
            nn.Softplus()
        )

        self.edge_net = nn.Linear(2 * gnn_out_dim + 1, autoreg_m_size)
        self.logit_net = nn.Sequential(
            nn.Linear(gnn_out_dim + autoreg_m_size + 1, 1),
        )

        self.action_encoding = action_encoding

    def forward(self, observation_states, previous_latent_state, current_latent_state, temperature=1.0, eval=False):
        """
            Outputs the logits of p(s_{t+1}, i_{t+1}|s_{t},\theta_{t},i_{t}, o_{t+1}).

        Inputs:
            - Input graph structure for GNN
            - previous_latent_state containing at least
                `theta`     of dimensions [batch, particles, num_agents, type_vector_dim]
                `s`     of dimensions [batch, particles, num_agents, state_vector_dim]
                `i`     of dimensions [batch, particles, num_agents, 1]
            - current_latent_state containing at least
                `all_encoded_actions`     of dimensions [batch, particles, num_agents, act_encoding_dim]
            - observations containing at least
                `all_phi_x`     of dimensions [batch, particles, num_agents, obs_encoding_dim]
                `encoded_action` of dimension [batch, particles, num_agents, action_encoding_dim]
        """

        input_h = previous_latent_state.theta
        input_s = previous_latent_state.s
        prev_flag = previous_latent_state.i
        prev_acts = current_latent_state.all_encoded_actions
        input_obs = observation_states.all_phi_x
        input_act = observation_states.encoded_action
        if not eval :
            sampled_i = observation_states.all_state_existence

        h_sizes = input_h.size()
        s_sizes = input_s.size()

        batch_size, num_particles, num_agent, h_dim = h_sizes[0], h_sizes[1], h_sizes[2], h_sizes[-1]
        s_dim = s_sizes[-1]

        # Removed since not needed for current assumption
        #input_obs = input_obs.view([batch_size, num_particles, 1, -1]).repeat([1,1, num_agent,1])
        input_obs_dim = input_obs.size()[-1]

        input = torch.cat([
            input_h,
            input_s,
            prev_acts,
            prev_flag,
            input_obs,
            input_act
        ], -1).view(-1, h_dim + s_dim + 2*self.action_encoding + 1 + input_obs_dim)

        n_out = self.gnn(input)
        reshaped_n_out = n_out.view(batch_size, num_particles, num_agent, -1)

        # Assume agent index 0 for the learner
        # Also assume learner always exists
        agent_flags = [torch.ones(batch_size, num_particles, 1, 1).to(self.device).double()]
        node_logs = [torch.zeros(batch_size, num_particles, 1, 1).to(self.device).double()]
        node_logits = [torch.zeros(batch_size, num_particles, 1, 1).to(self.device).double()]
        node_entropy = [torch.zeros(batch_size, num_particles, 1, 1).to(self.device).double()]

        if not eval :
            agent_state_flags = [sampled_i[:, :, 0, :].unsqueeze(-2)]
            node_state_logs = [torch.zeros(batch_size, num_particles, 1, 1).to(self.device).double()]

        # Implement autoregressive reasoning
        for agent_offset in range(1, num_agent):
            # Get node representation of other agents that have been added to particle
            other_agent_data = reshaped_n_out[:, :, :agent_offset, :]

            # Get node representation of the agent that is to be added to the particle
            added_agent_data = reshaped_n_out[:, :, agent_offset, :]
            added_agent_prev_flag = prev_flag[:, :, agent_offset, :]

            added_agent_data_reshaped = added_agent_data.unsqueeze(-2).repeat([1, 1, agent_offset, 1])
            concatenated_agent_flags = torch.cat(agent_flags, dim=-2)
            if not eval:
                concatenated_state_agent_flags = torch.cat(agent_state_flags, dim=-2)

            # Autoregressive input will be representation of agents that already exist
            # Representation of agent to be added to the environment
            # and data on the agents' existence at the previous timestep
            edge_data = torch.cat([other_agent_data, added_agent_data_reshaped, concatenated_agent_flags], dim=-1)
            if not eval:
                state_edge_data = torch.cat([other_agent_data, added_agent_data_reshaped, concatenated_state_agent_flags], dim=-1)

            # Sum message from all existing nodes for deciding whether new agent is added or not
            edge_msg = self.edge_net(edge_data).sum(dim=-2)
            if not eval:
                state_edge_msg = self.edge_net(state_edge_data).sum(dim=-2)

            # Compute p(add new agent | prev agents data)
            input_logit_data = torch.cat([added_agent_data, edge_msg, added_agent_prev_flag], dim=-1)
            added_probs_logits = self.logit_net(input_logit_data).view(batch_size, num_particles, 1, -1)
            if not eval:
                input_logit_state_data = torch.cat([added_agent_data, state_edge_msg, added_agent_prev_flag], dim=-1)
                added_probs_states_logits = self.logit_net(input_logit_state_data).view(batch_size, num_particles, 1, -1)

            added_probs = F.sigmoid(added_probs_logits/temperature)
            node_dist = dist.Bernoulli(probs=added_probs)
            node_sample_t = node_dist.sample()
            node_logs_t = node_dist.log_prob(node_sample_t)
            node_dist_entropy = node_dist.entropy()

            if not eval:
                added_state_probs = F.sigmoid(added_probs_states_logits / temperature)
                node_state_dist = dist.Bernoulli(probs=added_state_probs)
                node_state_logs_t = node_state_dist.log_prob(sampled_i[:,:,agent_offset,:].unsqueeze(-2))

            # Output sample existence (0/1) and log prob of sample
            new_flags = node_sample_t.view(batch_size, num_particles, 1, -1)
            if not eval:
                new_state_flags = sampled_i[:,:,agent_offset,:].unsqueeze(-2)
            new_logs = node_logs_t.view(batch_size, num_particles, 1, -1)
            new_entropy = node_dist_entropy.view(batch_size, num_particles, 1, -1)

            agent_flags.append(new_flags)
            node_logs.append(new_logs)
            node_logits.append(added_probs_logits)
            node_entropy.append(new_entropy)

            if not eval:
                agent_state_flags.append(new_state_flags)
                node_state_logs.append(node_state_logs_t)

        updated_agent_flags = torch.cat(agent_flags, dim=-2)
        agent_existence_logs = torch.cat(node_logs, dim=-2)
        agent_existence_entropy = torch.cat(node_entropy, dim=-2)
        all_node_logits = torch.cat(node_logits, dim=-2)

        agent_state_existence_logs = None
        if not eval:
            agent_state_existence_logs = torch.cat(node_state_logs, dim=-2)

        # Compute the updated state representations
        updated_agent_rep = self.s_update_net(reshaped_n_out)
        s_rep_mean, s_rep_var = self.s_update_net_mean(updated_agent_rep), self.s_update_net_var(updated_agent_rep)
        s_rep_dist = dist.Normal(s_rep_mean, s_rep_var)
        s_rep_sample = s_rep_dist.rsample() * updated_agent_flags

        return updated_agent_flags, agent_existence_logs, all_node_logits, agent_existence_entropy, s_rep_sample, s_rep_dist, agent_state_existence_logs

class TypeUpdateNet(nn.Module):
    """
        A class that deterministically updates the latent type of the agents conditioned on their
        new state vector and previous action & type vector.
    """

    def __init__(
            self, o_enc_dim, action_encoding, s_dim, h_dim
    ):
        """
            Args:
                o_enc_dim : The length of observation encodings in the current approach.
                action_encoding : The length of action encodings in the current approach.
                s_dim : The length of state vectors in a particle.
                h_dim : The length of agent type vectors in a particle.
        """
        super().__init__()

        self.type_rnn = nn.LSTM(s_dim + action_encoding + o_enc_dim + 1, h_dim, batch_first=False)
        self.action_encoding = action_encoding

    def forward(self, observation_states, previous_latent_state, current_latent_state):
        """
            Outputs the updated type vector of all agents given the
            observed observation, inferred states, previous action and previous type of all agents

        Inputs:
            - Input graph structure for GNN
            - observations containing at least
                `all_phi_x`     of dimensions [batch, particles, num_agents, obs_encoding_dim]
                `encoded_action` of dimension [batch, particles, num_agents, act_encoding_dim]
            - previous_latent_state containing at least
                `theta`     of dimensions [batch, particles, num_agents, type_vector_dim]
                `i`     of dimensions [batch, particles, num_agents, 1]
            - current_latent_state containing at least
                `s`     of dimensions [batch, particles, num_agents, state_vector_dim]
                `i`     of dimensions [batch, particles, num_agents, 1]
                `all_encoded_actions`     of dimensions [batch, particles, num_agents, act_encoding_dim]
        """

        input_c1 = previous_latent_state.cell1
        input_c2 = previous_latent_state.cell2
        input_s = current_latent_state.s
        cur_flag = current_latent_state.i
        input_obs = observation_states.all_phi_x
        agent_act = observation_states.encoded_action

        s_sizes = input_s.size()
        batch_size, num_particles, num_agent, s_dim = s_sizes[0], s_sizes[1], s_sizes[2],s_sizes[-1]

        # Not needed under current assumption
        # input_obs = input_obs.view([batch_size, num_particles, 1, -1]).repeat([1,1, num_agent,1])
        input_obs_dim = input_obs.size()[-1]

        input = torch.cat([
            input_s,
            cur_flag,
            input_obs,
            agent_act
        ], -1).view(-1, s_dim + 1 + self.action_encoding + input_obs_dim)

        n_out = input.view(-1,input.size(-1)).unsqueeze(0)
        input_c1 = input_c1.view(-1,input_c1.size(-1)).unsqueeze(0)
        input_c2 = input_c2.view(-1, input_c2.size(-1)).unsqueeze(0)

        updated_type, (new_c1, new_c2) = self.type_rnn(n_out,(input_c1, input_c2))
        res = updated_type.view(batch_size, num_particles, num_agent, -1) * cur_flag
        new_c1 = new_c1.view(batch_size, num_particles, num_agent, -1) * cur_flag
        new_c2 = new_c2.view(batch_size, num_particles, num_agent, -1) * cur_flag

        return res, new_c1, new_c2

class ObsDecoderNet(nn.Module):
    """
        A class that decodes latent variable estimates in a particle into graph-based representation of the observation.
    """

    def __init__(
            self, action_encoding, s_dim, h_dim,
            edge_dim, gnn_hdim1, gnn_hdim2,
            decoder_hid_dims, agent_obs_size,
            u_obs_size, device, with_global_feature
    ):
        """
            Args:
                action_encoding : The length of action encodings in the current approach.
                s_dim : The length of state vectors in a particle.
                h_dim : The length of agent type vectors in a particle.
                edge_dim : The length of message vectors exchanged between nodes in the GNN.
                gnn_hdim1 : The size of the first hidden layer of the GNN used in this model.
                gnn_hdim2 : The size of the second hidden layer of the GNN used in this model.
                gnn_out_dim : The size of the output vector of the GNN used in this model.
                with_global_feature : Flag in case agents need to reconstruct global features from obs.
        """

        super().__init__()
        self.device = device

        self.deconstruction_nn = nn.Sequential(
            nn.Linear(s_dim + h_dim + action_encoding + 1, gnn_hdim1),
            nn.ReLU(),
            nn.Linear(gnn_hdim1,gnn_hdim2),
            nn.ReLU(),
            nn.Linear(gnn_hdim2, decoder_hid_dims[-1])
        )

        self.visibility_net = nn.Sequential(
            nn.Linear(decoder_hid_dims[-1], decoder_hid_dims[0]),
            nn.ReLU(),
            nn.Linear(decoder_hid_dims[0], 1),
        )
        self.decoder = Decoder(agent_obs_size, decoder_hid_dims)
        self.with_global_feature = with_global_feature

        if self.with_global_feature:
            self.decoder_u = Decoder(u_obs_size, decoder_hid_dims)

        self.action_encoding = action_encoding

    def forward(self, current_latent_state):
        """
            Outputs the distribution that determines whether agents are visible/invisible in obs
            and the distribution over the features of agents that are visible in the obs

        Inputs:
            - Input graph structure for GNN
            - current_latent_state containing at least
                `theta`     of dimensions [batch, particles, num_agents, type_vector_dim]
                `s`     of dimensions [batch, particles, num_agents, state_vector_dim]
                `i`     of dimensions [batch, particles, num_agents, 1]
                `all_encoded_actions`     of dimensions [batch, particles, num_agents, act_encoding_dim]
        """

        input_h = current_latent_state.theta
        input_s = current_latent_state.s
        cur_flag = current_latent_state.i
        prev_acts = current_latent_state.all_encoded_actions

        h_sizes = input_h.size()
        s_sizes = input_s.size()

        batch_size, num_particles, num_agent, h_dim = h_sizes[0], h_sizes[1], h_sizes[2], h_sizes[-1]
        s_dim = s_sizes[-1]

        input = torch.cat([
            input_h,
            input_s,
            cur_flag,
            prev_acts
        ], -1).view(-1, h_dim + s_dim + 1 + self.action_encoding)

        n_out = self.deconstruction_nn(input)
        u_out = n_out.view(batch_size, num_particles, num_agent, -1).sum(dim=-2)

        agent_visibility_dist = dist.Bernoulli(
            probs=F.sigmoid(self.visibility_net(n_out).view(batch_size, num_particles, num_agent, -1))
        )

        agent_mean, agent_var = self.decoder(n_out)
        if self.with_global_feature:
            u_mean, u_var = self.decoder_u(u_out)

        agent_var = torch.ones_like(agent_var).to(self.device).double()

        if self.with_global_feature:
            u_var = torch.ones_like(u_var).to(self.device).double()

        agent_obs_dist = dist.Normal(
            agent_mean.view(batch_size, num_particles, num_agent, -1),
            agent_var.view(batch_size, num_particles, num_agent, -1)
        )

        u_obs_dist = None
        if self.with_global_feature:
            u_obs_dist = dist.Normal(
                u_mean.view(batch_size, num_particles, -1),
                u_var.view(batch_size, num_particles, -1)
            )

        return agent_visibility_dist, agent_obs_dist, u_obs_dist

class StateDecoderNet(nn.Module):
    """
        A class that decodes latent variable estimates in a particle into graph-based representation of the observation.
    """

    def __init__(
            self, action_encoding, s_dim, h_dim,
            edge_dim, gnn_hdim1, gnn_hdim2,
            decoder_hid_dims, agent_obs_size,
            u_obs_size, device, with_global_feature
    ):
        """
            Args:
                action_encoding : The length of action encodings in the current approach.
                s_dim : The length of state vectors in a particle.
                h_dim : The length of agent type vectors in a particle.
                edge_dim : The length of message vectors exchanged between nodes in the GNN.
                gnn_hdim1 : The size of the first hidden layer of the GNN used in this model.
                gnn_hdim2 : The size of the second hidden layer of the GNN used in this model.
                gnn_out_dim : The size of the output vector of the GNN used in this model.
                with_global_feature : Flag in case agents need to reconstruct global features from obs.
        """

        super().__init__()
        self.device = device
        self.deconstruction_nn = nn.Sequential(
            nn.Linear(s_dim + h_dim + action_encoding + 1, gnn_hdim1),
            nn.ReLU(),
            nn.Linear(gnn_hdim1, gnn_hdim2),
            nn.ReLU(),
            nn.Linear(gnn_hdim2, decoder_hid_dims[-1])
        )

        self.decoder = Decoder(agent_obs_size, decoder_hid_dims)
        self.with_global_feature = with_global_feature
        if self.with_global_feature:
            self.decoder_u = Decoder(u_obs_size, decoder_hid_dims)

        self.action_encoding = action_encoding

    def forward(self, current_latent_state):
        """
            Outputs the distribution that determines whether agents are visible/invisible in obs
            and the distribution over the features of agents that are visible in the obs

        Inputs:
            - current_latent_state containing at least
                `theta`     of dimensions [batch, particles, num_agents, type_vector_dim]
                `s`     of dimensions [batch, particles, num_agents, state_vector_dim]
                `i`     of dimensions [batch, particles, num_agents, 1]
                `all_encoded_actions`     of dimensions [batch, particles, num_agents, act_encoding_dim]
        """

        input_h = current_latent_state.theta
        input_s = current_latent_state.s
        cur_flag = current_latent_state.i
        prev_acts = current_latent_state.all_encoded_actions

        h_sizes = input_h.size()
        s_sizes = input_s.size()

        batch_size, num_particles, num_agent, h_dim = h_sizes[0], h_sizes[1], h_sizes[2], h_sizes[-1]
        s_dim = s_sizes[-1]

        input = torch.cat([
            input_h,
            input_s,
            cur_flag,
            prev_acts
        ], -1).view(-1, h_dim + s_dim + 1 + self.action_encoding)

        n_out = self.deconstruction_nn(input)
        u_out = n_out.view(batch_size, num_particles, num_agent, -1).sum(dim=-2)

        agent_mean, agent_var = self.decoder(n_out)
        if self.with_global_feature:
            u_mean, u_var = self.decoder_u(u_out)

        agent_var = torch.ones_like(agent_var).to(self.device).double()
        if self.with_global_feature:
            u_var = torch.ones_like(u_var).to(self.device).double()

        agent_obs_dist = dist.Normal(
            agent_mean.view(batch_size, num_particles, num_agent, -1),
            agent_var.view(batch_size, num_particles, num_agent, -1)
        )

        u_obs_dist = None
        if self.with_global_feature:
            u_obs_dist = dist.Normal(
                u_mean.view(batch_size, num_particles, -1),
                u_var.view(batch_size, num_particles, -1)
            )

        return agent_obs_dist, u_obs_dist
