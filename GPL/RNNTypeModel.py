import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import aesmc.rnn_state as st
import aesmc.util as ae_util
import aesmc.test_utils as tu
from aesmc.inference import sample_ancestral_index
import numpy as np
from BeliefModel import RNNTypeModel, RNNTypeModelStateRecons
from Network import RFMBlock, JointUtilLayer, Encoder, Decoder
import torch.distributions as dist
import torch.nn.functional as F
from functools import reduce
import dgl

class RNNState():
    """
        A class that contains information about the states for RNN computation.
    """
    def __init__(self, particle_state):
        """
            Constructor for PF_State.
                Args:
                    particle_state : State of particles used for particle filtering

        """
        self.particle_state = particle_state

    def detach(self):
        """
            Method to detach (stopping gradient flow) of the particle states and particle log weights.
        """
        return RNNState(
            self.particle_state.detach()
        )

    def cuda(self):
        """
            Method to move particle states & weights to cuda if using GPU.
        """
        return RNNState(
            self.particle_state.cuda()
        )

class GPLTypeInferenceModel(RNNTypeModel):
    def __init__(self,
                 action_space,
                 nr_inputs,
                 cnn_channels,
                 s_dim,
                 theta_dim,
                 gnn_act_pred_dims,
                 gnn_state_hid_dims,
                 state_hid_dims,
                 batch_size,
                 num_agents,
                 device,
                 separate_types=False,
                 encoder_batchnorm=False
                 ):
        super().__init__(action_space, encoding_dimension=theta_dim)
        #self.init_function = init_function
        self.batch_size = batch_size
        self.s_dim = s_dim
        self.theta_dim = theta_dim
        self.device = device
        self.num_agents = num_agents
        self.action_type = None
        self.action_shape = 0
        self.separate_types = separate_types

        # All encoder/decoders are defined in the encoder_decoder.py file
        def mul(num1, num2):
            return num1 * num2

        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
            self.action_type = "discrete"
        else:
            action_shape = action_space.shape[0]
            self.action_type = "continuous"

        self.action_shape = action_shape

        # Add network that updates the type of the agents in the environment
        self.type_updating_network = GPLTypeUpdateNet(
            nr_inputs, cnn_channels, gnn_state_hid_dims, state_hid_dims, s_dim, theta_dim, separate_types=separate_types,
            encoder_batch_norm=encoder_batchnorm, device=self.device
        ).to(self.device)

        ## Add network that predicts the likelihood of sampled actions from proposal_act_predictor_net
        self.act_predictor_net = GPLActionPredictionNet(
            s_dim, theta_dim,
            gnn_act_pred_dims[0], gnn_act_pred_dims[1], gnn_act_pred_dims[2], gnn_act_pred_dims[3],
            action_shape, self.action_type, device=self.device
        ).to(self.device)

    def create_graph_structure(self, obs):
        """
        This function creates a graph structure that will be used for the GNNs for particle updates.
        """
        obs_size = obs.all_x.size()
        num_graphs = obs_size[1]
        num_agents = obs_size[2]

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

        # Initialize
        if not self.separate_types:
            initial_state = st.State(
                    theta= torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double(),
                    cell1=torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double(),
                    cell2=torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double()
            )
        else:
            initial_state = st.State(
                theta=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell1=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell2=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                theta2=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell12=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell22=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double()
            )

        return initial_state

    def propagate(self, observation, actions, previous_latent_state, other_actions):
        """
        This is where the core of the particle update is happening.

        Args:
            observation, reward: Last observation and reward recieved from all n_e environments
            actions: Action vector (oneHot for discrete actions)
            previous_latent_state: previous latent state of type state.State
            other_actions: Action vector for other agents (oneHot for discrete actions)

        Returns:
            latent_state: New latent state
            others : other tensors for logging purposes

        """

        # Needed for legacy AESMC code
        ae_util.init(observation.is_cuda)

        # Legacy code: We need to pass in a (time) sequence of observations
        # With dim=0 for time
        img_observation = observation.unsqueeze(0)
        actions = actions.unsqueeze(0)

        # Legacy code: All values are wrapped in state.State (which can contain more than one value)
        observation_states = st.State(
            all_x=img_observation.contiguous(),
            all_a=actions.contiguous(),
        )

        # Create graph structures for latent_state
        graph_batch = self.create_graph_structure(observation_states)

        # Get 0th element (first element of time sequence) as input
        current_observation = observation_states.index_elements(0)
        if not self.separate_types:
            updated_type, updated_c1, updated_c2 = self.type_updating_network(current_observation, previous_latent_state)
        else:
            updated_type, updated_c1, updated_c2, updated_type2, updated_c12, updated_c22= self.type_updating_network(
                current_observation, previous_latent_state
            )
        # Update current latent state
        current_latent_state = st.State()

        # Set updated type vectors as type vectors of the current particle
        setattr(current_latent_state, "theta", updated_type)
        setattr(current_latent_state, "cell1", updated_c1)
        setattr(current_latent_state, "cell2", updated_c2)
        if self.separate_types:
            setattr(current_latent_state, "theta2", updated_type2)
            setattr(current_latent_state, "cell12", updated_c12)
            setattr(current_latent_state, "cell22", updated_c22)

        # Get the likelihood of sampled teammate actions based on action prediction network
        predictor_action_parameters = self.act_predictor_net(
            graph_batch, current_latent_state
        )

        # Codes to compute action reconstruction loss for all agents
        all_seen_actions = other_actions
        valid_seen_actions = (all_seen_actions.sum(dim=-1) != 0)
        all_seen_valid_actions = all_seen_actions[valid_seen_actions]

        seen_action_distribution = None
        if not all_seen_valid_actions.nelement() == 0:
            if self.action_type == "continuous":
                seen_action_distribution = dist.Normal(
                    predictor_action_parameters[0][:, 1:, :][valid_seen_actions],
                    predictor_action_parameters[1][:, 1:, :][valid_seen_actions]
                )
            else:
                seen_action_distribution = dist.OneHotCategorical(
                    logits=predictor_action_parameters[:, 1:, :][valid_seen_actions]
                )

        # Safeguard against empty tensors.
        action_reconstruction_log_prob = 0.0
        if not all_seen_valid_actions.nelement() == 0:
            seen_action_log_probs = seen_action_distribution.log_prob(all_seen_valid_actions)
            if self.action_type == "discrete":
                seen_action_log_probs = seen_action_log_probs.unsqueeze(-1)
            action_reconstruction_log_prob = seen_action_log_probs.mean()

        # Initialize dictionaries to store other values for logging
        others = {}
        others["action_reconstruction_log_prob"] = action_reconstruction_log_prob
        others["graph"] = graph_batch

        return current_latent_state, \
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

class GPLDecisionMakingModel(nn.Module):
    def __init__(self,
                 action_space,
                 s_dim,
                 theta_dim,
                 device,
                 mid_pair,
                 mid_nodes,
                 mid_pair_out=8,
                 with_sampling_inputs=False
                 ):
        super(GPLDecisionMakingModel,self).__init__()
        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
            self.action_type = "discrete"
        else:
            action_shape = action_space.shape[0]
            self.action_type = "continuous"

        self.device = device
        self.with_sampling_inputs = with_sampling_inputs
        self.cg = JointUtilLayer(s_dim+theta_dim,mid_pair, mid_nodes, action_shape, mid_pair_out, device=self.device).to(self.device)

    def cg_input_preprocess(self, graph_batch, latent_states):
        # Get theta components of latent states and use it as input to CG
        theta = None
        if self.with_sampling_inputs:
            theta = latent_states.theta_sample
        else:
            theta = latent_states.theta
        num_agents = theta.size()[-2]
        node_representations = theta
        zero_node_representations = node_representations[:, 0, :].unsqueeze(1).repeat([1, num_agents, 1])
        zero_edge_representations = node_representations[:, 0, :].unsqueeze(1).repeat(
            [1, num_agents*(num_agents-1), 1]
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

        return final_node_representations, final_edge_representation, final_edge_representation_flipped

    def cg_joint_actions_preprocess(self, latent_states, all_actions):

        all_particle_actions = all_actions.view(-1, all_actions.size()[-1])

        # Replace sampled action with real agent action in case agent is visible in obs
        valid_indices = (all_particle_actions.sum(dim=-1) != 0)
        joint_acts = torch.zeros_like(valid_indices).long().to(self.device)
        joint_acts[valid_indices] = torch.argmax(all_particle_actions[valid_indices], dim=-1)

        return joint_acts.tolist(), valid_indices.unsqueeze(-1).double().view(-1,1)


    def forward(
            self, graph_batch, latent_states, existence, mode="train", action_dist = None, all_actions=None
    ):

        # Resize existence
        existence = existence.view(-1,1)

        # Get latent state after filtering in inference mode
        node_probability = action_dist.probs.detach()

        # Preprocess CG input
        node, edge, edge_flipped = self.cg_input_preprocess(graph_batch, latent_states)

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

class ParticledGPLDecisionMakingModel(nn.Module):
    def __init__(self,
                 action_space,
                 s_dim,
                 theta_dim,
                 device,
                 mid_pair,
                 mid_nodes,
                 mid_pair_out=8,
                 with_rnn=False,
                 h_dim=50,
                 ):
        super(ParticledGPLDecisionMakingModel,self).__init__()
        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
            self.action_type = "discrete"
        else:
            action_shape = action_space.shape[0]
            self.action_type = "continuous"

        self.device = device
        self.with_rnn = with_rnn
        self.h_dim = h_dim
        self.cg = JointUtilLayer(
            s_dim+theta_dim, mid_pair, mid_nodes, action_shape, mid_pair_out, device=self.device
        ).to(self.device)

    def cg_input_preprocess(self, graph_batch, latent_states):
        # Get i, s, theta components of latent states and use it as input to CG
        theta = latent_states.theta_many_samples
        num_agents = theta.size()[-2]
        node_representations = theta

        zero_node_representations = node_representations[:, :, 0, :].unsqueeze(2).repeat([1, 1, num_agents, 1])
        zero_edge_representations = node_representations[:, :, 0, :].unsqueeze(2).repeat(
            [1, 1, num_agents*(num_agents-1), 1]
        )

        final_node_representations = torch.cat([zero_node_representations, node_representations], dim=-1)
        final_node_representations = final_node_representations.view(-1, final_node_representations.size()[-1])

        edges = graph_batch.edges()
        edge_zero_node_data = zero_edge_representations.view(-1, zero_edge_representations.size()[-1])
        resized_node_data = node_representations.contiguous().view(-1, node_representations.size()[-1])

        edge_representation_src = resized_node_data[edges[0]]
        edge_representation_dst = resized_node_data[edges[1]]

        final_edge_representation = torch.cat([
            edge_zero_node_data, edge_representation_src, edge_representation_dst
        ], dim=-1)
        final_edge_representation_flipped = torch.cat([
            edge_zero_node_data, edge_representation_dst, edge_representation_src
        ], dim=-1)

        return final_node_representations, final_edge_representation, final_edge_representation_flipped

    def cg_joint_actions_preprocess(self, latent_states, all_actions):

        num_particles = latent_states.theta_many_samples.size()[1]
        all_particle_actions = all_actions.unsqueeze(1).repeat(1, num_particles, 1, 1)
        all_particle_actions = all_particle_actions.view(-1, all_particle_actions.size()[-1])

        # Replace sampled action with real agent action in case agent is visible in obs
        valid_indices = (all_particle_actions.sum(dim=-1) != 0)
        joint_acts = torch.zeros_like(valid_indices).long().to(self.device)
        joint_acts[valid_indices] = torch.argmax(all_particle_actions[valid_indices], dim=-1)

        return joint_acts.tolist(), valid_indices.unsqueeze(-1).double()


    def forward(
            self, graph_batch, latent_states, existence, mode="train", action_dist = None, all_actions=None
    ):

        # Resize existence
        existence = existence.view(-1, 1)

        # Get latent state after filtering in inference mode
        node_probability = action_dist.probs.detach()

        # Preprocess CG input
        node, edge, edge_flipped = self.cg_input_preprocess(graph_batch, latent_states)

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

        num_batch, num_particles = latent_states.theta_many_samples.size()[0], latent_states.theta_many_samples.size()[1]
        return q_vals.view(num_batch, num_particles, -1)


class GPLActionPredictionNet(nn.Module):
    """
        A class that predicts the parameters of teammates' action distributions given their type vectors,
        inferred state, and existence at previous timestep.
    """
    def __init__(
            self, s_dim, h_dim, mid_dim, gnn_hdim1, gnn_hdim2, gnn_out_dim, act_dim, action_type, device, separate_types=False, with_sampling_inputs=False
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

        self.s_dim = s_dim
        self.h_dim = h_dim
        self.inp_processor = nn.Sequential(
            nn.Linear(s_dim + h_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim)
        )

        self.device = device

        self.gnn = RFMBlock(
            mid_dim, 0, 0, gnn_hdim1, gnn_hdim2, gnn_out_dim
        )
        self.action_type = action_type
        self.separate_types = separate_types
        self.with_sampling_inputs = with_sampling_inputs

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

        input_h = None
        if not self.with_sampling_inputs:
            input_h = previous_latent_state.theta
            if self.separate_types:
                input_h = previous_latent_state.theta2
        else:
            input_h = previous_latent_state.theta_sample
            if self.separate_types:
                input_h = previous_latent_state.theta2_sample

        h_sizes = input_h.size()
        batch_size, num_agents, h_dim = h_sizes[0], h_sizes[1], h_sizes[-1]

        input = input_h.view(-1, self.h_dim + self.s_dim)
        n_inp = self.inp_processor(input)

        _, n_out, _ = self.gnn(
            graph,
            torch.zeros(graph.number_of_edges(),0).to(self.device).double(),
            n_inp,
            torch.zeros(batch_size,0).to(self.device).double()
        )

        if self.action_type == "continuous":
            act_mean, act_var = self.mean_act_net(n_out), self.var_act_net(n_out)
            return act_mean.view(batch_size, num_agents, -1), \
                   act_var.view(batch_size, num_agents, -1)
        else:
            act_logits = self.logit_net(n_out).view(batch_size, num_agents, -1)
            return act_logits

class GPLTypeUpdateNet(nn.Module):
    """
        A class that deterministically updates the latent type of the agents conditioned on their
        new state vector and previous action & type vector.
    """

    def __init__(
            self, o_dim, o_layer_dims, obs_hidden_dims, s_projection_size, s_dim, h_dim, separate_types=False, encoder_batch_norm=False, device=None
    ):
        """
            Args:
                o_enc_dim : The length of observation encodings in the current approach.
                action_encoding : The length of action encodings in the current approach.
                s_dim : The length of state vectors in a particle.
                h_dim : The length of agent type vectors in a particle.
        """
        super().__init__()

        self.encoder_batch_norm = encoder_batch_norm
        self.phi_x = Encoder(
            o_dim,
            o_layer_dims,
            batch_norm=encoder_batch_norm
        )

        def mul(num1, num2):
            return num1 * num2

        self.output_dimension = [o_layer_dims[-1]]
        self.cnn_output_number = reduce(mul, self.output_dimension, 1)

        self.s_updating_net = nn.Sequential(
            nn.Linear(self.cnn_output_number, obs_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(obs_hidden_dims[0], obs_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(obs_hidden_dims[1], obs_hidden_dims[2]),
            nn.Linear(obs_hidden_dims[2], s_projection_size),
            nn.ReLU()
        )
        self.rnn_input_dim = s_projection_size
        self.type_rnn = nn.LSTM(s_projection_size, s_dim + h_dim, batch_first=False)

        self.separate_types = separate_types
        if self.separate_types:
            self.phi_x2 = Encoder(
                o_dim,
                o_layer_dims,
                batch_norm=encoder_batch_norm
            )

            self.s_updating_net2 = nn.Sequential(
                nn.Linear(self.cnn_output_number, obs_hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(obs_hidden_dims[0], obs_hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(obs_hidden_dims[1], obs_hidden_dims[2]),
                nn.Linear(obs_hidden_dims[2], s_projection_size),
                nn.ReLU()
            )
            self.rnn_input_dim2 = s_projection_size
            self.type_rnn2 = nn.LSTM(s_projection_size, s_dim + h_dim, batch_first=False)

    def forward(self, observation_states, previous_latent_state):
        """
            Outputs the updated type vector of all agents given the
            observed observation, inferred states, previous action and previous type of all agents

        Inputs:
            - Input graph structure for GNN
            - observations containing at least
                `all_x`     of dimensions [batch, num_agents, obs_encoding_dim]
            - previous_latent_state containing at least
                `cell1`     of dimensions [batch, num_agents, s_dim+h_dim]
                `cell2`     of dimensions [batch, num_agents, s_dim+h_dim]
        """

        input_c1 = previous_latent_state.cell1
        input_c2 = previous_latent_state.cell2
        input_obs = observation_states.all_x
        cur_flag = (input_obs[:, :, 1] == 1).double().unsqueeze(-1)

        obs_sizes = input_obs.size()
        batch_size, num_agent, obs_dim = obs_sizes[0], obs_sizes[1], obs_sizes[2]

        all_phi_x = self.phi_x(
            input_obs.view(-1, obs_dim)  # Collapse particles
        )  # Flatten CNN output

        input = all_phi_x.view(-1, self.cnn_output_number)
        encoded_input = self.s_updating_net(input)

        n_out = encoded_input.view(-1, self.rnn_input_dim).unsqueeze(0)
        input_c1 = input_c1.view(-1, input_c1.size(-1)).unsqueeze(0)
        input_c2 = input_c2.view(-1, input_c2.size(-1)).unsqueeze(0)

        updated_type, (new_c1, new_c2) = self.type_rnn(n_out, (input_c1, input_c2))

        res = updated_type.view(batch_size, num_agent, -1) * cur_flag
        new_c1 = new_c1.view(batch_size, num_agent, -1) * cur_flag
        new_c2 = new_c2.view(batch_size, num_agent, -1) * cur_flag

        if not self.separate_types:
            return res, new_c1, new_c2

        input_c12 = previous_latent_state.cell12
        input_c22 = previous_latent_state.cell22

        all_phi_x2 = self.phi_x2(
            input_obs.view(-1, obs_dim)  # Collapse particles
        )

        input2 = all_phi_x2.view(-1, self.cnn_output_number)
        encoded_input2 = self.s_updating_net2(input2)

        n_out2 = encoded_input2.view(-1, self.rnn_input_dim).unsqueeze(0)
        input_c12 = input_c12.view(-1, input_c12.size(-1)).unsqueeze(0)
        input_c22 = input_c22.view(-1, input_c22.size(-1)).unsqueeze(0)

        updated_type2, (new_c12, new_c22) = self.type_rnn2(n_out2, (input_c12, input_c22))
        res2 = updated_type2.view(batch_size, num_agent, -1) * cur_flag
        new_c12 = new_c12.view(batch_size, num_agent, -1) * cur_flag
        new_c22 = new_c22.view(batch_size, num_agent, -1) * cur_flag

        return res, new_c1, new_c2, res2, new_c12, new_c22

class GPLTypeUpdateNetWithRepStochasticity(nn.Module):
    """
        A class that deterministically updates the latent type of the agents conditioned on their
        new state vector and previous action & type vector.
    """

    def __init__(
            self, o_dim, o_layer_dims, obs_hidden_dims, s_projection_size, s_dim, h_dim, separate_types=False, encoder_batch_norm=False, device=None
    ):
        """
            Args:
                o_enc_dim : The length of observation encodings in the current approach.
                action_encoding : The length of action encodings in the current approach.
                s_dim : The length of state vectors in a particle.
                h_dim : The length of agent type vectors in a particle.
        """
        super().__init__()

        self.encoder_batch_norm = encoder_batch_norm
        self.phi_x = Encoder(
            o_dim,
            o_layer_dims,
            batch_norm=encoder_batch_norm
        )

        self.device = device

        def mul(num1, num2):
            return num1 * num2

        self.output_dimension = [o_layer_dims[-1]]
        self.cnn_output_number = reduce(mul, self.output_dimension, 1)

        self.s_updating_net = nn.Sequential(
            nn.Linear(self.cnn_output_number, obs_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(obs_hidden_dims[0], obs_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(obs_hidden_dims[1], obs_hidden_dims[2]),
            nn.Linear(obs_hidden_dims[2], s_projection_size),
            nn.ReLU()
        )
        self.rnn_input_dim = s_projection_size
        self.type_rnn_mean = nn.LSTM(s_projection_size, s_dim + h_dim, batch_first=False)
        self.type_rnn_var = nn.LSTM(s_projection_size, s_dim + h_dim, batch_first=False)

        self.separate_types = separate_types
        if self.separate_types:
            self.phi_x2 = Encoder(
                o_dim,
                o_layer_dims,
                batch_norm=encoder_batch_norm
            )

            self.s_updating_net2 = nn.Sequential(
                nn.Linear(self.cnn_output_number, obs_hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(obs_hidden_dims[0], obs_hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(obs_hidden_dims[1], obs_hidden_dims[2]),
                nn.Linear(obs_hidden_dims[2], s_projection_size),
                nn.ReLU()
            )
            self.rnn_input_dim2 = s_projection_size
            self.type_rnn2_mean = nn.LSTM(s_projection_size, s_dim + h_dim, batch_first=False)
            self.type_rnn2_var = nn.LSTM(s_projection_size, s_dim + h_dim, batch_first=False)

    def forward(self, observation_states, previous_latent_state):
        """
            Outputs the updated type vector of all agents given the
            observed observation, inferred states, previous action and previous type of all agents

        Inputs:
            - Input graph structure for GNN
            - observations containing at least
                `all_x`     of dimensions [batch, num_agents, obs_encoding_dim]
            - previous_latent_state containing at least
                `cell1`     of dimensions [batch, num_agents, s_dim+h_dim]
                `cell2`     of dimensions [batch, num_agents, s_dim+h_dim]
        """

        input_c1_mean = previous_latent_state.cell1
        input_c2_mean = previous_latent_state.cell2
        input_c1_var = previous_latent_state.cell1_var
        input_c2_var = previous_latent_state.cell2_var
        input_obs = observation_states.all_x

        obs_sizes = input_obs.size()
        batch_size, num_agent, obs_dim = obs_sizes[0], obs_sizes[1], obs_sizes[2]

        all_phi_x = self.phi_x(
            input_obs.view(-1, obs_dim)  # Collapse particles
        )  # Flatten CNN output

        input = all_phi_x.view(-1, self.cnn_output_number)
        encoded_input = self.s_updating_net(input)

        n_out = encoded_input.view(-1, self.rnn_input_dim).unsqueeze(0)
        input_c1_mean = input_c1_mean.view(-1, input_c1_mean.size(-1)).unsqueeze(0)
        input_c2_mean = input_c2_mean.view(-1, input_c2_mean.size(-1)).unsqueeze(0)
        input_c1_var = input_c1_var.view(-1, input_c1_var.size(-1)).unsqueeze(0)
        input_c2_var = input_c2_var.view(-1, input_c2_var.size(-1)).unsqueeze(0)

        updated_type_mean, (new_c1_mean, new_c2_mean) = self.type_rnn_mean(n_out, (input_c1_mean, input_c2_mean))
        updated_type_var, (new_c1_var, new_c2_var) = self.type_rnn_var(n_out, (input_c1_var, input_c2_var))

        res = updated_type_mean.view(batch_size, num_agent, -1)
        res_var = updated_type_var.view(batch_size, num_agent, -1)
        new_c1 = new_c1_mean.view(batch_size, num_agent, -1)
        new_c2 = new_c2_mean.view(batch_size, num_agent, -1)
        new_c1_var = new_c1_var.view(batch_size, num_agent, -1)
        new_c2_var = new_c2_var.view(batch_size, num_agent, -1)

        if not self.separate_types:
            return res, new_c1, new_c2,res_var, new_c1_var, new_c2_var, dist.Normal(res, F.softplus(res_var))

        input_c12_mean = previous_latent_state.cell12
        input_c22_mean = previous_latent_state.cell22
        input_c12_var = previous_latent_state.cell12_var
        input_c22_var = previous_latent_state.cell22_var

        all_phi_x2 = self.phi_x2(
            input_obs.view(-1, obs_dim)  # Collapse particles
        )

        input2 = all_phi_x2.view(-1, self.cnn_output_number)
        encoded_input2 = self.s_updating_net2(input2)

        n_out2 = encoded_input2.view(-1, self.rnn_input_dim).unsqueeze(0)
        input_c12_mean = input_c12_mean.view(-1, input_c12_mean.size(-1)).unsqueeze(0)
        input_c22_mean = input_c22_mean.view(-1, input_c22_mean.size(-1)).unsqueeze(0)
        input_c12_var = input_c12_var.view(-1, input_c12_var.size(-1)).unsqueeze(0)
        input_c22_var = input_c22_var.view(-1, input_c22_var.size(-1)).unsqueeze(0)

        updated_type2_mean, (new_c12_mean, new_c22_mean) = self.type_rnn2_mean(n_out2, (input_c12_mean, input_c22_mean))
        updated_type2_var, (new_c12_var, new_c22_var) = self.type_rnn2_var(n_out2, (input_c12_var, input_c22_var))

        res2_mean = updated_type2_mean.view(batch_size, num_agent, -1)
        res2_var = updated_type2_var.view(batch_size, num_agent, -1)
        new_c12_mean = new_c12_mean.view(batch_size, num_agent, -1)
        new_c22_mean = new_c22_mean.view(batch_size, num_agent, -1)
        new_c12_var = new_c12_var.view(batch_size, num_agent, -1)
        new_c22_var = new_c22_var.view(batch_size, num_agent, -1)

        return res, new_c1, new_c2,res_var, new_c1_var, new_c2_var, dist.Normal(res, F.softplus(res_var)), res2_mean, new_c12_mean, new_c22_mean, res2_var, new_c12_var, new_c22_var, dist.Normal(res2_mean, F.softplus(res2_var))

class GPLTypeUpdateNetWithOpennessModelling(nn.Module):
    """
        A class that deterministically updates the latent type of the agents conditioned on their
        new state vector and previous action & type vector.
    """

    def __init__(
            self, o_dim, o_layer_dims, obs_hidden_dims, s_projection_size, s_dim, h_dim, separate_types=False, encoder_batch_norm=False, device=None
    ):
        """
            Args:
                o_enc_dim : The length of observation encodings in the current approach.
                action_encoding : The length of action encodings in the current approach.
                s_dim : The length of state vectors in a particle.
                h_dim : The length of agent type vectors in a particle.
        """
        super().__init__()

        self.encoder_batch_norm = encoder_batch_norm
        self.phi_x = Encoder(
            o_dim,
            o_layer_dims,
            batch_norm=encoder_batch_norm
        )

        def mul(num1, num2):
            return num1 * num2

        self.device = device

        self.output_dimension = [o_layer_dims[-1]]
        self.cnn_output_number = reduce(mul, self.output_dimension, 1)

        self.s_updating_net = nn.Sequential(
            nn.Linear(self.cnn_output_number, obs_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(obs_hidden_dims[0], obs_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(obs_hidden_dims[1], obs_hidden_dims[2]),
            nn.Linear(obs_hidden_dims[2], s_projection_size),
            nn.ReLU()
        )
        self.rnn_input_dim = s_projection_size
        self.type_rnn = nn.LSTM(s_projection_size, s_dim + h_dim, batch_first=False)

        self.separate_types = separate_types

        self.edge_net = nn.Linear(2 * (s_dim+h_dim) + 1, h_dim)
        self.logit_net = nn.Sequential(
            nn.Linear((s_dim+h_dim) + h_dim + 1, 1)
        )

        if self.separate_types:
            self.phi_x2 = Encoder(
                o_dim,
                o_layer_dims,
                batch_norm=encoder_batch_norm
            )

            self.s_updating_net2 = nn.Sequential(
                nn.Linear(self.cnn_output_number, obs_hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(obs_hidden_dims[0], obs_hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(obs_hidden_dims[1], obs_hidden_dims[2]),
                nn.Linear(obs_hidden_dims[2], s_projection_size),
                nn.ReLU()
            )
            self.rnn_input_dim2 = s_projection_size
            self.type_rnn2 = nn.LSTM(s_projection_size, s_dim + h_dim, batch_first=False)

    def forward(self, observation_states, previous_latent_state, temperature=1.0, eval=False):
        """
            Outputs the updated type vector of all agents given the
            observed observation, inferred states, previous action and previous type of all agents

        Inputs:
            - Input graph structure for GNN
            - observations containing at least
                `all_x`     of dimensions [batch, num_agents, obs_encoding_dim]
            - previous_latent_state containing at least
                `cell1`     of dimensions [batch, num_agents, s_dim+h_dim]
                `cell2`     of dimensions [batch, num_agents, s_dim+h_dim]
        """

        input_c1 = previous_latent_state.cell1
        input_c2 = previous_latent_state.cell2
        prev_flag = previous_latent_state.i
        input_obs = observation_states.all_x
        if not eval:
            # TODO Add state information as input
            sampled_i = observation_states.all_state_existence

        obs_sizes = input_obs.size()
        batch_size, num_agent, obs_dim = obs_sizes[0], obs_sizes[1], obs_sizes[2]

        all_phi_x = self.phi_x(
            input_obs.view(-1, obs_dim)  # Collapse particles
        )  # Flatten CNN output

        input = all_phi_x.view(-1, self.cnn_output_number)
        encoded_input = self.s_updating_net(input)

        n_out = encoded_input.view(-1, self.rnn_input_dim).unsqueeze(0)
        input_c1 = input_c1.view(-1, input_c1.size(-1)).unsqueeze(0)
        input_c2 = input_c2.view(-1, input_c2.size(-1)).unsqueeze(0)

        updated_type, (new_c1, new_c2) = self.type_rnn(n_out, (input_c1, input_c2))

        # New addition
        reshaped_n_out = updated_type.squeeze(0).view(batch_size, num_agent, -1)

        # Assume agent index 0 for the learner
        # Also assume learner always exists
        agent_flags = [torch.ones(batch_size, 1, 1).to(self.device).double()]
        node_logs = [torch.zeros(batch_size, 1, 1).to(self.device).double()]
        node_logits = [torch.zeros(batch_size, 1, 1).to(self.device).double()]
        node_entropy = [torch.zeros(batch_size, 1, 1).to(self.device).double()]

        if not eval:
            agent_state_flags = [sampled_i[:, 0, :].unsqueeze(-2)]
            node_state_logs = [torch.zeros(batch_size, 1, 1).to(self.device).double()]

        # Implement autoregressive reasoning
        for agent_offset in range(1, num_agent):
            # Get node representation of other agents that have been added to particle
            other_agent_data = reshaped_n_out[:, :agent_offset, :]

            # Get node representation of the agent that is to be added to the particle
            added_agent_data = reshaped_n_out[:, agent_offset, :]
            added_agent_prev_flag = prev_flag[:, agent_offset, :]

            added_agent_data_reshaped = added_agent_data.unsqueeze(-2).repeat([1, agent_offset, 1])
            concatenated_agent_flags = torch.cat(agent_flags, dim=-2)
            if not eval:
                concatenated_state_agent_flags = torch.cat(agent_state_flags, dim=-2)

            # Autoregressive input will be representation of agents that already exist
            # Representation of agent to be added to the environment
            # and data on the agents' existence at the previous timestep
            edge_data = torch.cat([other_agent_data, added_agent_data_reshaped, concatenated_agent_flags], dim=-1)
            if not eval:
                state_edge_data = torch.cat(
                    [other_agent_data, added_agent_data_reshaped, concatenated_state_agent_flags], dim=-1)

            # Sum message from all existing nodes for deciding whether new agent is added or not
            edge_msg = self.edge_net(edge_data).sum(dim=-2)
            if not eval:
                state_edge_msg = self.edge_net(state_edge_data).sum(dim=-2)

            # Compute p(add new agent | prev agents data)
            input_logit_data = torch.cat([added_agent_data, edge_msg, added_agent_prev_flag], dim=-1)
            added_probs_logits = self.logit_net(input_logit_data).view(batch_size, 1, -1)
            if not eval:
                input_logit_state_data = torch.cat([added_agent_data, state_edge_msg, added_agent_prev_flag], dim=-1)
                added_probs_states_logits = self.logit_net(input_logit_state_data).view(batch_size, 1, -1)

            added_probs = F.sigmoid(added_probs_logits / temperature)
            node_dist = dist.Bernoulli(probs=added_probs)
            node_sample_t = node_dist.sample()
            node_logs_t = node_dist.log_prob(node_sample_t)
            node_dist_entropy = node_dist.entropy()

            if not eval:
                added_state_probs = F.sigmoid(added_probs_states_logits / temperature)
                node_state_dist = dist.Bernoulli(probs=added_state_probs)
                node_state_logs_t = node_state_dist.log_prob(sampled_i[:, agent_offset, :].unsqueeze(-2))

            # Output sample existence (0/1) and log prob of sample
            new_flags = node_sample_t.view(batch_size, 1, -1)
            if not eval:
                new_state_flags = sampled_i[:, agent_offset, :].unsqueeze(-2)
            new_logs = node_logs_t.view(batch_size, 1, -1)
            new_entropy = node_dist_entropy.view(batch_size, 1, -1)

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
        # New addition end

        res = updated_type.view(batch_size, num_agent, -1) * updated_agent_flags
        new_c1 = new_c1.view(batch_size, num_agent, -1) * updated_agent_flags
        new_c2 = new_c2.view(batch_size, num_agent, -1) * updated_agent_flags

        # print("agent_existence_logs", agent_existence_logs, agent_state_existence_logs)

        if not self.separate_types:
            return res, new_c1, new_c2, updated_agent_flags, agent_existence_logs, agent_state_existence_logs

        input_c12 = previous_latent_state.cell12
        input_c22 = previous_latent_state.cell22

        all_phi_x2 = self.phi_x2(
            input_obs.view(-1, obs_dim)  # Collapse particles
        )

        input2 = all_phi_x2.view(-1, self.cnn_output_number)
        encoded_input2 = self.s_updating_net2(input2)

        n_out2 = encoded_input2.view(-1, self.rnn_input_dim).unsqueeze(0)
        input_c12 = input_c12.view(-1, input_c12.size(-1)).unsqueeze(0)
        input_c22 = input_c22.view(-1, input_c22.size(-1)).unsqueeze(0)

        updated_type2, (new_c12, new_c22) = self.type_rnn2(n_out2, (input_c12, input_c22))
        res2 = updated_type2.view(batch_size, num_agent, -1) * updated_agent_flags
        new_c12 = new_c12.view(batch_size, num_agent, -1) * updated_agent_flags
        new_c22 = new_c22.view(batch_size, num_agent, -1) * updated_agent_flags

        return res, new_c1, new_c2, res2, new_c12, new_c22, updated_agent_flags, agent_existence_logs, agent_state_existence_logs

class GPLTypeInferenceModelObsRecons(RNNTypeModel):
    def __init__(self,
                 action_space,
                 nr_inputs,
                 agent_inputs,
                 u_inputs,
                 cnn_channels,
                 s_dim,
                 theta_dim,
                 gnn_act_pred_dims,
                 gnn_state_hid_dims,
                 gnn_decoder_hid_dims,
                 state_hid_dims,
                 batch_size,
                 num_agents,
                 device,
                 separate_types=False,
                 encoder_batchnorm=False,
                 with_global_features=True
                 ):
        super().__init__(action_space, encoding_dimension=theta_dim)
        #self.init_function = init_function
        self.batch_size = batch_size
        self.s_dim = s_dim
        self.agent_inputs = agent_inputs
        self.u_inputs = u_inputs
        self.theta_dim = theta_dim
        self.device = device
        self.num_agents = num_agents
        self.action_type = None
        self.action_shape = 0
        self.separate_types = separate_types
        self.with_global_feature = with_global_features

        # All encoder/decoders are defined in the encoder_decoder.py file
        def mul(num1, num2):
            return num1 * num2

        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
            self.action_type = "discrete"
        else:
            action_shape = action_space.shape[0]
            self.action_type = "continuous"

        self.action_shape = action_shape

        # Add network that updates the type of the agents in the environment
        self.type_updating_network = GPLTypeUpdateNet(
            nr_inputs, cnn_channels, gnn_state_hid_dims, state_hid_dims, s_dim, theta_dim, separate_types=separate_types,
            encoder_batch_norm=encoder_batchnorm
        ).to(self.device)
        self.decoder_network = ObsDecoderNet(
            s_dim+theta_dim,
            gnn_decoder_hid_dims[0],
            gnn_decoder_hid_dims[1],
            gnn_decoder_hid_dims[2],
            cnn_channels, agent_inputs - 1,
            u_inputs, device=self.device,
            with_global_feature=with_global_features
        ).to(self.device)

        ## Add network that predicts the likelihood of sampled actions from proposal_act_predictor_net
        self.act_predictor_net = GPLActionPredictionNet(
            s_dim, theta_dim,
            gnn_act_pred_dims[0], gnn_act_pred_dims[1], gnn_act_pred_dims[2], gnn_act_pred_dims[3],
            action_shape, self.action_type, device=self.device
        ).to(self.device)

    def create_graph_structure(self, obs):
        """
        This function creates a graph structure that will be used for the GNNs for particle updates.
        """
        obs_size = obs.all_x.size()
        num_graphs = obs_size[1]
        num_agents = obs_size[2]

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

        # Initialize
        if not self.separate_types:
            initial_state = st.State(
                    theta= torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double(),
                    cell1=torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double(),
                    cell2=torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double()
            )

        else:
            initial_state = st.State(
                theta=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell1=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell2=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                theta2=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell12=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell22=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double()
            )

        return initial_state

    def propagate(self, observation, actions, previous_latent_state, other_actions):
        """
        This is where the core of the particle update is happening.

        Args:
            observation, reward: Last observation and reward recieved from all n_e environments
            actions: Action vector (oneHot for discrete actions)
            previous_latent_state: previous latent state of type state.State
            other_actions: Action vector for other agents (oneHot for discrete actions)

        Returns:
            latent_state: New latent state
            others : other tensors for logging purposes

        """

        # Needed for legacy AESMC code
        ae_util.init(observation.is_cuda)

        # Legacy code: We need to pass in a (time) sequence of observations
        # With dim=0 for time
        img_observation = observation.unsqueeze(0)
        actions = actions.unsqueeze(0)

        # Legacy code: All values are wrapped in state.State (which can contain more than one value)
        observation_states = st.State(
            all_x=img_observation.contiguous(),
            all_a=actions.contiguous(),
        )

        # Create graph structures for latent_state
        graph_batch = self.create_graph_structure(observation_states)

        # Get 0th element (first element of time sequence) as input
        current_observation = observation_states.index_elements(0)
        if not self.separate_types:
            updated_type, updated_c1, updated_c2 = self.type_updating_network(current_observation, previous_latent_state)
        else:
            updated_type, updated_c1, updated_c2, updated_type2, updated_c12, updated_c22= self.type_updating_network(
                current_observation, previous_latent_state
            )
        # Update current latent state
        current_latent_state = st.State()

        # Set updated type vectors as type vectors of the current particle
        setattr(current_latent_state, "theta", updated_type)
        setattr(current_latent_state, "cell1", updated_c1)
        setattr(current_latent_state, "cell2", updated_c2)
        if self.separate_types:
            setattr(current_latent_state, "theta2", updated_type2)
            setattr(current_latent_state, "cell12", updated_c12)
            setattr(current_latent_state, "cell22", updated_c22)

        # Get agent visibility distribution and observation vectors from current vector
        agent_visibility_dist, obs_feature_dist, u_feature_dist = self.decoder_network(
            current_latent_state
        )

        # Compute likelihood of observed observation here
        agent_vis = current_observation.all_x[:, :, 1].unsqueeze(-1)
        obs_vis_log_prob = agent_visibility_dist.log_prob(agent_vis)

        all_obs = torch.cat(
            [current_observation.all_x[:, :, :1], current_observation.all_x[:, :, 2:]], dim=-1
        )

        agent_inputs = all_obs[:, :, :self.agent_inputs - 1]
        u_obs = all_obs[:, 0, self.agent_inputs - 1:]

        a_obs_log_prob = obs_feature_dist.log_prob(
            agent_inputs
        )

        u_obs_log_prob = 0
        if self.with_global_feature:
            u_obs_log_prob = u_feature_dist.log_prob(
                u_obs
            )

        # Get the likelihood of sampled teammate actions based on action prediction network
        predictor_action_parameters = self.act_predictor_net(
            graph_batch, current_latent_state
        )

        # Codes to compute action reconstruction loss for all agents
        all_seen_actions = other_actions
        valid_seen_actions = (all_seen_actions.sum(dim=-1) != 0)
        all_seen_valid_actions = all_seen_actions[valid_seen_actions]

        seen_action_distribution = None
        if not all_seen_valid_actions.nelement() == 0:
            if self.action_type == "continuous":
                seen_action_distribution = dist.Normal(
                    predictor_action_parameters[0][:, 1:, :][valid_seen_actions],
                    predictor_action_parameters[1][:, 1:, :][valid_seen_actions]
                )
            else:
                seen_action_distribution = dist.OneHotCategorical(
                    logits=predictor_action_parameters[:, 1:, :][valid_seen_actions]
                )

        # Safeguard against empty tensors.
        action_reconstruction_log_prob = 0.0
        if not all_seen_valid_actions.nelement() == 0:
            seen_action_log_probs = seen_action_distribution.log_prob(all_seen_valid_actions)
            if self.action_type == "discrete":
                seen_action_log_probs = seen_action_log_probs.unsqueeze(-1)
            action_reconstruction_log_prob = seen_action_log_probs.mean()

        obs_reconstruction_component = obs_vis_log_prob + a_obs_log_prob
        obs_reconstruction_component = obs_reconstruction_component.mean(dim=-1).mean(dim=-1)

        if self.with_global_feature:
            obs_reconstruction_component = obs_reconstruction_component + (u_obs_log_prob.mean(dim=-1))

        # Initialize dictionaries to store other values for logging
        others = {}
        others["action_reconstruction_log_prob"] = action_reconstruction_log_prob
        others["graph"] = graph_batch
        others["agent_obs_visibility_log_prob"] = obs_vis_log_prob
        others["agent_features_log_prob"] = a_obs_log_prob
        others["u_features_log_prob"] = u_obs_log_prob
        others["obs_reconstruction_log_prob"] = obs_reconstruction_component.mean()

        return current_latent_state, \
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

class GPLTypeInferenceModelStateRecons(RNNTypeModelStateRecons):
    def __init__(self,
                 action_space,
                 nr_inputs,
                 agent_inputs,
                 u_inputs,
                 cnn_channels,
                 s_dim,
                 theta_dim,
                 gnn_act_pred_dims,
                 gnn_state_hid_dims,
                 gnn_decoder_hid_dims,
                 state_hid_dims,
                 batch_size,
                 num_agents,
                 device,
                 separate_types=False,
                 encoder_batchnorm=False,
                 with_global_features=True
                 ):
        super().__init__(action_space, encoding_dimension=theta_dim)
        #self.init_function = init_function
        self.batch_size = batch_size
        self.s_dim = s_dim
        self.agent_inputs = agent_inputs
        self.u_inputs = u_inputs
        self.theta_dim = theta_dim
        self.device = device
        self.num_agents = num_agents
        self.action_type = None
        self.action_shape = 0
        self.separate_types = separate_types
        self.with_global_feature = with_global_features

        # All encoder/decoders are defined in the encoder_decoder.py file
        def mul(num1, num2):
            return num1 * num2

        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
            self.action_type = "discrete"
        else:
            action_shape = action_space.shape[0]
            self.action_type = "continuous"

        self.action_shape = action_shape

        # Add network that updates the type of the agents in the environment
        self.type_updating_network = GPLTypeUpdateNetWithOpennessModelling(
            nr_inputs, cnn_channels, gnn_state_hid_dims, state_hid_dims, s_dim, theta_dim, separate_types=separate_types,
            encoder_batch_norm=encoder_batchnorm, device=self.device
        ).to(self.device)
        self.decoder_network = StateDecoderNet(
            s_dim+theta_dim,
            gnn_decoder_hid_dims[0],
            gnn_decoder_hid_dims[1],
            gnn_decoder_hid_dims[2],
            cnn_channels, agent_inputs - 1,
            u_inputs, device=self.device,
            with_global_feature=with_global_features
        ).to(self.device)

        ## Add network that predicts the likelihood of sampled actions from proposal_act_predictor_net
        self.act_predictor_net = GPLActionPredictionNet(
            s_dim, theta_dim,
            gnn_act_pred_dims[0], gnn_act_pred_dims[1], gnn_act_pred_dims[2], gnn_act_pred_dims[3],
            action_shape, self.action_type, device=self.device
        ).to(self.device)

    def create_graph_structure(self, obs):
        """
        This function creates a graph structure that will be used for the GNNs for particle updates.
        """
        obs_size = obs.all_x.size()
        num_graphs = obs_size[1]
        num_agents = obs_size[2]

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

        # Initialize
        if not self.separate_types:
            initial_state = st.State(
                    theta= torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double(),
                    cell1=torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double(),
                    cell2=torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double()
            )
        else:
            initial_state = st.State(
                theta=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell1=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell2=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                theta2=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell12=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell22=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double()
            )

        initial_state.i = torch.zeros(self.batch_size, self.num_agents, 1).to(device).double()
        return initial_state

    def propagate(self, observation, state, actions, previous_latent_state, other_actions, state_agent_existence, eval, analysis_mode=False):
        """
        This is where the core of the particle update is happening.

        Args:
            observation, reward: Last observation and reward recieved from all n_e environments
            actions: Action vector (oneHot for discrete actions)
            previous_latent_state: previous latent state of type state.State
            other_actions: Action vector for other agents (oneHot for discrete actions)

        Returns:
            latent_state: New latent state
            others : other tensors for logging purposes

        """

        # Needed for legacy AESMC code
        ae_util.init(observation.is_cuda)

        # Legacy code: We need to pass in a (time) sequence of observations
        # With dim=0 for time
        img_observation = observation.unsqueeze(0)
        all_state = state.unsqueeze(0)
        actions = actions.unsqueeze(0)
        if not eval:
            state_existence = state_agent_existence.unsqueeze(0)

        # Legacy code: All values are wrapped in state.State (which can contain more than one value)
        observation_states = st.State(
            all_x=img_observation.contiguous(),
            all_x_state=all_state.contiguous(),
            all_a=actions.contiguous(),
        )

        if not eval:
            observation_states.all_state_existence = state_existence.contiguous()

        # Create graph structures for latent_state
        graph_batch = self.create_graph_structure(observation_states)

        # Get 0th element (first element of time sequence) as input
        current_observation = observation_states.index_elements(0)
        if not self.separate_types:
            updated_type, updated_c1, updated_c2, updated_agent_flags, agent_existence_logs, agent_state_existence_logs = self.type_updating_network(current_observation, previous_latent_state, eval=eval)
        else:
            updated_type, updated_c1, updated_c2, updated_type2, updated_c12, updated_c22, updated_agent_flags, agent_existence_logs, agent_state_existence_logs = self.type_updating_network(
                current_observation, previous_latent_state, eval=eval
            )
        # Update current latent state
        current_latent_state = st.State()

        # Set updated type vectors as type vectors of the current particle
        setattr(current_latent_state, "theta", updated_type)
        setattr(current_latent_state, "cell1", updated_c1)
        setattr(current_latent_state, "cell2", updated_c2)
        if self.separate_types:
            setattr(current_latent_state, "theta2", updated_type2)
            setattr(current_latent_state, "cell12", updated_c12)
            setattr(current_latent_state, "cell22", updated_c22)

        setattr(current_latent_state, "i", updated_agent_flags)

        # Get agent visibility distribution and observation vectors from current vector
        obs_feature_dist, u_feature_dist = self.decoder_network(
            current_latent_state
        )

        # Compute likelihood of observed observation here
        agent_vis = current_observation.all_x_state[:, :, 1].unsqueeze(-1)

        all_state = torch.cat(
            [current_observation.all_x_state[:, :, :1], current_observation.all_x_state[:, :, 2:]], dim=-1
        )

        agent_inputs = all_state[:, :, :self.agent_inputs - 1]
        u_obs = all_state[:, 0, self.agent_inputs - 1:]

        a_obs_log_prob = obs_feature_dist.log_prob(
            agent_inputs
        )

        u_obs_log_prob = 0
        if self.with_global_feature:
            u_obs_log_prob = u_feature_dist.log_prob(
                u_obs
            )

        # Get the likelihood of sampled teammate actions based on action prediction network
        predictor_action_parameters = self.act_predictor_net(
            graph_batch, current_latent_state
        )

        # Codes to compute action reconstruction loss for all agents
        all_seen_actions = other_actions
        valid_seen_actions = (all_seen_actions.sum(dim=-1) != 0)
        all_seen_valid_actions = all_seen_actions[valid_seen_actions]

        seen_action_distribution = None
        if not all_seen_valid_actions.nelement() == 0:
            if self.action_type == "continuous":
                seen_action_distribution = dist.Normal(
                    predictor_action_parameters[0][:, 1:, :][valid_seen_actions],
                    predictor_action_parameters[1][:, 1:, :][valid_seen_actions]
                )
            else:
                seen_action_distribution = dist.OneHotCategorical(
                    logits=predictor_action_parameters[:, 1:, :][valid_seen_actions]
                )

        # Safeguard against empty tensors.
        action_reconstruction_log_prob = 0.0
        if not all_seen_valid_actions.nelement() == 0:
            seen_action_log_probs = seen_action_distribution.log_prob(all_seen_valid_actions)
            if self.action_type == "discrete":
                seen_action_log_probs = seen_action_log_probs.unsqueeze(-1)
            action_reconstruction_log_prob = seen_action_log_probs.mean()

        obs_reconstruction_component = a_obs_log_prob
        obs_reconstruction_component = obs_reconstruction_component.mean(dim=-1).mean(dim=-1)

        if self.with_global_feature:
            obs_reconstruction_component = obs_reconstruction_component + (u_obs_log_prob.mean(dim=-1))

        # Initialize dictionaries to store other values for logging
        others = {}
        others["action_reconstruction_log_prob"] = action_reconstruction_log_prob
        others["graph"] = graph_batch
        others["agent_features_log_prob"] = a_obs_log_prob
        others["u_features_log_prob"] = u_obs_log_prob
        others["state_reconstruction_log_prob"] = obs_reconstruction_component.mean()
        if not eval:
            others["agent_existence_log_prob"] = agent_state_existence_logs.mean()

        return current_latent_state, \
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

class GPLTypeInferenceModelStateReconsStochastic(RNNTypeModelStateRecons):
    def __init__(self,
                 action_space,
                 nr_inputs,
                 agent_inputs,
                 u_inputs,
                 cnn_channels,
                 s_dim,
                 theta_dim,
                 gnn_act_pred_dims,
                 gnn_state_hid_dims,
                 gnn_decoder_hid_dims,
                 state_hid_dims,
                 batch_size,
                 num_agents,
                 device,
                 num_particles=10,
                 separate_types=False,
                 encoder_batchnorm=False,
                 with_global_features=True
                 ):
        super().__init__(action_space, encoding_dimension=theta_dim)
        #self.init_function = init_function
        self.batch_size = batch_size
        self.s_dim = s_dim
        self.agent_inputs = agent_inputs
        self.u_inputs = u_inputs
        self.theta_dim = theta_dim
        self.device = device
        self.num_agents = num_agents
        self.action_type = None
        self.action_shape = 0
        self.separate_types = separate_types
        self.with_global_feature = with_global_features
        self.num_particles = num_particles

        # All encoder/decoders are defined in the encoder_decoder.py file
        def mul(num1, num2):
            return num1 * num2

        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
            self.action_type = "discrete"
        else:
            action_shape = action_space.shape[0]
            self.action_type = "continuous"

        self.action_shape = action_shape

        # Add network that updates the type of the agents in the environment
        self.type_updating_network = GPLTypeUpdateNetWithRepStochasticity(
            nr_inputs, cnn_channels, gnn_state_hid_dims, state_hid_dims, s_dim, theta_dim, separate_types=separate_types,
            encoder_batch_norm=encoder_batchnorm, device=self.device
        ).to(self.device)
        self.decoder_network = StateDecoderNetWithOpenness(
            s_dim+theta_dim,
            gnn_decoder_hid_dims[0],
            gnn_decoder_hid_dims[1],
            gnn_decoder_hid_dims[2],
            cnn_channels, agent_inputs - 1,
            u_inputs, device=self.device,
            with_global_feature=with_global_features
        ).to(self.device)

        ## Add network that predicts the likelihood of sampled actions from proposal_act_predictor_net
        self.act_predictor_net = GPLActionPredictionNet(
            s_dim, theta_dim,
            gnn_act_pred_dims[0], gnn_act_pred_dims[1], gnn_act_pred_dims[2], gnn_act_pred_dims[3],
            action_shape, self.action_type, device=self.device, with_sampling_inputs=True
        ).to(self.device)

    def create_graph_structure(self, obs):
        """
        This function creates a graph structure that will be used for the GNNs for particle updates.
        """
        obs_size = obs.all_x.size()
        num_graphs = obs_size[1]
        num_agents = obs_size[2]

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

    def create_graph_structure_particles(self, obs):
        """
        This function creates a graph structure that will be used for the GNNs for particle updates.
        """
        obs_size = obs.all_x.size()
        num_graphs = obs_size[1] * self.num_particles
        num_agents = obs_size[2]

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

        # Initialize
        if not self.separate_types:
            initial_state = st.State(
                    theta= torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double(),
                    theta_var=torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                    ).to(device).double(),
                    cell1=torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double(),
                    cell1_var=torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                    ).to(device).double(),
                    cell2=torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim+self.s_dim
                    ).to(device).double(),
                    cell2_var=torch.zeros(
                        self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                    ).to(device).double(),
                    theta_many_samples=torch.zeros(
                        self.batch_size, self.num_particles, self.num_agents, self.theta_dim + self.s_dim
                    ).to(device).double(),
                    theta_many_samples_likelihood=torch.zeros(
                        self.batch_size, self.num_particles, self.num_agents, self.theta_dim + self.s_dim
                    ).to(device).double()
            )
        else:
            initial_state = st.State(
                theta=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                theta_var=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell1=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell1_var=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell2=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell2_var=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                theta2=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                theta2_var=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell12=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell12_var=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell22=torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                cell22_var = torch.zeros(
                    self.batch_size, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                theta_many_samples=torch.zeros(
                    self.batch_size, self.num_particles, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double(),
                theta_many_samples_likelihood=torch.zeros(
                    self.batch_size, self.num_particles, self.num_agents, self.theta_dim + self.s_dim
                ).to(device).double()
            )

        return initial_state

    def propagate(self, observation, state, actions, previous_latent_state, other_actions, state_agent_existence, eval=False, analysis_mode=False):
        """
        This is where the core of the particle update is happening.

        Args:
            observation, reward: Last observation and reward recieved from all n_e environments
            actions: Action vector (oneHot for discrete actions)
            previous_latent_state: previous latent state of type state.State
            other_actions: Action vector for other agents (oneHot for discrete actions)

        Returns:
            latent_state: New latent state
            others : other tensors for logging purposes

        """

        # Needed for legacy AESMC code
        ae_util.init(observation.is_cuda)

        # Legacy code: We need to pass in a (time) sequence of observations
        # With dim=0 for time
        img_observation = observation.unsqueeze(0)
        all_state = state.unsqueeze(0)
        actions = actions.unsqueeze(0)
        state_existence = state_agent_existence.unsqueeze(0)

        # Legacy code: All values are wrapped in state.State (which can contain more than one value)
        observation_states = st.State(
            all_x=img_observation.contiguous(),
            all_x_state=all_state.contiguous(),
            all_a=actions.contiguous(),
        )

        observation_states.all_state_existence = state_existence.contiguous()

        # Create graph structures for latent_state
        graph_batch = self.create_graph_structure(observation_states)
        graph_batch_particles = self.create_graph_structure_particles(observation_states)

        # Get 0th element (first element of time sequence) as input
        current_observation = observation_states.index_elements(0)

        if not self.separate_types:
            updated_type, updated_c1, updated_c2, updated_type_var, updated_c1_var, updated_c2_var, type1_dist = self.type_updating_network(current_observation, previous_latent_state)
        else:
            updated_type, updated_c1, updated_c2, updated_type_var, updated_c1_var, updated_c2_var, type1_dist, \
            updated_type2, updated_c12, updated_c22, updated_type2_var, updated_c12_var, updated_c22_var, type2_dist= self.type_updating_network(
                current_observation, previous_latent_state
            )
        # Update current latent state
        current_latent_state = st.State()
        sample_sizes = type1_dist.sample().size()

        type1_dist_kl_div = dist.kl_divergence(
            type1_dist, dist.Normal(torch.zeros(sample_sizes), torch.ones(sample_sizes))
        )

        type2_dist_kl_div = None
        if self.separate_types:
            type2_dist_kl_div = dist.kl_divergence(
                type2_dist, dist.Normal(torch.zeros(sample_sizes), torch.ones(sample_sizes))
            )

        # Set updated type vectors as type vectors of the current particle
        setattr(current_latent_state, "theta", updated_type)
        setattr(current_latent_state, "cell1", updated_c1)
        setattr(current_latent_state, "cell2", updated_c2)
        setattr(current_latent_state, "theta_var", updated_type_var)
        setattr(current_latent_state, "cell1_var", updated_c1_var)
        setattr(current_latent_state, "cell2_var", updated_c2_var)
        setattr(current_latent_state, "theta_sample", type1_dist.rsample())

        many_sample_size = [self.num_particles] + list(type1_dist.sample().size())
        type1_dist_expanded = type1_dist.expand(torch.Size(many_sample_size))
        many_sample = type1_dist_expanded.rsample()
        many_sample_likelihood = type1_dist_expanded.log_prob(many_sample)

        setattr(current_latent_state, "theta_many_samples", many_sample.permute([1,0,2,3]))
        setattr(current_latent_state, "theta_many_samples_likelihood", many_sample_likelihood.permute([1,0,2,3]))

        if self.separate_types:
            setattr(current_latent_state, "theta2", updated_type2)
            setattr(current_latent_state, "cell12", updated_c12)
            setattr(current_latent_state, "cell22", updated_c22)
            setattr(current_latent_state, "theta2_var", updated_type2_var)
            setattr(current_latent_state, "cell12_var", updated_c12_var)
            setattr(current_latent_state, "cell22_var", updated_c22_var)
            setattr(current_latent_state, "theta2_sample", type2_dist.rsample())

        # Get agent visibility distribution and observation vectors from current vector
        if not analysis_mode:

            obs_feature_dist, u_feature_dist, agent_existence_log_prob, sampled_existence = self.decoder_network(
               current_latent_state, current_observation, analysis_mode
            )


            # Compute likelihood of observed observation here
            agent_vis = current_observation.all_x_state[:, :, 1].unsqueeze(-1)

            all_state = torch.cat(
                [current_observation.all_x_state[:, :, :1], current_observation.all_x_state[:, :, 2:]], dim=-1
            )

            agent_inputs = all_state[:, :, :self.agent_inputs - 1]
            u_obs = all_state[:, 0, self.agent_inputs - 1:]

            a_obs_log_prob = obs_feature_dist.log_prob(
                agent_inputs
            )

            u_obs_log_prob = 0
            if self.with_global_feature:
                u_obs_log_prob = u_feature_dist.log_prob(
                    u_obs
                )

            # Get the likelihood of sampled teammate actions based on action prediction network
            predictor_action_parameters = self.act_predictor_net(
                graph_batch, current_latent_state
            )

            # Codes to compute action reconstruction loss for all agents
            all_seen_actions = other_actions
            valid_seen_actions = (all_seen_actions.sum(dim=-1) != 0)
            all_seen_valid_actions = all_seen_actions[valid_seen_actions]

            seen_action_distribution = None
            if not all_seen_valid_actions.nelement() == 0:
                if self.action_type == "continuous":
                    seen_action_distribution = dist.Normal(
                        predictor_action_parameters[0][:, 1:, :][valid_seen_actions],
                        predictor_action_parameters[1][:, 1:, :][valid_seen_actions]
                    )
                else:
                    seen_action_distribution = dist.OneHotCategorical(
                        logits=predictor_action_parameters[:, 1:, :][valid_seen_actions]
                    )

            # Safeguard against empty tensors.
            action_reconstruction_log_prob = 0.0
            if not all_seen_valid_actions.nelement() == 0:
                seen_action_log_probs = seen_action_distribution.log_prob(all_seen_valid_actions)
                if self.action_type == "discrete":
                    seen_action_log_probs = seen_action_log_probs.unsqueeze(-1)
                action_reconstruction_log_prob = seen_action_log_probs.mean()

            obs_reconstruction_component = a_obs_log_prob
            obs_reconstruction_component = obs_reconstruction_component.mean(dim=-1).mean(dim=-1)
            obs_reconstruction_component = obs_reconstruction_component + agent_existence_log_prob.mean(dim=-1).mean(dim=-1)

            if self.with_global_feature:
                obs_reconstruction_component = obs_reconstruction_component + (u_obs_log_prob.mean(dim=-1))

            # Initialize dictionaries to store other values for logging
            others = {}
            others["action_reconstruction_log_prob"] = action_reconstruction_log_prob
            others["graph"] = graph_batch
            others["graph_particles"] = graph_batch_particles
            others["agent_features_log_prob"] = a_obs_log_prob
            others["u_features_log_prob"] = u_obs_log_prob
            others["state_reconstruction_log_prob"] = obs_reconstruction_component.mean()
            others["agent_existence_log_prob"] = agent_existence_log_prob.mean()
            others["KL_div_type1"] = type1_dist_kl_div
            others["KL_div_type2"] = type2_dist_kl_div
            others["agent_existence_prob"] = sampled_existence
            

            return current_latent_state, \
                   others

        else: 
            # In analysis mode we sample from all particles. 
            sampled_existence = torch.tensor([])
            obs_feature_dist  = [] # torch.tensor([])
            u_feature_dist = [] # torch.tensor([])
            agent_existence_log_prob = torch.tensor([])
            predictor_action_parameters = [] 
            # sample from the decoder net from each particle
            for _ in range(self.num_particles):
                # sample new particle
                setattr(current_latent_state, "theta_sample", type1_dist.rsample())       
                # run decoder for each particle
                obs_feature_dist_n, u_feature_dist_n, agent_existence_log_prob_n, sampled_existence_n = self.decoder_network(
                current_latent_state, current_observation
                )

                # concatenate for each particle
                obs_feature_dist.append(obs_feature_dist_n) # = torch.cat([obs_feature_dist, obs_feature_dist_n])
                u_feature_dist.append(u_feature_dist_n)#  = torch.cat([u_feature_dist, u_feature_dist_n])
                agent_existence_log_prob = torch.cat([agent_existence_log_prob, agent_existence_log_prob_n])
                sampled_existence = torch.cat([sampled_existence, sampled_existence_n])

                # Get the likelihood of sampled teammate actions based on action prediction network
                # but for every particle
                predictor_action_parameters_n = self.act_predictor_net(
                    graph_batch, current_latent_state
                )

                predictor_action_parameters.append(predictor_action_parameters_n)



            # Compute likelihood of observed observation here
            agent_vis = current_observation.all_x_state[:, :, 1].unsqueeze(-1)

            all_state = torch.cat(
                [current_observation.all_x_state[:, :, :1], current_observation.all_x_state[:, :, 2:]], dim=-1
            )

            agent_inputs = all_state[:, :, :self.agent_inputs - 1]
            u_obs = all_state[:, 0, self.agent_inputs - 1:]


            a_obs_log_prob = [obs_feature_dist_per_particle.log_prob(agent_inputs) for obs_feature_dist_per_particle in obs_feature_dist]


            u_obs_log_prob = 0
            if self.with_global_feature:
                u_obs_log_prob = [u_feature_dist_per_particle.log_prob(u_obs) for u_feature_dist_per_particle in u_feature_dist]


            # Codes to compute action reconstruction loss for all agents
            all_seen_actions = other_actions
            valid_seen_actions = (all_seen_actions.sum(dim=-1) != 0)
            all_seen_valid_actions = all_seen_actions[valid_seen_actions]

            seen_action_distribution = None
            if not all_seen_valid_actions.nelement() == 0:
                seen_action_distribution = []
                for k in range(self.num_particles):

                    if self.action_type == "continuous":
                        seen_action_distributiona.append(dist.Normal(
                            predictor_action_parameters[k][0][:, 1:, :][valid_seen_actions],
                            predictor_action_parameters[k][1][:, 1:, :][valid_seen_actions]
                        ))
                    else:
                        seen_action_distribution.append(dist.OneHotCategorical(
                            logits=predictor_action_parameters[k][:, 1:, :][valid_seen_actions]
                        ))

            
            #print(seen_action_distribution)
            #print(all_seen_valid_actions)
            

            # Safeguard against empty tensors.
            action_reconstruction_log_prob = 0.0
            if not all_seen_valid_actions.nelement() == 0:
                
                # calculate log prob per particle 
                seen_action_log_probs_particle = [seen_action_distribution_particle.log_prob(all_seen_valid_actions) 
                            for seen_action_distribution_particle in seen_action_distribution]
                if self.action_type == "discrete":
                    seen_action_log_probs_particle = [particle.unsqueeze(-1) for particle in seen_action_log_probs_particle]
                action_reconstruction_log_prob_particle = [particle.mean() for particle in seen_action_log_probs_particle]
                action_reconstruction_log_prob = torch.tensor(action_reconstruction_log_prob_particle).mean()

 
            obs_reconstruction_component = [a_obs_log_prob_per_particle.mean(dim=-1).mean(dim=-1) for a_obs_log_prob_per_particle in a_obs_log_prob]
            obs_reconstruction_component = torch.tensor(obs_reconstruction_component).mean(dim=-1)
            agent_existence_component = [agent_existence_log_prob_per_particle.mean(dim=-1).mean(dim=-1) for agent_existence_log_prob_per_particle in agent_existence_log_prob]
            obs_reconstruction_component = obs_reconstruction_component + torch.tensor(agent_existence_component).mean(dim=-1)
            

            

            if self.with_global_feature:
                u_component = [u_obs_log_prob_per_particle.mean(dim=-1) for u_obs_log_prob_per_particle in u_obs_log_prob] 
                obs_reconstruction_component = obs_reconstruction_component + torch.tensor(u_component).mean(dim=-1)

            # Initialize dictionaries to store other values for logging
            others = {}
            others["action_reconstruction_log_prob"] = action_reconstruction_log_prob
            others["graph"] = graph_batch
            others["graph_particles"] = graph_batch_particles
            others["agent_features_log_prob"] = a_obs_log_prob
            others["u_features_log_prob"] = u_obs_log_prob
            others["state_reconstruction_log_prob"] = obs_reconstruction_component.mean()
            others["agent_existence_log_prob"] = agent_existence_log_prob.mean()
            others["KL_div_type1"] = type1_dist_kl_div
            others["KL_div_type2"] = type2_dist_kl_div
            others["agent_existence_prob"] = sampled_existence
            

            return current_latent_state, \
                   others

    def predict_action(self, graph_batch, previous_latent_state):
        predictor_action_parameters = self.act_predictor_net(
            graph_batch, previous_latent_state
        )

        predictor_action_parameters = predictor_action_parameters.unsqueeze(1).repeat(1,self.num_particles,1,1)

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

class ObsDecoderNet(nn.Module):
    """
        A class that decodes latent variable estimates in a particle into graph-based representation of the observation.
    """

    def __init__(
            self, h_dim, edge_dim, gnn_hdim1, gnn_hdim2,
            decoder_hid_dims, agent_obs_size,
            u_obs_size, device, with_global_feature
    ):
        """
            Args:
                h_dim : The length of agent type vectors in a particle.
                gnn_hdim1 : The size of the first hidden layer of the GNN used in this model.
                gnn_hdim2 : The size of the second hidden layer of the GNN used in this model.
                gnn_out_dim : The size of the output vector of the GNN used in this model.
                with_global_feature : Flag in case agents need to reconstruct global features from obs.
        """

        super().__init__()
        self.device = device

        self.deconstruction_nn = nn.Sequential(
            nn.Linear(h_dim, gnn_hdim1),
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
        h_sizes = input_h.size()

        batch_size, num_agent, h_dim = h_sizes[0], h_sizes[1], h_sizes[2]

        input = input_h
        n_out = self.deconstruction_nn(input)
        u_out = n_out.view(batch_size, num_agent, -1).sum(dim=-2)

        agent_visibility_dist = dist.Bernoulli(
            probs=F.sigmoid(self.visibility_net(n_out).view(batch_size, num_agent, -1))
        )

        agent_mean, agent_var = self.decoder(n_out)
        if self.with_global_feature:
            u_mean, u_var = self.decoder_u(u_out)

        agent_var = torch.ones_like(agent_var).to(self.device).double()

        if self.with_global_feature:
            u_var = torch.ones_like(u_var).to(self.device).double()

        agent_obs_dist = dist.Normal(
            agent_mean.view(batch_size, num_agent, -1),
            agent_var.view(batch_size, num_agent, -1)
        )

        u_obs_dist = None
        if self.with_global_feature:
            u_obs_dist = dist.Normal(
                u_mean.view(batch_size, -1),
                u_var.view(batch_size, -1)
            )

        return agent_visibility_dist, agent_obs_dist, u_obs_dist

class StateDecoderNet(nn.Module):
    """
        A class that decodes latent variable estimates in a particle into graph-based representation of the observation.
    """

    def __init__(
            self, h_dim, edge_dim, gnn_hdim1, gnn_hdim2,
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
            nn.Linear(h_dim + 1, gnn_hdim1),
            nn.ReLU(),
            nn.Linear(gnn_hdim1, gnn_hdim2),
            nn.ReLU(),
            nn.Linear(gnn_hdim2, decoder_hid_dims[-1])
        )

        self.decoder = Decoder(agent_obs_size, decoder_hid_dims)
        self.with_global_feature = with_global_feature
        if self.with_global_feature:
            self.decoder_u = Decoder(u_obs_size, decoder_hid_dims)

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
        cur_flag = current_latent_state.i

        h_sizes = input_h.size()

        batch_size, num_agent, h_dim = h_sizes[0], h_sizes[1], h_sizes[2]

        input = torch.cat([
            input_h,
            cur_flag
        ], -1).view(-1, h_dim + 1)

        n_out = self.deconstruction_nn(input)
        u_out = n_out.view(batch_size, num_agent, -1).sum(dim=-2)

        agent_mean, agent_var = self.decoder(n_out)
        if self.with_global_feature:
            u_mean, u_var = self.decoder_u(u_out)

        agent_var = torch.ones_like(agent_var).to(self.device).double()
        if self.with_global_feature:
            u_var = torch.ones_like(u_var).to(self.device).double()

        agent_obs_dist = dist.Normal(
            agent_mean.view(batch_size, num_agent, -1),
            agent_var.view(batch_size, num_agent, -1)
        )

        u_obs_dist = None
        if self.with_global_feature:
            u_obs_dist = dist.Normal(
                u_mean.view(batch_size, -1),
                u_var.view(batch_size, -1)
            )

        return agent_obs_dist, u_obs_dist

class StateDecoderNetWithOpenness(nn.Module):
    """
        A class that decodes latent variable estimates in a particle into graph-based representation of the states.
    """

    def __init__(
            self, h_dim, edge_dim, gnn_hdim1, gnn_hdim2,
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
            nn.Linear(h_dim, gnn_hdim1),
            nn.ReLU(),
            nn.Linear(gnn_hdim1, gnn_hdim2),
            nn.ReLU(),
            nn.Linear(gnn_hdim2, decoder_hid_dims[-1])
        )

        self.edge_net = nn.Linear(2*decoder_hid_dims[-1] + 1, h_dim)
        self.logit_net = nn.Sequential(
            nn.Linear(decoder_hid_dims[-1] + h_dim, 1)
        )

        self.decoder = Decoder(agent_obs_size, decoder_hid_dims)
        self.with_global_feature = with_global_feature
        if self.with_global_feature:
            self.decoder_u = Decoder(u_obs_size, decoder_hid_dims)

    def forward(self, current_latent_state, observation_states, analysis_mode=False):
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

        input_h = current_latent_state.theta_sample
        sampled_i = observation_states.all_state_existence

        h_sizes = input_h.size()

        batch_size, num_agent, h_dim = h_sizes[0], h_sizes[1], h_sizes[2]
        input = input_h.view(-1, h_dim)

        n_out = self.deconstruction_nn(input)
        n_out_reshaped = n_out.view(batch_size, num_agent, -1)
        u_out = n_out_reshaped.sum(dim=-2)

        # Assume agent index 0 for the learner always exists
        agent_flags = [sampled_i[:, 0, :].unsqueeze(-2)]
        node_logs = [torch.zeros(batch_size, 1, 1).to(self.device).double()]
        node_probs = []
        # Implement autoregressive reasoning
        for agent_offset in range(1, num_agent):
            # Get node representation of other agents that have been added to particle
            other_agent_data = n_out_reshaped[:, :agent_offset, :]

            # Get node representation of the agent that is to be added to the particle
            added_agent_data = n_out_reshaped[:, agent_offset, :]

            added_agent_data_reshaped = added_agent_data.unsqueeze(-2).repeat([1, agent_offset, 1])
            concatenated_agent_flags = torch.cat(agent_flags, dim=-2)

            # Autoregressive input will be representation of agents that already exist
            # Representation of agent to be added to the environment
            # and data on the agents' existence at the previous timestep
            edge_data = torch.cat([other_agent_data, added_agent_data_reshaped, concatenated_agent_flags], dim=-1)

            # Sum message from all existing nodes for deciding whether new agent is added or not
            edge_msg = self.edge_net(edge_data).sum(dim=-2)

            # Compute p(add new agent | prev agents data)
            input_logit_data = torch.cat([added_agent_data, edge_msg], dim=-1)
            added_probs_logits = self.logit_net(input_logit_data).view(batch_size, 1, -1)
            added_probs = F.sigmoid(added_probs_logits)
            node_dist = dist.Bernoulli(probs=added_probs)
            node_sample = sampled_i[:, agent_offset, :].unsqueeze(-2)
            add_logs = node_dist.log_prob(node_sample)

            # Output lsample existence (0/1) and log prob of sample
            new_flags = node_sample.view(batch_size, 1, -1)
            new_logs = add_logs.view(batch_size, 1, -1)

            agent_flags.append(new_flags)
            node_logs.append(new_logs)
            # storing prob for computing existence in analysis
            node_probs.append(added_probs.detach())
        
        agent_existence_logs = torch.cat(node_logs, dim=-2)

        agent_mean, agent_var = self.decoder(n_out)
        if self.with_global_feature:
            u_mean, u_var = self.decoder_u(u_out)

        agent_var = torch.ones_like(agent_var).to(self.device).double()
        if self.with_global_feature:
            u_var = torch.ones_like(u_var).to(self.device).double()

        agent_obs_dist = dist.Normal(
            agent_mean.view(batch_size, num_agent, -1),
            agent_var.view(batch_size, num_agent, -1)
        )

        u_obs_dist = None
        if self.with_global_feature:
            u_obs_dist = dist.Normal(
                u_mean.view(batch_size, -1),
                u_var.view(batch_size, -1)
            )
        if not analysis_mode:
            return agent_obs_dist, u_obs_dist, agent_existence_logs, node_probs
        return agent_obs_dist, u_obs_dist, agent_existence_logs, torch.tensor(node_probs).flatten()
