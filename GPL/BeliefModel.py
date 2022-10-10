import torch.nn as nn

# Container to return all required values from model

class BeliefModelPriv(nn.Module):
    """
    Parent class to GPLBeliefModel.
    """

    def __init__(self, action_space, encoding_dimension):
        super().__init__()
        self.action_space = action_space
        self.encoding_dimension = encoding_dimension
        self.device = None

    def propagate(self, observation, reward, actions, previous_latent_state, other_actions, state, state_agent_existence, eval):
        """
        To be provided by child class. Updates the particles of the agent according to experience.
        Args:
            observation: current observation
            reward: current reward (not really needed atm, but we could learn to predict it)
            actions: last action on all n_s environments
            previous_latent_state: Latent state of particles from previous time step.
            predicted_times: list of ints or None, indicating whether predictions should be returned.

        Returns:
            new_latent_state: Updated latent state
            total_encoding_loss: E.g. L^{ELBO} in write up
            encoding_losses: log(w_{t}) for each particle
            avg_num_killed_particles: Average numer of killed particles in particle filter
        """
        raise NotImplementedError("Should be provided by child class, GPLBeliefModule.")

    def new_latent_state(self):
        """
        To be provided by child class. Creates particle ensembles for belief module.
        """
        raise NotImplementedError("Should be provided by child class, e.g. GPLBeliefModule.")

    def vec_conditional_new_latent_state(self, latent_state, mask):
        """
        To be provided by child class. Creates a new latent state for each environment in which the episode ended.
        """

    def forward(self, current_memory, with_resample=True, eval=False, with_noise="False"):
        """
        Run the model forward and compute all the stuff we need (see policy_return dictionary).

        Args:
            current_memory: Contains
              - 'current_obs': Current observation
              - 'oneHotActions': Actions, either oneHot (if discrete) or just the action (if continuous)
              - 'states': previous latent state of particles

        Returns:
            policy_return (dictionary): Contains:
              - 'latent_state': Updated latent state of model
              - 'total_encoding_loss': L^{ELBO} in write up, logsumexp(log(w_{t}))
              - 'encoding losses' (tuple): log(w_{t}) for ech particle
              - 'num_killed_particles': Number of averaged killed (no longer sampled) particles
              - 'others': Other tensors outputted for logging purposes
        """

        policy_return = {
            'latent_state': None,
            'total_encoding_loss': None,
            'encoding_losses': None,
            'num_killed_particles': None,
            'others': None
        }

        # Run the propagate function based on seen experience
        latent_state, total_encoding_loss, encoding_losses, n_killed_p, others = self.propagate(
                observation=current_memory['current_obs'].to(self.device),
                reward=current_memory['rewards'].to(self.device),
                actions=current_memory['actions'].to(self.device).detach(),
                previous_latent_state=current_memory['states'].to(self.device),
                other_actions=current_memory['other_actions'].to(self.device),
                state=current_memory['current_state'].to(self.device),
                state_agent_existence=current_memory["current_state"].to(self.device)[:,:,1].unsqueeze(-1),
                with_resample=with_resample, eval=eval, with_noise=with_noise
        )

        # Fill up policy_return with return values
        policy_return["latent_state"] = latent_state
        policy_return["total_encoding_loss"] = total_encoding_loss
        policy_return["encoding_losses"] = encoding_losses
        policy_return["num_killed_particles"] = n_killed_p
        policy_return["others"] = others

        return policy_return

class BeliefModel(nn.Module):
    """
    Parent class to GPLBeliefModel.
    """

    def __init__(self, action_space, encoding_dimension):
        super().__init__()
        self.action_space = action_space
        self.encoding_dimension = encoding_dimension
        self.device = None

    def propagate(self, observation, reward, actions, previous_latent_state, other_actions, state_agent_existence, eval):
        """
        To be provided by child class. Updates the particles of the agent according to experience.
        Args:
            observation: current observation
            reward: current reward (not really needed atm, but we could learn to predict it)
            actions: last action on all n_s environments
            previous_latent_state: Latent state of particles from previous time step.
            predicted_times: list of ints or None, indicating whether predictions should be returned.

        Returns:
            new_latent_state: Updated latent state
            total_encoding_loss: E.g. L^{ELBO} in write up
            encoding_losses: log(w_{t}) for each particle
            avg_num_killed_particles: Average numer of killed particles in particle filter
        """
        raise NotImplementedError("Should be provided by child class, GPLBeliefModule.")

    def new_latent_state(self):
        """
        To be provided by child class. Creates particle ensembles for belief module.
        """
        raise NotImplementedError("Should be provided by child class, e.g. GPLBeliefModule.")

    def vec_conditional_new_latent_state(self, latent_state, mask):
        """
        To be provided by child class. Creates a new latent state for each environment in which the episode ended.
        """

    def forward(self, current_memory, with_resample=True, eval=False):
        """
        Run the model forward and compute all the stuff we need (see policy_return dictionary).

        Args:
            current_memory: Contains
              - 'current_obs': Current observation
              - 'oneHotActions': Actions, either oneHot (if discrete) or just the action (if continuous)
              - 'states': previous latent state of particles

        Returns:
            policy_return (dictionary): Contains:
              - 'latent_state': Updated latent state of model
              - 'total_encoding_loss': L^{ELBO} in write up, logsumexp(log(w_{t}))
              - 'encoding losses' (tuple): log(w_{t}) for ech particle
              - 'num_killed_particles': Number of averaged killed (no longer sampled) particles
              - 'others': Other tensors outputted for logging purposes
        """

        policy_return = {
            'latent_state': None,
            'total_encoding_loss': None,
            'encoding_losses': None,
            'num_killed_particles': None,
            'others': None
        }

        # Run the propagate function based on seen experience
        latent_state, total_encoding_loss, encoding_losses, n_killed_p, others = self.propagate(
                observation=current_memory['current_obs'].to(self.device),
                reward=current_memory['rewards'].to(self.device),
                actions=current_memory['actions'].to(self.device).detach(),
                previous_latent_state=current_memory['states'].to(self.device),
                other_actions=current_memory['other_actions'].to(self.device),
                state_agent_existence=current_memory["current_state"].to(self.device)[:,:,1].unsqueeze(-1),
                with_resample=with_resample, eval=eval
        )

        # Fill up policy_return with return values
        policy_return["latent_state"] = latent_state
        policy_return["total_encoding_loss"] = total_encoding_loss
        policy_return["encoding_losses"] = encoding_losses
        policy_return["num_killed_particles"] = n_killed_p
        policy_return["others"] = others

        return policy_return
    
# Container to return all required values from RNN model
class RNNTypeModel(nn.Module):
    """
    Parent class to GPLBeliefModel.
    """

    def __init__(self, action_space, encoding_dimension):
        super().__init__()
        self.action_space = action_space
        self.encoding_dimension = encoding_dimension
        self.device = None

    def propagate(self, observation, actions, previous_latent_state, other_actions):
        """
        To be provided by child class. Updates the particles of the agent according to experience.
        Args:
            observation: current observation
            actions: last action on all n_s environments
            previous_latent_state: Latent state of particles from previous time step.
            other_actions: actions of other agents in the previous timestep.

        Returns:
            new_latent_state: Updated latent state
            others: Other tensors outputted for debugging/logging purposes
        """
        raise NotImplementedError("Should be provided by child class.")

    def new_latent_state(self):
        """
        To be provided by child class. Creates new latent state for learning.
        """
        raise NotImplementedError("Should be provided by child class.")

    def forward(self, current_memory):
        """
        Run the model forward and compute all the stuff we need (see policy_return dictionary).

        Args:
            current_memory: Contains
              - 'current_obs': Current observation
              - 'actions': Most recent action of learner, either oneHot (if discrete) or just the action (if continuous)
              - 'states': previous latent state of particles
              - 'other_actions': Most recent actions of other teammates, either oneHot (if discrete) or just the action (if continuous)

        Returns:
            policy_return (dictionary): Contains:
              - 'latent_state': Updated latent state of model
              - 'others': Other tensors outputted for logging purposes
        """

        policy_return = {
            'latent_state': None,
            'others': None
        }

        # Run the propagate function based on seen experience
        latent_state, others = self.propagate(
                observation=current_memory['current_obs'].to(self.device),
                actions=current_memory['actions'].to(self.device).detach(),
                previous_latent_state=current_memory['states'].to(self.device),
                other_actions=current_memory['other_actions'].to(self.device),
        )

        # Fill up policy_return with return values
        policy_return["latent_state"] = latent_state
        policy_return["others"] = others

        return policy_return

# Container to return all required values from RNN model with state reconstruction
class RNNTypeModelStateRecons(nn.Module):
    """
    Parent class to GPLBeliefModel.
    """

    def __init__(self, action_space, encoding_dimension):
        super().__init__()
        self.action_space = action_space
        self.encoding_dimension = encoding_dimension
        self.device = None

    def propagate(self, observation, state, actions, previous_latent_state, other_actions, state_agent_existence, eval):
        """
        To be provided by child class. Updates the particles of the agent according to experience.
        Args:
            observation: current observation
            actions: last action on all n_s environments
            previous_latent_state: Latent state of particles from previous time step.
            other_actions: actions of other agents in the previous timestep.

        Returns:
            new_latent_state: Updated latent state
            others: Other tensors outputted for debugging/logging purposes
        """
        raise NotImplementedError("Should be provided by child class.")

    def new_latent_state(self):
        """
        To be provided by child class. Creates new latent state for learning.
        """
        raise NotImplementedError("Should be provided by child class.")

    def forward(self, current_memory, eval=False, analysis_mode=False):
        """
        Run the model forward and compute all the stuff we need (see policy_return dictionary).

        Args:
            current_memory: Contains
              - 'current_obs': Current observation
              - 'actions': Most recent action of learner, either oneHot (if discrete) or just the action (if continuous)
              - 'states': previous latent state of particles
              - 'other_actions': Most recent actions of other teammates, either oneHot (if discrete) or just the action (if continuous)

        Returns:
            policy_return (dictionary): Contains:
              - 'latent_state': Updated latent state of model
              - 'others': Other tensors outputted for logging purposes
        """

        policy_return = {
            'latent_state': None,
            'others': None
        }

        # Run the propagate function based on seen experience
        latent_state, others = self.propagate(
                observation=current_memory['current_obs'].to(self.device),
                state=current_memory['current_state'].to(self.device),
                actions=current_memory['actions'].to(self.device).detach(),
                previous_latent_state=current_memory['states'].to(self.device),
                other_actions=current_memory['other_actions'].to(self.device),
                state_agent_existence=current_memory["current_state"].to(self.device)[:, :, 1].unsqueeze(-1),
                eval=eval, analysis_mode=analysis_mode
        )

        # Fill up policy_return with return values
        policy_return["latent_state"] = latent_state
        policy_return["others"] = others

        return policy_return
