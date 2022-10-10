import numpy as np
import pickle as pkl

class SequentialExperienceReplay(object):
    def __init__(self, ob_shape, act_shape, max_num_seq=2000, max_sequence_len=500):
        self.seq_list = [
            ExperienceReplay(ob_shape, act_shape, max_size=max_sequence_len) for _ in range(max_num_seq)
        ]
        self.seq_pointer = 0
        self.size = 0
        self.max_num_seq = max_num_seq

    def add(self, obses, acts, rewards, dones, n_obs, infos, obs_complete, acts_complete, n_obs_complete):
        experiences = list(zip(*[obses, acts, rewards, dones, n_obs, infos, obs_complete, acts_complete, n_obs_complete]))
        for experience in experiences:
            added_exp = list(zip(*list(experience)))
            n_added_seq = len(added_exp)
            for idx, exp in enumerate(added_exp):
                added_tuple = (
                    np.asarray([exp[0]]),
                    np.asarray([exp[1]]),
                    np.asarray([exp[2]]),
                    np.asarray([exp[3]]),
                    np.asarray([exp[4]]),
                    [exp[5]],
                    np.asarray([exp[6]]),
                    np.asarray([exp[8]]),
                    np.asarray([exp[7]])
                )
                self.seq_list[(self.seq_pointer + idx) % self.max_num_seq].add(*added_tuple)
        self.seq_pointer = (self.seq_pointer + n_added_seq) % self.max_num_seq
        self.size = min(self.size + n_added_seq, self.max_num_seq)

    def sample(self, num_samples):
        sampled_idx = np.random.randint(self.size, size=num_samples).tolist()
        sampled_seqs = [self.seq_list[sampled_id] for sampled_id in sampled_idx]
        return [sampled_seq.sample_all() for sampled_seq in sampled_seqs]

    def sample_all(self):
        return [sampled_seq.sample_all() for sampled_seq in self.seq_list]

    def save(self, dir_location):
        metadata = [self.seq_pointer, self.size, self.max_num_seq]
        with open(dir_location + '/sequence_replay.pkl', 'wb') as f:
            pkl.dump(self.seq_list, f)
        with open(dir_location + '/metadata.pkl', 'wb') as f:
            pkl.dump(metadata, f)

    def load(self, dir_location):
        with open(dir_location+"/sequence_replay.pkl", 'rb') as f:
            self.seq_list = pkl.load(f)
        with open(dir_location+"/metadata.pkl", 'rb') as f:
            metadata = pkl.load(f)

        self.seq_pointer = metadata[0]
        self.size = metadata[1]
        self.max_num_seq = metadata[2]

class ExperienceReplay(object):
    def __init__(self, ob_shape, act_shape, max_size=1e6):

        self.size = 0
        self.pointer = 0
        self.max_size = max_size

        obs_shape, acts_shape = [max_size], [max_size]
        obs_shape.extend(list(ob_shape))
        acts_shape.extend(list(act_shape))

        self.obs = np.zeros(obs_shape)
        self.obs_complete = np.zeros(obs_shape)
        self.n_obs = np.zeros(obs_shape)
        self.n_obs_complete = np.zeros(obs_shape)
        self.actions = np.zeros(acts_shape)
        self.actions_complete = np.zeros(acts_shape)
        self.rews = np.zeros([max_size, 1])
        self.dones = np.zeros([max_size, 1])
        self.info = [None] * max_size

    def add(self, ob, act, reward, done, n_ob, info, obs_complete, n_obs_complete, act_complete):
        input_size = ob.shape[0]

        if not self.pointer + input_size > self.max_size:
            self.obs[self.pointer:self.pointer+input_size] = ob
            self.actions[self.pointer:self.pointer+input_size] = act
            self.n_obs[self.pointer:self.pointer+input_size] = n_ob
            self.rews[self.pointer:self.pointer+input_size] = reward
            self.dones[self.pointer:self.pointer+input_size] = done
            self.info[self.pointer:self.pointer+input_size] = info
            self.obs_complete[self.pointer:self.pointer + input_size] = obs_complete
            self.n_obs_complete[self.pointer:self.pointer + input_size] = n_obs_complete
            self.actions_complete[self.pointer:self.pointer+input_size] = act_complete

        else:
            available = self.max_size - self.pointer
            self.obs[self.pointer:self.pointer + available] = ob[:available]
            self.actions[self.pointer:self.pointer + available] = act[:available]
            self.n_obs[self.pointer:self.pointer + available] = n_ob[:available]
            self.rews[self.pointer:self.pointer + available] = reward[:available]
            self.dones[self.pointer:self.pointer + available] = done[:available]
            self.info[self.pointer:self.pointer + available] = info[:available]
            self.obs_complete[self.pointer:self.pointer + available] = obs_complete[:available]
            self.n_obs_complete[self.pointer:self.pointer + available] = n_obs_complete[:available]
            self.actions_complete[self.pointer:self.pointer + available] = act_complete[:available]

            self.obs[0:input_size-available] = ob[available:]
            self.actions[0:input_size-available] = act[available:]
            self.n_obs[0:input_size-available] = n_ob[available:]
            self.rews[0:input_size-available] = reward[available:]
            self.dones[0:input_size-available] = done[available:]
            self.info[0:input_size-available] = info[available:]
            self.obs_complete[0:input_size - available] = obs_complete[available:]
            self.n_obs_complete[0:input_size - available] = n_obs_complete[available:]
            self.actions_complete[0:input_size-available] = act_complete[available:]

        self.size = min(self.size + input_size, self.max_size)
        self.pointer = (self.pointer + input_size) % self.max_size

    def sample(self, num_samples):
        sampled_idx = np.random.randint(self.size, size=num_samples)

        return self.obs[sampled_idx], self.actions[sampled_idx], \
               self.rews[sampled_idx], self.dones[sampled_idx], \
               self.n_obs[sampled_idx], [self.info[idx] for idx in sampled_idx.tolist()], \
               self.obs_complete[sampled_idx], self.actions_complete[sampled_idx], \
               self.n_obs_complete[sampled_idx]

    def sample_all(self):
        return self.obs, self.actions, self.rews, \
               self.dones, self.n_obs, self.info, \
               self.obs_complete, self.actions_complete, self.n_obs_complete

    def save(self, dir_location):
        with open(dir_location+'/obs.npy', 'wb') as f:
            np.save(f, self.obs)
        with open(dir_location+'/n_obs.npy', 'wb') as f:
            np.save(f, self.n_obs)
        with open(dir_location+'/actions.npy', 'wb') as f:
            np.save(f, self.actions)
        with open(dir_location+'/rewards.npy', 'wb') as f:
            np.save(f, self.rews)
        with open(dir_location+'/dones.npy', 'wb') as f:
            np.save(f, self.dones)
        with open(dir_location+'/info.pkl', 'wb') as f:
            pkl.dump(self.info, f)
        with open(dir_location+'/obs_complete.pkl', 'wb') as f:
            pkl.dump(self.obs_complete, f)
        with open(dir_location+'/actions_complete.pkl', 'wb') as f:
            pkl.dump(self.actions_complete, f)
        with open(dir_location+'/n_obs_complete.pkl', 'wb') as f:
            pkl.dump(self.n_obs_complete, f)

    def load(self, dir_location):
        self.obs = np.load(dir_location+"/obs.npy")
        self.n_obs = np.load(dir_location+"/n_obs.npy")
        self.actions = np.load(dir_location+"/actions.npy")
        self.rews = np.load(dir_location+"/rewards.npy")
        self.dones = np.load(dir_location+"/dones.npy")
        with open(dir_location+"/info.pkl", 'rb') as f:
            self.info = pkl.load(f)
        self.obs_complete = np.load(dir_location + "/obs_complete.npy")
        self.actions_complete = np.load(dir_location+"/actions_complete.npy")
        self.n_obs_complete = np.load(dir_location+"/n_obs_complete.npy")

        self.size = sum([a != None for a in self.info])
        self.pointer = (self.size) % self.max_size



