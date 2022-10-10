from gym.envs.registration import register
import numpy as np

register(
    id='fortattack-v0',
    entry_point='FortAttack_gym.envs:OpenFortAttackGlobalEnv',
    kwargs={
        'max_timesteps':100, 'num_guards': 5, 'num_attackers' : 5, 'active_agents':3, 'num_freeze_steps':20,
        'seed':100, 'reward_mode':"normal", 'vision_radius': 2.0, 'arguments':None, 'with_oppo_modelling':False,
        'team_mode': None, 'agent_type':-1, 'cone_angle': np.pi/3, 'obs_mode': "conic"
    }
)
