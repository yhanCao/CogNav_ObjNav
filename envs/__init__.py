import torch
from .habitat import construct_envs_gibson, construct_envs_hm3d


# @brief: 根据yaml文件构建habitat环境，其中包括simulator的一些信息[agent info; sensor...]，Task[actions, measurements...]
def make_vec_envs(args):
    if args.task_config == 'tasks/objectnav_gibson.yaml':
        envs = construct_envs_gibson(args)
    elif args.task_config == 'tasks/objectnav_hm3d.yaml':
        envs,scenes = construct_envs_hm3d(args)
    envs = VecPyTorch(envs, args.device)
    return envs,scenes

# Adapted from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.device = device

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        obs, reward, done, info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def plan_act_and_preprocess(self, inputs):
        obs, reward, done, info = self.venv.plan_act_and_preprocess(inputs)
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, reward, done, info

    def close(self):
        return self.venv.close()