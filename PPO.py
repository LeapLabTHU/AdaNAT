import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from libs.policy import PolicyNet

from loguru import logger


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self, state_opt=None):
        self.actions = []
        self.states = {k: [] for k in state_opt}
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        for k, v in self.states.items():
            del v[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __repr__(self):
        return str(self.__dict__)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'states':
                for k, v in value.items():
                    if k not in self.states:
                        continue
                    self.states[k].append(v)
            else:
                if isinstance(value, torch.Tensor):
                    value = value.squeeze()
                self.__dict__[key].append(value)

    def to_tensor(self):
        res = []
        for key, value in self.__dict__.items():
            if key == 'is_terminals':
                continue
            if key == 'states':
                res.append({k: torch.cat(v).detach() for k, v in value.items()})
            else:
                res.append(torch.cat(value).detach())
        return res


class ActorCritic(nn.Module):
    def __init__(self, action_dim, has_continuous_action_space, action_std_init, device, state_opt, feat_dim, args):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.device = device
        self.state_opt = state_opt
        self.args = args

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = action_std_init * action_std_init
        assert has_continuous_action_space
        self.actor = PolicyNet(in_channels=feat_dim, out_channels=action_dim, hidden_size=512)
        self.critic = PolicyNet(in_channels=feat_dim, out_channels=1, hidden_size=512)
        logger.info('Policy net has {} parameters'.format(sum(p.numel() for p in self.parameters())))

    def set_action_std(self, new_action_std):
        self.action_var = new_action_std * new_action_std

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy, action_mean


class PPO(nn.Module):
    def __init__(self, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, device=None, action_std_init=0.6,
                 state_opt=None, feat_dim=None, args=None):
        super(PPO, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.buffer = RolloutBuffer(state_opt)

        self.policy = ActorCritic(action_dim, has_continuous_action_space, action_std_init, device, state_opt, feat_dim, args=args).to(device)

        self.policy_old = ActorCritic(action_dim, has_continuous_action_space, action_std_init, device, state_opt, feat_dim, args=args).to(device)
        self.policy_old.requires_grad_(False)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.args = args
        
        self.MseLoss = nn.MSELoss()
        logger.info('PPO net has {} parameters'.format(sum(p.numel() for p in self.parameters())))

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, update_buffer=True):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state)

        if update_buffer:
            self.buffer.update(states=state, actions=action, logprobs=action_logprob, state_values=state_val)
        return action

    def forward(self, *args, flag=None, **kwargs):
        if flag == 'select_action':
            return self.select_action(kwargs['state'], update_buffer=kwargs.get('update_buffer', True))
        elif flag == 'save':
            return self.save(kwargs['checkpoint_path'])
        elif flag == 'store_transition':
            self.buffer.update(rewards=kwargs['reward'], is_terminals=kwargs['done'])
        else:
            old_states, old_actions, old_logprobs, rewards, advantages, args = args
            logprobs, state_values, dist_entropy, actions = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            loss = loss.mean()

            return loss

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))





