import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from core.model import BaseActorCriticNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ActorAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            flags
    ):
        self.model = BaseActorCriticNetwork(input_size, output_size, flags.std)

        self.output_size = output_size
        self.input_size = input_size
        self.flags = flags

        self.actor_optimizer = optim.SGD(self.model.actor.parameters(),
                                         lr=0.00005, momentum=0.9)
        self.critic_optimizer = optim.SGD(self.model.critic.parameters(),
                                          lr=0.0001, momentum=0.9)
        self.model = self.model.to(device)

    def get_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.tensor(state, device=device, dtype=torch.float32).reshape(1, -1)
            policy = self.model.actor(state)

            if evaluate:
                action = policy.mode().cpu().numpy().reshape(-1)
                return action
            else:
                action = policy.sample()
                return action.cpu().numpy().reshape(-1), policy.log_probs(action).cpu().item()

    def train_model(self, s_batch, action_batch, reward_batch, done_batch, behavior_logprob_batch, timeout_batch):
        s_batch = np.array(s_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        done_batch = np.array(done_batch)
        behavior_logprob_batch = np.array(behavior_logprob_batch)
        timeout_batch = np.array(timeout_batch)

        data_len = len(s_batch)

        critic_loss = 0
        actor_loss = 0

        indices = np.array(list(range(data_len)))
        valid_indices = indices[np.logical_not(timeout_batch)].tolist()

        # update critic
        with torch.no_grad():
            cur_policy, cur_value = self.model(torch.tensor(s_batch, device=device, dtype=torch.float32))
            cur_logprob_batch = cur_policy.log_probs(torch.tensor(action_batch, device=device, dtype=torch.float32)).cpu().detach().numpy()

        discounted_reward, _ = self.discount_return(reward_batch, done_batch, cur_value.cpu().detach().numpy(), behavior_logprob_batch, cur_logprob_batch, timeout_batch)

        for _ in range(self.flags.critic_update_iter):
            sample_idx = random.sample(valid_indices, self.flags.batch_size)
            sample_value = self.model.critic(torch.tensor(s_batch[sample_idx], device=device, dtype=torch.float32))
            loss = F.mse_loss(sample_value.squeeze(), torch.tensor(discounted_reward[sample_idx], device=device, dtype=torch.float32))
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            critic_loss += loss.cpu().item() / self.flags.critic_update_iter

        # update actor
        with torch.no_grad():
            cur_value = self.model.critic(torch.tensor(s_batch, device=device, dtype=torch.float32))
        discounted_reward, adv = self.discount_return(reward_batch, done_batch, cur_value.cpu().detach().numpy(), behavior_logprob_batch, cur_logprob_batch, timeout_batch)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(self.flags.actor_update_iter):
            sample_idx = random.sample(valid_indices, self.flags.batch_size)
            weight = torch.tensor(np.exp(np.minimum(adv[sample_idx] / self.flags.beta, np.log(self.flags.max_weight))), device=device, dtype=torch.float32).reshape(-1, 1)
            cur_policy = self.model.actor(torch.tensor(s_batch[sample_idx], device=device, dtype=torch.float32))

            probs = -cur_policy.log_probs(torch.tensor(action_batch[sample_idx], device=device, dtype=torch.float32))
            loss = probs * weight

            loss = loss.mean()

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            actor_loss += loss.cpu().item() / self.flags.actor_update_iter

        return critic_loss, actor_loss

    def discount_return(self, reward, done, value, behavior_logprob, cur_logprob, timeout):
        value = value.squeeze()
        cur_logprob = cur_logprob.squeeze()
        num_step = len(value)
        discounted_return = np.zeros([num_step])
        adv = np.zeros([num_step])

        clipped_rhos = np.minimum(np.exp(cur_logprob - behavior_logprob), 1)

        gae = 0
        for t in range(num_step - 1, -1, -1):
            if done[t]:
                if timeout[t]:
                    delta = value[t]
                else:
                    delta = reward[t] - value[t]
            else:
                delta = reward[t] + (1 - done[t]) * self.flags.discount * value[t + 1] - value[t]

            gae = delta + clipped_rhos[t] * self.flags.discount * self.flags.lambda_coeff * (1 - done[t]) * gae

            discounted_return[t] = gae + value[t]

            adv[t] = gae

        # for t in range(num_step - 1, -1, -1):
        #     curr_r = reward[t]
        #     if done[t]:
        #         if timeout[t]:
        #             curr_val = curr_r
        #         else:
        #             curr_val = value[t]
        #     else:
        #         next_ret = discounted_return[t + 1]
        #         curr_val = curr_r + self.flags.discount * ((1.0 - self.flags.lambda_coeff) * value[t + 1] + self.flags.lambda_coeff * next_ret)
        #     discounted_return[t] = curr_val
        #     adv[t] = curr_val - value[t]

        return discounted_return, adv
