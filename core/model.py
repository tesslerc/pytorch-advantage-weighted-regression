import torch
import torch.nn as nn
from torch.distributions import Normal

FixedNormal = Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)
entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)
FixedNormal.mode = lambda self: self.mean


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias)

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class GaussianAction(nn.Module):
    def __init__(self, size_in, size_out, std):
        super().__init__()
        self.fc_mean = nn.Linear(size_in, size_out)

        # ====== INITIALIZATION ======
        self.fc_mean.weight.data.mul_(0.1)
        self.fc_mean.bias.data.mul_(0.0)

        self.logstd = (torch.ones(1, size_out).to(device) * std).log()

    def forward(self, x):
        action_mean = self.fc_mean(x)

        return FixedNormal(action_mean, self.logstd.exp())


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, std):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias((torch.ones(1, num_outputs) * std).log())

    def forward(self, x, logstd):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class BaseActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, std):
        super(BaseActorCriticNetwork, self).__init__()
        linear = nn.Linear

        self.actor = nn.Sequential(
            linear(input_size, 128),
            nn.ReLU(),
            linear(128, 64),
            nn.ReLU(),
            GaussianAction(64, output_size, std)
        )
        self.critic = nn.Sequential(
            linear(input_size, 128),
            nn.ReLU(),
            linear(128, 64),
            nn.ReLU(),
            linear(64, 1)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                nn.init.xavier_normal_(p.weight)
                p.bias.data.zero_()

    def forward(self, state):
        policy = self.actor(state)
        value = self.critic(state)
        return policy, value
