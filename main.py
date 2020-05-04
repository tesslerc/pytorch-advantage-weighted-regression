from collections import deque

import gym
import numpy as np
import argparse
import torch
import copy
from core.agent import ActorAgent
try:
    import wandb
except:
    wandb = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_policy(flags, agent):
    env = gym.make(flags.env)
    eval_episodes = 100
    total_return = 0
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            state = np.array(state)
            action = agent.get_action(state, True)
            state, reward, done, _ = env.step(action)

            total_return += reward

    return total_return / eval_episodes


def main(flags):
    env = gym.make(flags.env)

    if flags.wandb:
        wandb.init(project="awr")

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]

    agent = ActorAgent(
        input_size,
        output_size,
        flags
    )

    val = evaluate_policy(flags, agent)
    print("[Step {}] Evaluated reward: {}".format(0, val))
    best_params = agent.model.state_dict()
    best_return = val

    states = deque(maxlen=flags.replay_size)
    actions = deque(maxlen=flags.replay_size)
    rewards = deque(maxlen=flags.replay_size)
    dones = deque(maxlen=flags.replay_size)
    target_logprob = deque(maxlen=flags.replay_size)
    timeout = deque(maxlen=flags.replay_size)

    s_traj, a_traj, r_traj, d_traj, lp_traj, t_traj = [], [], [], [], [], []

    state = env.reset()
    episode_return = 0
    episode_timesteps = 0
    for step in range(flags.max_timesteps):
        episode_timesteps += 1
        if (step + 1) % flags.eval_freq == 0:
            val = evaluate_policy(flags, agent)
            if val > best_return:
                best_params = copy.deepcopy(agent.model.state_dict())
                best_return = val

            print("[Step {}] Evaluated reward: {}".format(step, val))
            if flags.wandb:
                wandb.log({"Evaluated reward": val}, step=step)

        state = np.array(state)
        action, log_prob = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        timedout = episode_timesteps >= env._max_episode_steps

        episode_return += reward

        s_traj.append(state)
        a_traj.append(action)
        r_traj.append(reward)
        d_traj.append(done)
        t_traj.append(timedout)
        lp_traj.append(log_prob)

        state = next_state[:]

        if (step + 1) % flags.samples_per_iter == 0 and step >= flags.start_timesteps:
            critic_loss, actor_loss = agent.train_model(states, actions, rewards, dones, target_logprob, timeout)

            if flags.wandb:
                wandb.log({"Critic loss": critic_loss, "Actor loss": actor_loss}, step=step)

        if done:
            if flags.wandb:
                wandb.log({"Reward": episode_return}, step=step)
            else:
                print("[Step {}] Episode reward: {}".format(step, episode_return))
            episode_return = 0
            episode_timesteps = 0
            while len(s_traj) > 0:
                states.append(s_traj.pop(0))
                actions.append(a_traj.pop(0))
                rewards.append(r_traj.pop(0))
                dones.append(d_traj.pop(0))
                target_logprob.append(lp_traj.pop(0))
                timeout.append(t_traj.pop(0))

            state = env.reset()

    agent.model.load_state_dict(best_params)
    val = evaluate_policy(flags, agent)
    print("[Final performance] Evaluated reward: {}".format(val))
    if flags.wandb:
        wandb.log({"Final performance": val})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=int(25e3), type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=int(5e3), type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)  # Max time steps to run environment
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--lambda_coeff", default=0.95, type=float)
    parser.add_argument("--actor_lr", default=0.00005, type=float)
    parser.add_argument("--critic_lr", default=0.0001, type=float)
    parser.add_argument("--actor_update_iter", default=1000, type=int)
    parser.add_argument("--critic_update_iter", default=200, type=int)
    parser.add_argument("--samples_per_iter", default=2048, type=int)
    parser.add_argument("--replay_size", default=50000, type=int)
    parser.add_argument("--beta", default=0.05, type=float)
    parser.add_argument("--std", default=0.4, type=float)
    parser.add_argument("--max_weight", default=20., type=float)
    parser.add_argument("--wandb", action="store_true")  # Logging using weights and biases
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    if wandb is None:
        args.wandb = False

    main(args)
