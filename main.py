import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
# from gym_navigation.envs.navigation import ContinuousNavigation2DEnv, ContinuousNavigation2DNREnv
import cv2
import os

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.05, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--dclr', type=float, default=0.00001, metavar='G',
                    help='learning rate of discriminator (default: 0.00001)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--latent_size', type=int, default=2, metavar='N',
                    help='latent variable length (default: 2)')                    
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--buffer_size', type=int, default=100000, metavar='N',
                    help='Replay buffer for discriminator')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--suffix', type=str, default="",
                    help='suffix for model path')
args = parser.parse_args()

# Built-in config
bt_conf = dict()
bt_conf['render'] = False       # The env has '_render_trajectory' method or not
bt_conf['alpha tuning'] = True  # Scheduled alpha decreasing or not
bt_conf['include_r'] = False    # Include real reward in training or not

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
# agent.load_model(env_name=args.env_name)

#TesnorboardX
logdir = 'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
    args.env_name, args.policy, "autotune" if args.automatic_entropy_tuning else "")
writer = SummaryWriter(logdir=logdir)
logdir_img = logdir + '/img'
if not os.path.exists(logdir_img):
    os.makedirs(logdir_img)

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0
l_s = args.latent_size

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    episode_sr = 0 # pseudo reward
    episode_allr = 0 # all rewards
    done = False
    state = env.reset()

    context = np.random.random(l_s)
    context = context * 2 - 1. # scale to [-1, 1)
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state, context)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, disc_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/disc', disc_loss, updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

                # Reduce the entropy reward gain
                if bt_conf['alpha tuning']:
                    agent.adjust_alpha(1.0000046051807898) # reduced to 0.01 times in 1 million steps

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        pseudo_reward = agent.pseudo_score(context, state)

        episode_sr += pseudo_reward

        all_reward = pseudo_reward + reward
        episode_allr += all_reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        if bt_conf['include_r']:
            r = all_reward
        else:
            r = pseudo_reward
        memory.push(context, state, action, r, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    writer.add_scalar('reward/train_pseudo', episode_sr, i_episode)
    writer.add_scalar('reward/train_all', episode_allr, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, sr: {}, all: {}".format(i_episode,
        total_numsteps, episode_steps, round(episode_reward, 2), round(episode_sr, 2), round(episode_allr, 2)))

    if i_episode % 50 == 0 and args.eval == True:
        avg_reward = 0.
        avg_sr = 0.
        avg_all = 0.
        avg_reward_x = 0.
        avg_sr_x = 0.
        avg_all_x = 0.        
        episodes = 20
        # The test part is not compatible with high dimensional latent variables
        # at this time. An example for the 2D case.
        c = np.linspace(-1.0, 1.0, num=episodes)
        context = np.stack([c for _ in range(l_s)], axis=1)

        # Using mean for evaluation
        for i in range(episodes):
            state = env.reset()
            traj = []
            traj.append([state, None, 0.0, False])
            episode_reward = 0
            episode_sr = 0
            episode_allr = 0
            done = False
            
            while not done:
                action = agent.select_action(state, context[i], eval=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                traj.append([next_state, action, reward, done])

                pseudo_reward = agent.pseudo_score(context[i], state)
                episode_sr += pseudo_reward

                episode_allr += (pseudo_reward + reward)
                state = next_state
            avg_reward += episode_reward
            avg_sr += episode_sr
            avg_all += episode_allr
            if bt_conf['render']:
                img = env._render_trajectory(traj)
                cv2.imwrite("{}/test-{}-{}.png".format(logdir_img, i, context[i][0]), img * 255.0)

        # Sample actions for evaluation
        for i  in range(episodes):
            state = env.reset()
            traj = []
            traj.append([state, None, 0.0, False])
            episode_reward = 0
            episode_sr = 0
            episode_allr = 0
            done = False
            
            while not done:
                action = agent.select_action(state, context[i], eval=False)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                traj.append([next_state, action, reward, done])

                pseudo_reward = agent.pseudo_score(context[i], state)
                episode_sr += pseudo_reward

                episode_allr += (pseudo_reward + reward)
                state = next_state
            avg_reward_x += episode_reward
            avg_sr_x += episode_sr
            avg_all_x += episode_allr
            if bt_conf['render']:
                img = env._render_trajectory(traj)
                cv2.imwrite("{}/train-{}-{}.png".format(logdir_img, i, context[i][0]), img * 255.0)
        avg_reward /= episodes
        avg_sr /= episodes
        avg_all /= episodes
        avg_reward_x /= episodes
        avg_sr_x /= episodes
        avg_all_x /= episodes
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)
        writer.add_scalar('avg_reward/test_pseudo', avg_sr, i_episode)
        writer.add_scalar('avg_reward/test_all', avg_all, i_episode)
        writer.add_scalar('avg_reward_x/test', avg_reward_x, i_episode)
        writer.add_scalar('avg_reward_x/test_pseudo', avg_sr_x, i_episode)
        writer.add_scalar('avg_reward_x/test_all', avg_all_x, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}, Avg. SR: {}".format(episodes, round(avg_reward, 2), round(avg_sr, 2)))
        print("----------------------------------------")

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward_x: {}, Avg. SR_x: {}".format(episodes, round(avg_reward_x, 2), round(avg_sr_x, 2)))
        print("----------------------------------------")
# Save model
agent.save_model(args.env_name, suffix=args.suffix)
env.close()

