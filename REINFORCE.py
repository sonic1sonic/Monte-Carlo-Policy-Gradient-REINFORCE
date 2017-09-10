""" Monte-Carlo Policy Gradient """

from __future__ import print_function

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from torch.autograd import Variable

MAX_EPISODES = 1500
MAX_TIMESTEPS = 200

ALPHA = 3e-5
GAMMA = 0.99

class reinforce(nn.Module):

    def __init__(self):
        super(reinforce, self).__init__()
        # policy network
        self.fc1 = nn.Linear(4, 128)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def get_action(self, state):
        state = Variable(torch.Tensor(state))
        state = torch.unsqueeze(state, 0)
        probs = self.forward(state)
        probs = torch.squeeze(probs, 0)
        action = probs.multinomial()
        action = action.data
        action = action[0]
        return action

    def pi(self, s, a):
        s = Variable(torch.Tensor([s]))
        probs = self.forward(s)
        probs = torch.squeeze(probs, 0)
        return probs[a]

    def update_weight(self, states, actions, rewards, optimizer):
        G = Variable(torch.Tensor([0]))
        # for each step of the episode t = T - 1, ..., 0
        # r_tt represents r_{t+1}
        for s_t, a_t, r_tt in zip(states[::-1], actions[::-1], rewards[::-1]):
            G = Variable(torch.Tensor([r_tt])) + GAMMA * G
            loss = (-1.0) * G * torch.log(self.pi(s_t, a_t))
            # update policy parameter \theta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def main():

    env = gym.make('CartPole-v0')

    agent = reinforce()
    optimizer = optim.Adam(agent.parameters(), lr=ALPHA)

    for i_episode in range(MAX_EPISODES):

        state = env.reset()

        states = []
        actions = []
        rewards = [0]   # no reward at t = 0

        for timesteps in range(MAX_TIMESTEPS):

            action = agent.get_action(state)

            states.append(state)
            actions.append(action)

            state, reward, done, _ = env.step(action)

            rewards.append(reward)

            if done:
                print("Episode {} finished after {} timesteps".format(i_episode, timesteps+1))
                break

        agent.update_weight(states, actions, rewards, optimizer)

    env.close()

if __name__ == "__main__":
    main()
