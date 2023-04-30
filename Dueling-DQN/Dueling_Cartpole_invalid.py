import gym
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt



class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(4, 64)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64, 256)
        self.fc_adv = nn.Linear(64, 256)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, 2)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        adv_average = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - adv_average

        return Q


class DuelingDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = DuelingDQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)


    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.from_numpy(state).float()
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        # Ajout d'une transition à la mémoire de l'agent
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.choices(self.memory, k=batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.from_numpy(state).float()
            next_state_tensor = torch.from_numpy(next_state).float()

            q_values = self.model(state_tensor)
            next_q_values = self.model(next_state_tensor)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(next_q_values)

            q_values[0][action] = target

            q_values = q_values.squeeze(0)
            target = torch.tensor(target)

            self.optimizer.zero_grad()
            loss = F.mse_loss(q_values, target)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DuelingDQNAgent(state_size, action_size)
batch_size = 32
episodes = 2000
rewards = []
for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0

    while not done:
        # Exploration/exploitation et action de l'agent
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Stockage de la transition dans la mémoire de l'agent
        agent.remember(state, action, reward, next_state, done)

        # Mise à jour de l'état actuel
        state = next_state

        # Calcul de la récompense totale
        total_reward += reward

        if done:
            # Entraînement de l'agent après chaque épisode
            agent.replay(batch_size)
    rewards.append(total_reward)

    # Affichage des résultats de l'épisode
    print("Episode:", episode + 1, "Total Reward:", total_reward)
plt.plot(rewards)
plt.xlabel('Épisode')
plt.ylabel('Récompense')
plt.title('Évolution des récompenses')
plt.show()