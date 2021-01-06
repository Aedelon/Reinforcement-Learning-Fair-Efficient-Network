# Code taken from https://github.com/PKU-AI-Edge/FEN
# Modified by Chloé TERWAGNE to respect the paper without hierarchy

# JOB SCHEDULING WITHOUT HIERARCHY

import os, sys, time
import numpy as np
import tensorflow as tf
import random
from keras.utils import np_utils, to_categorical
import tensorflow.compat.v1.keras.backend as KTF
import copy
import matplotlib.pyplot as plt
import json

# CHOOSE YOUR VARIABLES TO TEST THIS ENVIRONMENT

SIZE_POLICY_LAYER = 128
PPO_EPSILON = 0.2
FEN_EPSILON = 0.1
GAMMA = 0.98
NB_AGENTS = 4
NB_EPISODE = 1000
NB_STEPS = 1000
NB_RUNS = 5

WITH_GOSSIP = True


# Get state (position + distance between target and agent + observation) of each agent
# Get also the agents in observation of each agent
def get_obs(ant, target, env, n_agent):
    h = []
    h_2 = []
    for k in range(n_agent):
        state = []
        close_agent = []
        # Position of the agent
        state.append(ant[k][0])
        state.append(ant[k][1])
        # Distance with the target
        state.append(target[0] - ant[k][0])
        state.append(target[1] - ant[k][1])
        # Can the agent see the other agent ?
        for i in range(-1, 2):
            for j in range(-1, 2):
                r = int(env[ant[k][0] + i][ant[k][1] + j])
                state.append(r)
                if r != 0:
                    close_agent.append(r)

        h.append(state)
        h_2.append(close_agent)
    return h, h_2


# Select action for each agent
# ACTION 0 = stay
# ACTION 1 = go left
# ACTION 2 = go right
# ACTION 3 = go down
# ACTION 4 = go up
# +
# Get rewards (re = rewards table)
def step(env, ant, action):
    agents = [1, 2, 3, 4]
    n_agent = 4
    next_ant = []
    for i in range(n_agent):
        x = ant[i][0]
        y = ant[i][1]
        if action[i] == 0:
            next_ant.append([x, y])
        if action[i] == 1:
            x = x - 1
            if x == 0:
                next_ant.append([x + 1, y])
                continue
            if env[x][y] not in agents:
                env[x][y] = 1
                next_ant.append([x, y])
            else:
                next_ant.append([x + 1, y])
        if action[i] == 2:
            x = x + 1
            if x == 6:
                next_ant.append([x - 1, y])
                continue
            if env[x][y] not in agents:
                env[x][y] = 1
                next_ant.append([x, y])
            else:
                next_ant.append([x - 1, y])
        if action[i] == 3:
            y = y - 1
            if y == 0:
                next_ant.append([x, y + 1])
                continue
            if env[x][y] not in agents:
                env[x][y] = 1
                next_ant.append([x, y])
            else:
                next_ant.append([x, y + 1])
        if action[i] == 4:
            y = y + 1
            if y == 6:
                next_ant.append([x, y - 1])
                continue
            if env[x][y] not in agents:
                env[x][y] = 1
                next_ant.append([x, y])
            else:
                next_ant.append([x, y - 1])

    ant = next_ant
    env *= 0
    re = [0] * n_agent
    for i in range(n_agent):
        env[ant[i][0]][ant[i][1]] = i + 1
        if (ant[i][0] == target[0]) & (ant[i][1] == target[1]):
            re[i] = 1
    return env, ant, re


# Class MLP that estimate the value of the rewards for PPO
# Output = real
class ValueNetwork():
    # Create the MLP
    def __init__(self, num_features, hidden_size, learning_rate=10e-3):
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.session = tf.compat.v1.Session()

            self.observations = tf.compat.v1.placeholder(shape=[None, self.num_features], dtype=tf.float32)
            self.W = [
                tf.compat.v1.get_variable("W1", shape=[self.num_features, self.hidden_size]),
                tf.compat.v1.get_variable("W2", shape=[self.hidden_size, self.hidden_size]),
                tf.compat.v1.get_variable("W3", shape=[self.hidden_size, 1])
            ]
            self.layer_1 = tf.nn.relu(tf.matmul(self.observations, self.W[0]))
            self.layer_2 = tf.nn.relu(tf.matmul(self.layer_1, self.W[1]))
            self.output = tf.reshape(tf.matmul(self.layer_2, self.W[2]), [-1])

            self.rollout = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.compat.v1.losses.mean_squared_error(self.output, self.rollout)
            self.grad_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            self.minimize = self.grad_optimizer.minimize(self.loss)

            init = tf.compat.v1.global_variables_initializer()
            self.session.run(init)

    # Get the estimated reward for the batch of states
    def get(self, states):
        value = self.session.run(self.output, feed_dict={self.observations: states})
        return value

    # Update
    def update(self, states, discounted_rewards):
        _, loss = self.session.run([self.minimize, self.loss], feed_dict={
            self.observations: states, self.rollout: discounted_rewards
        })


# Class MLP that choose the action for the policy in PPO
# Output = proba
class PPOPolicyNetwork():
    # Create the MLP
    def __init__(self, num_features, layer_size, num_actions, epsilon=.2,
                 learning_rate=3e-4):
        self.tf_graph = tf.Graph()

        with self.tf_graph.as_default():
            self.session = tf.compat.v1.Session()

            self.observations = tf.compat.v1.placeholder(shape=[None, num_features], dtype=tf.float32)
            self.W = [
                tf.compat.v1.get_variable("W1", shape=[num_features, layer_size]),
                tf.compat.v1.get_variable("W2", shape=[layer_size, layer_size]),
                tf.compat.v1.get_variable("W3", shape=[layer_size, num_actions])
            ]

            self.saver = tf.compat.v1.train.Saver(self.W, max_to_keep=1000)

            self.output = tf.nn.relu(tf.matmul(self.observations, self.W[0]))
            self.output = tf.nn.relu(tf.matmul(self.output, self.W[1]))
            self.output = tf.nn.softmax(tf.matmul(self.output, self.W[2]))

            self.advantages = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

            self.chosen_actions = tf.compat.v1.placeholder(shape=[None, num_actions], dtype=tf.float32)
            self.old_probabilities = tf.compat.v1.placeholder(shape=[None, num_actions], dtype=tf.float32)

            self.new_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.output, axis=1)
            self.old_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.old_probabilities, axis=1)

            # Avoid 0 division
            self.ratio = tf.divide(self.new_responsible_outputs,
                                   tf.maximum(self.old_responsible_outputs, 1e-9))

            self.loss = tf.reshape(
                tf.minimum(
                    tf.multiply(self.ratio, self.advantages),
                    tf.multiply(tf.clip_by_value(self.ratio, 1 - epsilon, 1 + epsilon), self.advantages)),
                [-1]
            ) - 0.03 * self.new_responsible_outputs * tf.math.log(self.new_responsible_outputs + 1e-10)
            self.loss = -tf.reduce_mean(self.loss)

            self.W0_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.W1_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.W2_grad = tf.compat.v1.placeholder(dtype=tf.float32)

            self.gradient_placeholders = [self.W0_grad, self.W1_grad, self.W2_grad]
            self.trainable_vars = self.W
            self.gradients = [(np.zeros(var.get_shape()), var) for var in self.trainable_vars]

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            self.get_grad = self.optimizer.compute_gradients(self.loss, self.trainable_vars)
            self.apply_grad = self.optimizer.apply_gradients(zip(self.gradient_placeholders, self.trainable_vars))
            init = tf.compat.v1.global_variables_initializer()
            self.session.run(init)

    # Get current distribution to get the probabilities by action for a given observation
    def get_dist(self, states):
        dist = self.session.run(self.output, feed_dict={self.observations: states})
        return dist

    # Update
    def update(self, states, chosen_actions, ep_advantages):
        old_probabilities = self.session.run(self.output, feed_dict={self.observations: states})
        self.session.run(self.apply_grad, feed_dict={
            self.W0_grad: self.gradients[0][0],
            self.W1_grad: self.gradients[1][0],
            self.W2_grad: self.gradients[2][0],

        })
        self.gradients, loss = self.session.run([self.get_grad, self.output], feed_dict={
            self.observations: states,
            self.advantages: ep_advantages,
            self.chosen_actions: chosen_actions,
            self.old_probabilities: old_probabilities
        })

    # Save PPO
    def save_w(self, name):
        self.saver.save(self.session, name + '.ckpt')

    # Save PPO
    def restore_w(self, name):
        self.saver.restore(self.session, name + '.ckpt')


# Get the first part of A_t function (for PPO)
def discount_rewards(rewards, gamma):
    running_total = 0
    discounted = np.zeros_like(rewards)
    for r in reversed(range(len(rewards))):
        running_total = running_total * gamma + rewards[r]
        discounted[r] = running_total
    return discounted


# INIT - MAIN
nb_runs = NB_RUNS
n_agent = NB_AGENTS

for run in range(nb_runs):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    KTF.set_session(session)
    # T = number of steps before updating the sub-policy and get another one
    T = 25
    totalTime = 0
    # Coefficient for calculating reward (see course - long term reward has less cost than early term reward)
    gamma = GAMMA
    n_episode = NB_EPISODE
    max_steps = NB_STEPS
    i_episode = 0
    # number of action
    n_actions = 5
    # number of sub policies

    # Stuff to make stats
    mean_fair_efficient_reward = np.zeros(n_episode)
    avg_agents_episode_1 = []
    avg_agents_episode_500 = []
    avg_agents_episode_999 = []
    for i in range(n_agent):
        avg_agents_episode_1.append(np.zeros(max_steps))
        avg_agents_episode_500.append(np.zeros(max_steps))
        avg_agents_episode_999.append(np.zeros(max_steps))

    # Creating PPO (actor - critic) for the sub-policy
    # In that case, the sub-policy or common to every agents
    Pi = PPOPolicyNetwork(num_features=13, num_actions=n_actions,
                          layer_size=SIZE_POLICY_LAYER,
                          epsilon=PPO_EPSILON, learning_rate=0.0003)
    V = ValueNetwork(num_features=13,
                     hidden_size=SIZE_POLICY_LAYER,
                     learning_rate=0.001)

    # Begin the training
    while i_episode < n_episode:
        reward_for_episode_for_all = []

        # Each episode, we will re-initialize the whole environment but with the trained
        # controller and the trained sub-policies
        i_episode += 1

        # Create array of variable for calculate utilities for each agent
        avg = [0] * n_agent
        u_bar = [0] * n_agent
        utili = [0] * n_agent
        u = [[] for _ in range(n_agent)]
        max_u = 0.25

        # Create batch to update the selected sub-policy (will be recreated each T steps)
        ep_actions = [[] for _ in range(n_agent)]
        ep_rewards = [[] for _ in range(n_agent)]
        ep_states = [[] for _ in range(n_agent)]

        # Array of selected sub-policies for each agent
        signal = [0] * n_agent
        rat = [0.0] * n_agent

        # Total score
        score = 0
        # Total steps
        steps = 0
        # Create the environment (the map)
        # Matrix 8x8 to have no bug with the observation but
        # he real env is 5x5
        env = np.zeros((8, 8))
        # Put a target somewhere in the environment
        target = np.random.randint(2, 5, 2)
        # Array of position of the agent in the map
        # Agent will be 1 on the map
        ant = []
        for i in range(n_agent):
            ant.append(np.random.randint(1, 6, 2))
            env[ant[i][0]][ant[i][1]] = i + 1

        # array of "reward" (utility) for each agent
        su = np.zeros(n_agent)

        # Get first observation
        obs, nodes = get_obs(ant, target, env, n_agent)

        # Begin the current episode
        while steps < max_steps:
            steps += 1
            action = []
            # Each step we choose a new action
            # The following process is the same as for the controller
            # We get just the observation so the num_features is 13
            for i in range(n_agent):
                h = copy.deepcopy(obs[i])
                p = Pi.get_dist(np.array([h]))[0]
                action.append(np.random.choice(range(n_actions), p=p))
                ep_states[i].append(h)
                ep_actions[i].append(to_categorical(action[i], n_actions))

            # Do the action on the environment for each agent
            env, ant, rewards = step(env, ant, action)
            # Put the reward in the array of reward for each agent
            su += np.array(rewards)
            # Add to the total score
            score += sum(rewards)
            # Get new observation for each agent
            obs, nodes = get_obs(ant, target, env, n_agent)

            # Calculation of the "rewards" (utility)
            for i in range(n_agent):
                # Add to the each array utility agent the reward
                u[i].append(rewards[i])
                # Average the reward for the agent
                u_bar[i] = sum(u[i]) / len(u[i])

            # Gossip algorithm
            if WITH_GOSSIP:
                if steps % T == 0:
                    avg = copy.deepcopy(u_bar)
                    for i in range(n_agent):
                        avg_1 = avg[i]
                        for j in nodes[i]:
                            if i != j:
                                avg_1 += 1 / (np.maximum(len(nodes[i]), len(nodes[j - 1])) + 1) * (avg[j - 1] - avg[i])
                        avg[i] = avg_1

            for i in range(n_agent):
                # Average of the average reward of each agent - CENTRALIZED
                if not WITH_GOSSIP:
                    avg[i] = sum(u_bar) / len(u_bar)

                if avg[i] != 0:
                    rat[i] = (u_bar[i] - avg[i]) / avg[i]
                else:
                    rat[i] = 0

                # Stuff for stats
                if i_episode == 1:
                    avg_agents_episode_1[i][steps - 1] = avg[i]
                if i_episode == 500:
                    avg_agents_episode_500[i][steps - 1] = avg[i]
                if i_episode == 999:
                    avg_agents_episode_999[i][steps - 1] = avg[i]

                # Average the reward for the agent < max_u (= 1/4) take it
                # The utili can't get over 1
                utili[i] = min(1, avg[i] / max_u)

                ep_rewards[i].append(utili[i] / (0.1 + abs(rat[i])))

            # Update the policy
            if steps % T == 0:
                for i in range(n_agent):
                    # Batch to update policy
                    ep_actions[i] = np.array(ep_actions[i])
                    ep_rewards[i] = np.array(ep_rewards[i], dtype=np.float_)
                    ep_states[i] = np.array(ep_states[i])

                    # Get the target reward with the formula of reward on time (see course)
                    targets = discount_rewards(ep_rewards[i], GAMMA)
                    # Update Critic MLP before estimating the reward
                    V.update(ep_states[i], targets)
                    # Get the estimated reward
                    vs = V.get(ep_states[i])
                    # Calculate A_t for PPO (see paper and video PPO)
                    ep_advantages = targets - vs
                    ep_advantages = (ep_advantages - np.mean(ep_advantages)) / (np.std(ep_advantages) + 0.0000000001)
                    # UPDATE PPO of policy
                    Pi.update(ep_states[i], ep_actions[i], ep_advantages)

                    reward_for_episode_for_all += ep_rewards

                # Re-initialize BATCH
                ep_actions = [[] for _ in range(n_agent)]
                ep_rewards = [[] for _ in range(n_agent)]
                ep_states = [[] for _ in range(n_agent)]

        mean_fair_efficient_reward[i_episode - 1] = np.mean(reward_for_episode_for_all)

        print('RUNS:', run)
        print('EPISODE:', i_episode)
        print('MEAN SCORE BY STEP:', score / max_steps)
        print('MFER:', mean_fair_efficient_reward[i_episode - 1])
        print(su)
        print('-------------------')

    # FOR THE PAPER --------------
    print('RUNS:', run)
    print('MFER:', mean_fair_efficient_reward)
    print('TABLE AVG:')
    with open('/data_centralizedbis' + str(run) + '.txt', 'w') as writefile:
        writefile.write("mean_fair_efficient_reward\n")
        json.dump(mean_fair_efficient_reward.tolist(), writefile)
        writefile.write("\n")
        writefile.write("avg_agents_episode_1\n")
        for i in range(n_agent):
            json.dump(avg_agents_episode_1[i].tolist(), writefile)
            writefile.write("\n")
        writefile.write("avg_agents_episode_500\n")
        for i in range(n_agent):
            json.dump(avg_agents_episode_500[i].tolist(), writefile)
            writefile.write("\n")
        writefile.write("avg_agents_episode_999\n")
        for i in range(n_agent):
            json.dump(avg_agents_episode_999[i].tolist(), writefile)
            writefile.write("\n")
