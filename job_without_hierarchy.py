# JOB SCHEDULING
#commit
import os, sys, time
import numpy as np
import tensorflow as tf
import csv
import random
from keras.utils import np_utils, to_categorical
import keras.backend.tensorflow_backend as KTF
import copy
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Initialisation environment
# env = map
# ant = list of agent position
# The lines are just shit
n_agent = 4
env = np.zeros((8, 8))
target = np.random.randint(2, 5, 2)
ant = []
for i in range(n_agent):
    ant.append(np.random.randint(1, 6, 2))
    env[ant[i][0]][ant[i][1]] = 1
start = time.time()

# Get state (position + distance between target and agent + observation) of each agent
def get_obs(ant, target, env, n_agent):
    h = []
    for k in range(n_agent):
        state = []
        # Position of the agent
        state.append(ant[k][0])
        state.append(ant[k][1])
        # Distance with the target
        state.append(target[0] - ant[k][0])
        state.append(target[1] - ant[k][1])
        # Can the agent see the other agent ?
        for i in range(-1, 2):
            for j in range(-1, 2):
                state.append(env[ant[k][0] + i][ant[k][1] + j])
        h.append(state)
    return h


# Select action for each agent
# ACTION 0 = stay
# ACTION 1 = go left
# ACTION 2 = go right
# ACTION 3 = go down
# ACTION 4 = go up
# +
# Get rewards (re = rewards table)
def step(env, ant, action):
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
            if env[x][y] != 1:
                env[x][y] = 1
                next_ant.append([x, y])
            else:
                next_ant.append([x + 1, y])
        if action[i] == 2:
            x = x + 1
            if x == 6:
                next_ant.append([x - 1, y])
                continue
            if env[x][y] != 1:
                env[x][y] = 1
                next_ant.append([x, y])
            else:
                next_ant.append([x - 1, y])
        if action[i] == 3:
            y = y - 1
            if y == 0:
                next_ant.append([x, y + 1])
                continue
            if env[x][y] != 1:
                env[x][y] = 1
                next_ant.append([x, y])
            else:
                next_ant.append([x, y + 1])
        if action[i] == 4:
            y = y + 1
            if y == 6:
                next_ant.append([x, y - 1])
                continue
            if env[x][y] != 1:
                env[x][y] = 1
                next_ant.append([x, y])
            else:
                next_ant.append([x, y - 1])
    ant = next_ant
    env *= 0
    re = [0] * n_agent
    for i in range(n_agent):
        env[ant[i][0]][ant[i][1]] = 1
        if (ant[i][0] == target[0]) & (ant[i][1] == target[1]):
            re[i] = 1
    return env, ant, re


# Class MLP that estimate the value of the rewards for PPO
# Output = real
class ValueNetwork():
    # Create the MLP
    def __init__(self, num_features, hidden_size, learning_rate=.01):
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.session = tf.Session()

            self.observations = tf.placeholder(shape=[None, self.num_features], dtype=tf.float32)
            self.W = [
                tf.get_variable("W1", shape=[self.num_features, self.hidden_size]),
                tf.get_variable("W2", shape=[self.hidden_size, self.hidden_size]),
                tf.get_variable("W3", shape=[self.hidden_size, 1])
            ]
            self.layer_1 = tf.nn.relu(tf.matmul(self.observations, self.W[0]))
            self.layer_2 = tf.nn.relu(tf.matmul(self.layer_1, self.W[1]))
            self.output = tf.reshape(tf.matmul(self.layer_2, self.W[2]), [-1])

            self.rollout = tf.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.losses.mean_squared_error(self.output, self.rollout)
            self.grad_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.minimize = self.grad_optimizer.minimize(self.loss)

            init = tf.global_variables_initializer()
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
                 learning_rate=9e-4):
        self.tf_graph = tf.Graph()

        with self.tf_graph.as_default():
            self.session = tf.Session()

            self.observations = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
            self.W = [
                tf.get_variable("W1", shape=[num_features, layer_size]),
                tf.get_variable("W2", shape=[layer_size, layer_size]),
                tf.get_variable("W3", shape=[layer_size, num_actions])
            ]

            self.saver = tf.train.Saver(self.W, max_to_keep=1000)

            self.output = tf.nn.relu(tf.matmul(self.observations, self.W[0]))
            self.output = tf.nn.relu(tf.matmul(self.output, self.W[1]))
            self.output = tf.nn.softmax(tf.matmul(self.output, self.W[2]))

            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            self.chosen_actions = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)
            self.old_probabilities = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)

            self.new_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.output, axis=1)
            self.old_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.old_probabilities, axis=1)

            self.ratio = self.new_responsible_outputs / self.old_responsible_outputs

            self.loss = tf.reshape(
                tf.minimum(
                    tf.multiply(self.ratio, self.advantages),
                    tf.multiply(tf.clip_by_value(self.ratio, 1 - epsilon, 1 + epsilon), self.advantages)),
                [-1]
            ) - 0.03 * self.new_responsible_outputs * tf.log(self.new_responsible_outputs + 1e-10)
            self.loss = -tf.reduce_mean(self.loss)

            self.W0_grad = tf.placeholder(dtype=tf.float32)
            self.W1_grad = tf.placeholder(dtype=tf.float32)
            self.W2_grad = tf.placeholder(dtype=tf.float32)

            self.gradient_placeholders = [self.W0_grad, self.W1_grad, self.W2_grad]
            self.trainable_vars = self.W
            self.gradients = [(np.zeros(var.get_shape()), var) for var in self.trainable_vars]

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.get_grad = self.optimizer.compute_gradients(self.loss, self.trainable_vars)
            self.apply_grad = self.optimizer.apply_gradients(zip(self.gradient_placeholders, self.trainable_vars))
            init = tf.global_variables_initializer()
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


def plot_reward(array_r_mean_over_agents, nb_simulation):

    x = [x for x in range(1, len(array_r_mean_over_agents)+1)]
    fig, axs = plt.subplots(2, 1)

    r_mean_over_agents = np.mean(array_r_mean_over_agents, axis=0)
    r_agents = np.mean(array_r_mean_over_agents, axis=0)

    axs[0].plot(x, r_mean_over_agents, linestyle='-', label="FEN with Hierarchy in job scheduling")

    # plot mean
    for i in range(0, len(r_agents)):
        axs[1].plot(x, r_agents[i], linestyle='-', label="Agent "+str(i+1))
        #yerror_bottom = [(estimate_action[i] - 2*standard_dev[i]) for x in range(0, time_steps+1)]
        #yerror_top = [(estimate_action[i] + 2*standard_dev[i]) for x in range(0, time_steps + 1)]
        #axes[i].fill_between(x, yerror_bottom, yerror_top, facecolor='lightpink', label=r'$Q_a^*\pm 2 std$')

    axs[0].title.set_text('Learning curve averaged over '+str(nb_simulation)+" simulations")
    axs[0].set_ylabel('Mean fair-efficient reward')
    axs[1].title.set_text('Learning curve for each agent averaged over '+str(nb_simulation)+" simulations")
    axs[1].set_ylabel('Fair-efficient reward for each agent')

    #fig.suptitle( r'Comparison between $Q_{ai}^*$ and actual $Q_{ai}$ estimate for each arm i over time')
    fig.text(0.5, 0.04, "Episodes", ha="center", va="center")
    #fig.text(0.05, 0.5, "estimate value (averaged over "+str(nb_trials)+" simulations)", ha="center", va="center", rotation=90)
    plt.legend(loc='lower right')
    plt.show()




# INIT - MAIN
def run_fen_wo_hierarchy(T, totalTime, GAMMA, n_episode, max_steps, i_episode, n_actions, render):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    KTF.set_session(session)

    # ----------------------
    mean_reward_all_player = []
    mean_reward_by_agent = [[] for _ in range(n_agent)]
    episodes_x = []
    # ------------------------


    Pi = PPOPolicyNetwork(num_features=13, num_actions=n_actions, layer_size=256, epsilon=0.2, learning_rate=0.0003)
    V = ValueNetwork(num_features=13, hidden_size=256, learning_rate=0.001)

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
        # Je pense que le nom est mal choisi
        ep_actions = [[] for _ in range(n_agent)]
        ep_rewards = [[] for _ in range(n_agent)]
        ep_states = [[] for _ in range(n_agent)]

        # Array of selected sub-policies for each agent

        rat = [0.0] * n_agent

        # Total score ??
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
            env[ant[i][0]][ant[i][1]] = 1

        # array of "reward" (utility) for each agent
        su = [0] * n_agent
        ac = [0] * n_actions
        su = np.array(su)

        # Get first observation
        obs = get_obs(ant, target, env, n_agent)

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
            obs = get_obs(ant, target, env, n_agent)

            # Calculation of the "rewards" (utility)
            # I think it's the calculation for the controller
            for i in range(n_agent):
                # Add to the each array utility agent the reward
                u[i].append(rewards[i])
                # Average the reward for the agent
                u_bar[i] = sum(u[i]) / len(u[i])

            for i in range(n_agent):
                # Average of the average reward of each agent
                avg[i] = sum(u_bar) / len(u_bar)
                if avg[i] != 0:
                    # !!!!!! Can't get this formula, need help !!!!!
                    rat[i] = (u_bar[i] - avg[i]) / avg[i]
                else:
                    rat[i] = 0

                # Average the reward for the agent < max_u (= 1/4) take it
                # The utili can't get over 1
                # !!!!! Don't know either where i can find that in the paper !!!!!
                utili[i] = min(1, avg[i] / max_u)

                ep_rewards[i].append(utili[i] / (0.1 + abs(rat[i])))

            # Utility 2
            # I think it's the calculation for sub-policies
            # Sub-policy 1 = maximazing the result
            # Sub-policy 2-3-4 = try to learn diverse behaviors to meet
            # the controller's demand of fairness

            # Update PPO of the sub-policy of each agent
            if steps % T == 0:
                for i in range(n_agent):
                    # I DON'T KNOW WHERE IS THE "-1" OF THE PAPER FORMULA
                    # Thius line is to complete the batch of reward to update the controller

                    # Batch to update sub-policy
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
                    # !!!!!!!! Something in PPO i don't understand - TO INVESTIGATE !!!!!!
                    ep_advantages = (ep_advantages - np.mean(ep_advantages)) / (np.std(ep_advantages) + 0.0000000001)
                    # UPDATE PPO of sub-policy
                    Pi.update(ep_states[i], ep_actions[i], ep_advantages)
                    reward_for_episode_for_all += ep_rewards

                    #print("In loop ep_reward = ", ep_rewards)
                # Re-initialize BATCH
                ep_actions = [[] for _ in range(n_agent)]
                ep_rewards = [[] for _ in range(n_agent)]
                ep_states = [[] for _ in range(n_agent)]

        mean_reward_all_player.append(np.mean(reward_for_episode_for_all))

        print(i_episode)
        print(score / max_steps)
        print(su)
        uti = np.array(su) / max_steps

    return  mean_reward_all_player
# T = number of steps before updating the sub-policy and get another one
T = 25
totalTime = 0
# Coefficient for calculating reward (see course - long term reward has less cost than early term reward)
GAMMA = 0.98
n_episode = 1000 #100000
max_steps = 1000
i_episode = 0
# number of action
n_actions = 5
# number of sub policies
# Print plot
render = False


nb_simulation = 5

array_mean_reward_by_agent = []
array_mean_reward_all_agent = []

for i in range(nb_simulation):
    print("simulation ",i)
    mean_reward_all_player = run_fen_wo_hierarchy(T, totalTime, GAMMA, n_episode, max_steps, i_episode, n_actions, render)
    print(mean_reward_all_player)
    array_mean_reward_all_agent.append(mean_reward_all_player)

print("\ntot=\n",array_mean_reward_all_agent )

#with open("S1_wo_hier_output_array_by_agent.csv", "w", newline='') as f:
#    writer = csv.writer(f)
#    writer.writerows(np.mean(array_mean_reward_by_agent, axis=0))
with open("S_wo_hier_output_array_all_agent.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(array_mean_reward_all_agent)


end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("time elapsed {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


#plot_reward(array_mean_reward_all_agent, nb_simulation)