import numpy as np
from collections import deque
import random

from keras.layers import Dense, Input, Lambda, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, RMSprop
import keras.backend as K
import tensorflow as tf

from SumTree import SumTree


def huber_loss(y_true, y_pred):
    HUBER_LOSS_DELTA = 2.0
    err = y_true - y_pred
    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)


class PERMemory:  # stored as ( s, a, r, s_ ) in SumTree

    def __init__(self, capacity, a):
        self.tree = SumTree(capacity)
        self.e = 0.01
        self.a = a
        self.entries = 0
        self.capacity = capacity

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def save(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

        if self.entries < self.capacity:
            self.entries += 1

    def sample(self, n):
        indexes, priorities, batch = self.tree.sample(n)
        return indexes, priorities, batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


class Memory:
    def __init__(self, capacity):
        self.samples = deque(maxlen=capacity)

    def save(self,obs):
        self.samples.append(obs)

    def sample(self,n):
        idx = None
        prio = None
        batch = random.sample(self.samples, n)
        return idx, prio, batch

    def size(self):
        return len(self.samples)


class LinearAnneal:
    def __init__(self, start, end, steps, down=True):
        self.start = start
        self.end = end
        self.down = down

        if down:
            self.stepsize = (start-end)/steps
        else:
            self.stepsize = (end-start)/steps


    def __call__(self, step, *args, **kwargs):
        if self.down:
            return max(self.end, self.start - step*self.stepsize)
        else:
            return min(self.end, self.start + step*self.stepsize)


def MLP(input_shape, layers, n_actions, activation='relu', optimizer='rmsprop',
        lr=0.01,loss='huber', dueling=False):

    if activation == 'lrelu':
        activation = LeakyReLU(alpha=0.01)

    x = Input(shape=input_shape)
    h = Dense(layers[0], activation=activation)(x)
    for layer_size in layers[1:]:
        h = Dense(layer_size, activation=activation)(h)

    if dueling:
        y = Dense(n_actions + 1, activation='linear')(h)
        output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                   output_shape=(n_actions,))(y)
    else:
        output = Dense(n_actions, activation='linear')(h)

    model = Model(inputs=x, outputs=output)

    if optimizer == 'adam':
        opt = Adam(lr, clipvalue=10)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr, clipvalue=10)

    if loss == 'mse':
        loss = 'mse'
    elif loss == 'huber':
        loss = huber_loss

    model.compile(optimizer=opt, loss=loss)

    return model


class DQNAgent:

    def __init__(self, input_shape, action_space, layers=[64], activation='relu', start_eps=1,
                 end_eps=0.01, eps_steps=10000, optimizer='rmsprop',loss='huber', lr=0.01, mem_capacity=10000,
                 batch_size=64, gamma=0.99, double_dqn=False, update_target_network_freq=None,
                 per_prio=False, per_alpa=0.1, dueling=False, warmup=10, per_is=False, per_beta=0.1,
                 per_beta_steps=5000):

        self.epsilon = LinearAnneal(start_eps, end_eps, steps=eps_steps)
        self.gamma = gamma

        self.action_space = action_space
        self.input_shape = input_shape
        self.steps = 0
        self.warmup_steps = 0
        self.warmup = warmup
        self.batch_size = batch_size

        self.model = MLP(input_shape, layers, action_space.n, activation,
                         optimizer, lr, loss, dueling)

        self.double_dqn = double_dqn
        self.update_target_network_freq = update_target_network_freq
        if self.double_dqn:
            self.target_model = MLP(input_shape, layers, action_space.n, activation,
                         optimizer, lr, loss, dueling)

        self.per_prio = per_prio
        if self.per_prio:
            self.memory = PERMemory(mem_capacity, per_alpa)
        else:
            self.memory = Memory(capacity=mem_capacity)

        self.per_is = per_is
        self.per_beta = LinearAnneal(per_beta,1.0,per_beta_steps, down=False)

    def act(self,state):
        if np.random.random() < self.epsilon(self.steps):
            return 0, self.action_space.sample()
        else:
            pred = self.model.predict(state)
            a = np.argmax(pred)
            q = pred.max()
            return q, a

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def store(self,obs):

        if self.per_prio:

            if self.warmup_steps < self.warmup:
                r = obs[2]
                self.memory.save(abs(r),obs)
            else:
                _,_,errors = self.get_train_data([obs])
                self.memory.save(errors[0], obs)
        else:
            self.memory.save(obs)

        if self.warmup_steps < self.warmup:
            self.warmup_steps += 1
        else:
            self.steps += 1

        if self.double_dqn:
            if self.steps % self.update_target_network_freq == 0:
                self.update_target_network()

       #print(self.steps, self.epsilon(self.steps))

    def get_train_data(self, batch):

        errors = np.zeros(len(batch))

        states = np.array([obs[0] for obs in batch]).reshape(-1, self.input_shape[0])
        states_ = np.array([obs[3] for obs in batch]).reshape(-1, self.input_shape[0])

        p = self.model.predict(states)
        p_ = self.model.predict(states_)

        if self.double_dqn:
            p_target = self.target_model.predict(states_)
        else:
            p_target = p_

        for i in range(len(batch)):
            obs = batch[i]
            s, a, r, s_, done = obs

            old = p[i][a]

            if done:
                p[i][a] = r
            else:
                p[i][a] = r + self.gamma * p_target[i][np.argmax(p_[i])]

            errors[i] = abs( old - p[i][a])

        return states, p, errors

    def replay(self):

        indexes, priorities, batch = self.memory.sample(self.batch_size)

        if self.per_is:
            sampling_probabilities = priorities / self.memory.tree.total()
            w = np.power(self.memory.entries * sampling_probabilities, -self.per_beta(self.steps))
            w = w / w.max()

        states, p, errors = self.get_train_data(batch)

        if self.per_prio:
            for i in range(len(batch)):
                idx = indexes[i]
                self.memory.update(idx, errors[i])

        if self.per_is:
            hist = self.model.fit(states, p, sample_weight=w,batch_size=self.batch_size, epochs=1, verbose=0)
        else:
            hist = self.model.fit(states, p, batch_size=self.batch_size, epochs=1, verbose=0)

        return hist.history['loss'][0]


def train(agent,env, train_steps, phi, train_freq, warmup, print_ep_freq):

    episode_rewards = []
    episode_train_step = []
    q_values = []

    loss = 0
    steps = 0
    episode = 0

    while steps < train_steps:
        episode_reward = 0
        episode_step = 0
        q_episode = 0

        s = env.reset()
        s = phi(s)
        d = False
        while not d:
            q, a = agent.act(s)
            s_, r, d, _ = env.step(a)
            s_ = phi(s_)

            agent.store([s, a, r, s_, d])
            s = s_

            episode_reward += r
            episode_step += 1

            steps += 1
            q_episode += q
            q_values.append(q)

            if steps % train_freq == 0 and steps > warmup:
                loss = agent.replay()

        episode_rewards.append(episode_reward)
        episode_train_step.append(steps)


        episode += 1
        if episode % print_ep_freq == 0:
            mess = 'Ep:{} Stps:{}/{} R:{:.2f} Ep_stps:{} Eps:{:.2f} Per_B:{:.2f} Q:{:.2f} Loss:{:.2e}'
            print(mess.format(episode, steps, train_steps, episode_reward,
                              episode_step, agent.epsilon(agent.steps),agent.per_beta(agent.steps),q_episode,loss))

    return q_values, episode_rewards, episode_train_step
