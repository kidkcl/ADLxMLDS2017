from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        # epsilon parameters
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps

        # training paramters
        self.EPISODES = 100000
        self.BATCH_SIZE = 32
        self.TRAIN_START = 50000
        self.UPDATE_TARGET_MODEL = 10000
        self.NO_OPERATION = 10
        self.DISCOUNT_FACTOR = 0.99

        self.memory = deque(maxlen=200000)

        # build model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()
        self.sess = tf.InteractiveSession(config=config)
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.sess.run(tf.global_variables_initializer())

        if args.test_dqn:
            print('loading trained model')
            self.model.load_weights('./save_model/breakout_dqn.39000.h5')

        # TODO
        # continue previous training
        #self.model.load_weights('./breakout_dqn.39000.h5')
        #self.epsilon = 0.09999910000442391

    def init_game_setting(self):
        pass

    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    def train(self):
        global_step = 0
        state = self.env.reset()

        # TODO
        #global_step = 2431224
        for episode in range(38500, self.EPISODES):
            done = False
            step, score = 0, 0

            state = self.env.reset()
            # random action at the start of the episode
            for _ in range(random.randint(1, self.NO_OPERATION)):
                state, _, _, _ = self.env.step(0)

            while not done:
                global_step += 1
                step += 1

                action = self.make_action(state, test = False)
                next_state, reward, done, info = self.env.step(action)

                self.avg_q_max += np.amax(self.model.predict(np.asarray([state])) [0])

                # save the sample <s, a, r, s'> to the replay memory
                self.memory.append((state, action, reward, next_state, done))
                # every some time interval, train the model
                self.train_replay()
                # update the target model with model
                if global_step % self.UPDATE_TARGET_MODEL == 0:
                    self.update_target_model()

                score += reward
                state = next_state

                if done:
                    print("episode,",episode,",score,",score,
                          ",memory length,",len(self.memory),",epsilon,",self.epsilon,
                          ",global_step,",global_step,",avg_q,",self.avg_q_max/float(step),
                          ",avg_loss,",self.avg_loss/float(step))
                    self.avg_q_max, self.avg_loss = 0, 0

            if episode % 1000 == 0:
                self.model.save_weights("./breakout_dqn." + str(episode) + ".h5")

    def make_action(self, observation, test=True):
        if np.random.rand() <= self.epsilon and test == False:
            return self.env.get_random_action()
        else:
            q_value = self.model.predict(np.asarray([observation]))
            return np.argmax(q_value[0])

    def train_replay(self):
        if len(self.memory) < self.TRAIN_START or len(self.memory) < self.BATCH_SIZE:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.BATCH_SIZE)

        target = np.zeros((self.BATCH_SIZE));
        state, next_state = [], []
        action, reward, done = [], [], []
        for i in range(self.BATCH_SIZE):
            state.append(mini_batch[i][0])
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            next_state.append(mini_batch[i][3])
            done.append(mini_batch[i][4])

        target_value = self.target_model.predict(np.asarray(next_state))

        # From target model
        for i in range(self.BATCH_SIZE):
            if done[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.DISCOUNT_FACTOR * np.amax(target_value[i])

        loss = self.optimizer([state, action, target])
        self.avg_loss += loss[0]
