from agent_dir.agent import Agent
import scipy
import numpy as np
#import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Input
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config) 

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)

def preprocess_frame(I):
    I = I[35:195] # crop
    #I = I[::2, ::2, 0] # downsample by factor 2
    I = 0.2126 * I[::2, ::2, 0] + 0.7152 * I[::2, ::2, 1] + 0.0722 * I[::2, ::2, 2]
    I[I == 144] = 0 # remove bg
    I[I == 109] = 0
    #I[I != 0] = 1
    return np.expand_dims(I.astype(np.float),axis=0)

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.state_size = [1, 80, 80]
        self.action_size = env.get_action_space().n
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probabilities = []
        self.scores = []
        #self.model = self._build_model()
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.model = keras.models.load_model('./save_model/pong.h5')
        else:
           self.model = self._build_model()
           self.model.save('./save_model/pong.h5')
        self.model.summary()


    def _build_model(self):
        model = Sequential()
        model.add(Reshape((1,80,80),input_shape=(1,80,80))) # wrong
        # add more conv2d v0.1.1
        model.add(Conv2D(16, (8, 8), kernel_initializer="he_uniform", strides=(4,4), activation="relu", padding="same"))
        model.add(Conv2D(32, (4, 4), kernel_initializer="he_uniform", activation="relu", padding="same", strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, kernel_initializer="he_uniform", activation="relu"))
        #model.add(Dense(32, kernel_initializer="he_uniform", activation="relu"))
        print(self.action_size)
        model.add(Dense(self.action_size, activation='softmax'))
        optimizer = RMSprop(lr=1e-4, rho=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.prev_x = None

    def get_action(self, state):
        state = state.reshape([1, state.shape[0], state.shape[1], state.shape[2]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probabilities.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def pg_one(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.vstack([self.states])
        Y = self.probabilities + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probabilities, self.gradients, self.rewards = [], [], [], []

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        done = False
        running_reward = None
        episode = 0
        score = 0
        #self.model = keras.models.load_model('./save_model/pong.h5')
        prev_x = None
        state = self.env.reset()

        while True:
            cur_x = preprocess_frame(state)
            x = cur_x - prev_x if prev_x is not None else np.zeros([1, 80, 80])
            prev_x = cur_x

            action, prob = self.get_action(x)
            state, reward, done, info = self.env.step(action)
            score += reward
            self.remember(x, action, prob, reward)
            if(done):
                episode +=1
                self.pg_one()
                print('Episode: %d - Score: %f.' % (episode, score))
                running_reward = score if running_reward is None else running_reward * 0.99 + score * 0.01
                print('resetting env. running mean: %f' % (running_reward))
                state = self.env.reset()
                self.scores.append(score)
                score = 0
                prev_x = None
                self.model.save('./save_model/pong.h5')
                if running_reward > 3:
                    break
                #plt.figure()
                #plt.plot(range(episode), self.scores)
                #plt.savefig('pong.png')
                npy = np.asarray(self.scores)
                np.save('scores.npy', npy)

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        cur_x = preprocess_frame(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros([1, 80, 80])
        self.prev_x = cur_x
        #state = x.reshape([1, x.shape[0]])
        state = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
        action_prob = self.model.predict(state, batch_size=1).flatten()
        # self.probs.append(action_prob)
        prob = action_prob / np.sum(action_prob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action
        # return self.env.get_random_action()

