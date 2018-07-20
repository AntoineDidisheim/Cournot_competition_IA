import tensorflow as tf
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class Player:
    def __init__(self, game, id_player="auto", action_range=10, end_of_random_action_time=1000., start_production = 50,random_action_range=10):
        self.id = id_player
        self.end_of_random_action_time = end_of_random_action_time
        self.production = start_production

        self.epsilon = 0.9
        self.gamma = 0.000
        self.action_range = action_range
        self.random_action_range = random_action_range
        self.current_action=0

        self.history_prediction_profits = np.empty([2*(self.action_range-1)+1])
        self.history_prediction_production = np.empty([2*(self.action_range-1)+1])

        self.input_dim = len(game.get_lag_history()) + 1
        self.history_input = np.append(game.get_lag_history(), 0).reshape(1, self.input_dim)
        self.history_rewards = np.array([0])
        self.first_round = True

    def update_q(self, state, reward, next_state):
        # here we don't add the action to production because it is already done in sim
        prod = np.array([self.production])-50
        new_input = np.append(prod, state).reshape(1,self.input_dim)
        new_reward = np.array([reward])
        #save the new output
        self.update_history(new_input,new_reward)
        self.training_procedure(training_input=new_input, training_reward=new_reward, next_state=next_state)

    def training_procedure(self, training_input, training_reward, next_state):
        print("no training procedure defined")

    def select_action(self, current_state, step, verbose= False):
        if step > self.end_of_random_action_time:
            self.random_action_range =1
        threshold = max(self.epsilon, step / self.end_of_random_action_time)
        # forces the random action to disapeared aftor a fixed number of step
        if random.random() < self.epsilon:
            # Exploit best option with probability epsilon
            action = 0
            expected_gain = -1000000000000000000000000000000000

            all_prod_potential = np.empty([2 * (self.action_range-1) + 1])
            all_pred_potential = np.empty([2*(self.action_range-1)+1])
            k=-1
            for i in np.append(np.arange(self.action_range), -np.arange(self.action_range - 1) - 1):
                k=k+1 # for the history
                potential_production = np.array([max(self.production+i,0)])
                temp_input = np.append(potential_production-50,current_state)
                # predicted_profits = self.model.predict(temp_input.reshape(1,self.input_dim))
                predicted_profits = self.make_prediction(X=temp_input.reshape(1,self.input_dim))
                all_prod_potential[k] = max(0,self.production+i)-50
                all_pred_potential[k] = predicted_profits
                if predicted_profits>expected_gain:
                    expected_gain=predicted_profits
                    action=i
                    self.current_action = i
        #     finaly we save the history of preds:
            self.history_prediction_production = np.vstack([self.history_prediction_production, all_prod_potential])
            self.history_prediction_profits = np.vstack([self.history_prediction_profits, all_pred_potential])
            if verbose:
                print("prod:",all_prod_potential)
                print("pred:",all_pred_potential)
                v = np.append(np.arange(self.action_range), -np.arange(self.action_range - 1) - 1)
                print(action, v)
        else:
            # Explore random option with probability 1 - epsilon
            action = random.randint(-self.random_action_range,self.random_action_range)
            self.current_action = action
        return action

    def make_prediction(self,X):
        pred = 0
        return pred

    def update_history(self, new_input, new_reward):
        self.history_input = np.vstack([self.history_input, new_input])
        self.history_rewards = np.append(self.history_rewards, new_reward)
        # if its first round we remove the template
        if self.first_round:
            self.history_input = self.history_input[1:, :]
            self.history_rewards = self.history_rewards[1:]
            self.first_round = False


class PlayerNNet(Player):
    def __init__(self, game, id_player="auto", action_range=10, end_of_random_action_time=1000., start_production = 50,layer_size=200,random_action_range=10):
        Player.__init__(self, game, id_player, action_range, end_of_random_action_time, start_production,random_action_range=random_action_range)

        # defining a NNet
        h1_dim = layer_size
        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.y = tf.placeholder(tf.float32, [None, 1])
        W1 = tf.Variable(tf.random_normal([self.input_dim, h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h1 = tf.nn.sigmoid(tf.matmul(self.x, W1) + b1)
        W2 = tf.Variable(tf.random_normal([h1_dim, h1_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
        W3 = tf.Variable(tf.random_normal([h1_dim, 1]))
        b3 = tf.Variable(tf.constant(0.01, shape=[1]))
        self.q = tf.add(tf.matmul(h2, W3), b3)

        loss = tf.square(self.y - self.q)
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def make_prediction(self, X):
        predicted_profits = self.sess.run(self.q, feed_dict={self.x: X})
        return predicted_profits

    def training_procedure(self, training_input, training_reward, next_state):
        training_reward = training_reward.reshape(1,1)

        # if self.history_input.shape[0]>10:
        #     in_ = self.history_input
        #     out_ = self.history_rewards.reshape(len(self.history_rewards),1)
        #     self.sess.run(self.train_op, feed_dict={self.x: in_, self.y: out_})
        for i in range(10):
            self.sess.run(self.train_op, feed_dict={self.x: training_input, self.y: training_reward})


class PlayerRndForrest(Player):
    def __init__(self, game, id_player="auto", action_range=10, end_of_random_action_time=1000., start_production = 50,layer_size=200,random_action_range=10):
        Player.__init__(self, game, id_player, action_range, end_of_random_action_time, start_production,random_action_range=random_action_range)

        self.model = RandomForestRegressor(n_estimators=10)
        # init the fit
        self.model.fit(X=self.history_input,y=self.history_rewards)

    def make_prediction(self,X):
        predicted_profits = self.model.predict(X)
        return predicted_profits

    def training_procedure(self, training_input, training_reward, next_state):
        self.model.fit(X=self.history_input, y=self.history_rewards)

