import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from cournot import CournotCompetitionGame
training_size=1000
input_size = 1


# using our cournot competition we will create the random sample
game = CournotCompetitionGame(1)

for i in range(training_size):
    qty = np.array([np.random.randint(20,80)])
    # qty = np.array([np.random.randint(2,8)])*10
    game.get_profits(qty)
train_x = game.history_quantity
train_y = game.history_profits

# adding noise
sigma = 0 # you can add some noise to the y here to see the impact on the fit
train_y = train_y+np.random.normal(scale=sigma,size=(train_y.shape))




#just renaming the variables
output = train_y
input = train_x

# important, always standardize the input
input = (input-np.mean(input))/np.var(input)



# output=np.random.normal(size=(training_size,1))
# input=np.random.normal(size=(training_size,input_size))
h1_dim=200
number_of_layer = 2 # you can set he one or two layers

if number_of_layer==2:
    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32, [None, 1])
    W1 = tf.Variable(tf.random_normal([input_size, h1_dim]))
    b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
    h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1) # change here to relu
    W2 = tf.Variable(tf.random_normal([h1_dim, h1_dim]))
    b2 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2) #change here to sigmoid
    W3 = tf.Variable(tf.random_normal([h1_dim, 1]))
    b3 = tf.Variable(tf.constant(0.01, shape=[1]))
    q = tf.add(tf.matmul(h2, W3), b3)

if number_of_layer ==1:
    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32, [None, 1])
    W1 = tf.Variable(tf.random_normal([input_size, h1_dim]))
    b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
    h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1) # change here to relu
    W2 = tf.Variable(tf.random_normal([h1_dim, 1]))
    b2 = tf.Variable(tf.constant(0.1, shape=[1]))

    q = tf.add(tf.matmul(h1, W2), b2)



loss= tf.square(y - q)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

import time
start = time.time()
# for e epoch we will use the full training sample, train our nnet with and print the current estimation
for e in range(100):
    # for each sample in the training set we do a forward backward pass
    for i in range(training_size):
        out_ = np.asmatrix(np.array(output[i][0]))
        in_ = np.asmatrix(input[i])
        sess.run(train_op, feed_dict={x: in_, y: out_})
    # computing the current loss just to print it
    l = sess.run(loss,feed_dict={x:input, y: output})
    # finally we use the training set to make a forward pass and prediction
    pred = sess.run(q, feed_dict={x: input})
    # plt.plot(pred)
    plt.scatter(train_x,train_y)
    plt.scatter(train_x,pred)
    plt.title(e)
    plt.show()
    print(sum(l))

sess.close()