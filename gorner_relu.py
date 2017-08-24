import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import math
import numpy as np
import sys,time
import matplotlib.pyplot as plt
plt.ion()
start_time=time.time()
tf.set_random_seed(0)

mnist= read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
L1, L2= 200, 100 # LAYER SIZES
L3, L4, L5= 60, 30, 10

# WEIGHTS AND BIASES
W1= tf.Variable(tf.truncated_normal([784,L1], dtype=tf.float64, stddev=0.1))
B1= tf.Variable(tf.constant(0.1, tf.float64, [L1]))
W2= tf.Variable(tf.truncated_normal([L1,L2], dtype=tf.float64, stddev=0.1))
B2= tf.Variable(tf.constant(0.1, tf.float64, [L2]))
W3= tf.Variable(tf.truncated_normal([L2,L3], dtype=tf.float64, stddev=0.1))
B3= tf.Variable(tf.constant(0.1, tf.float64, [L3]))
W4= tf.Variable(tf.truncated_normal([L3,L4], dtype=tf.float64, stddev=0.1))
B4= tf.Variable(tf.constant(0.1, tf.float64, [L4]))
W5= tf.Variable(tf.truncated_normal([L4,L5], dtype=tf.float64, stddev=0.1))
B5= tf.Variable(tf.constant(0.1, tf.float64, [L5]))

# CREATE MODEL
X= tf.placeholder(tf.float64, [None, 28, 28, 1])
X= tf.reshape(X,[-1,28*28])

lrate= tf.placeholder(tf.float64)
pkeep= tf.placeholder(tf.float64)
Yt= tf.nn.relu(tf.matmul(X,W1)+B1)
Y1= tf.nn.dropout(Yt, pkeep)
Yt= tf.nn.relu(tf.matmul(Y1,W2)+B2)
Y2= tf.nn.dropout(Yt, pkeep)
Yt= tf.nn.relu(tf.matmul(Y2,W3)+B3)
Y3= tf.nn.dropout(Yt, pkeep)
Yt= tf.nn.relu(tf.matmul(Y3,W4)+B4)
Y4= tf.nn.dropout(Yt, pkeep)
Y5= tf.nn.softmax(tf.matmul(Y4,W5)+B5)

Y= tf.placeholder(tf.float64,[None,10])

bsize= 100
cross_entropy= tf.nn.softmax_cross_entropy_with_logits(logits=Y5, labels=Y)
cross_entropy= tf.reduce_mean(cross_entropy)*bsize

is_correct= tf.equal(tf.argmax(Y5,1),tf.argmax(Y,1))
accuracy= tf.reduce_mean(tf.cast(is_correct,tf.float64))

max_learn_rate= 0.005
min_learn_rate= 0.0001
decay_speed= 5000.0
optimizer= tf.train.GradientDescentOptimizer(0.003)
train_step= optimizer.minimize(cross_entropy)

init= tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)

for i in range(1,10000+1):
    batch_X,batch_Y= mnist.train.next_batch(bsize)
    batch_X= np.reshape(batch_X,[bsize,28*28])
    batch_Y= np.reshape(batch_Y,[bsize,10])
    learn_rate= min_learn_rate + (max_learn_rate-min_learn_rate)* math.exp(-i/decay_speed)
    
    run_data= {X: batch_X, Y: batch_Y, pkeep: 0.75, lrate: learn_rate}
    sess.run(train_step, feed_dict=run_data)
        
    if i%1000==0:
        train_data={X: batch_X, Y: batch_Y, pkeep: 1.0}
        a,c= sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print(str(i) + ": accuracy:" + "{0:.3f}".format(a) + " loss: " + "{0:.3f}".format(c) + " (lr:" + "{0:.5f}".format(learn_rate) + ")")
    
        test_data= {X:np.reshape(mnist.test.images,[10000,784]), Y:np.reshape(mnist.test.labels,[10000,10]), pkeep: 1.0}
        a,c= sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print(str(i) + ": accuracy:" + "{0:.3f}".format(a) + " loss: " + "{0:.3f}".format(c) + " (lr:" + "{0:.5f}".format(learn_rate) + ")")

print("--- %s minutes ---" % ((time.time() - start_time)/60.0))    

    