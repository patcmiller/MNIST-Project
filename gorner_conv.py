import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import math
import numpy as np
import sys,time
import matplotlib.pyplot as plt
plt.ion()
start_time=time.time()
tf.set_random_seed(0)

def print_acc(b,bsize,nbatches,sess):
    a,c= 0.0,0.0
    for i in range(0,nbatches,bsize):
        a+= accuracy.eval(session=sess, feed_dict={X: mnist.test.images[i:i+bsize], Y: mnist.test.labels[i:i+bsize], pkeep: 1.0})
        c+= cross_entropy.eval(session=sess, feed_dict={X: mnist.test.images[i:i+bsize], Y: mnist.test.labels[i:i+bsize], pkeep: 1.0})
    print("{0:05d}".format(b) + ": accuracy: " + "{0:.4f}".format(a/(nbatches/float(bsize))) + " loss: " + "{0:5.3f}".format(c/(nbatches/float(bsize))) + " (lr:" + "{0:.5f}".format(learn_rate) + ")")

mnist= read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
L1, L2= 4, 8 # LAYER SIZES
L3, L4, L5= 12, 24, 200

# WEIGHTS AND BIASES
W1= tf.Variable(tf.truncated_normal([7,7,1,L1], dtype=tf.float32, stddev=0.1))
B1= tf.Variable(tf.constant(0.1, tf.float32, [L1]))
W2= tf.Variable(tf.truncated_normal([6,6,L1,L2], dtype=tf.float32, stddev=0.1))
B2= tf.Variable(tf.constant(0.1, tf.float32, [L2]))
W3= tf.Variable(tf.truncated_normal([5,5,L2,L3], dtype=tf.float32, stddev=0.1))
B3= tf.Variable(tf.constant(0.1, tf.float32, [L3]))
W4= tf.Variable(tf.truncated_normal([4,4,L3,L4], dtype=tf.float32, stddev=0.1))
B4= tf.Variable(tf.constant(0.1, tf.float32, [L4]))
W5= tf.Variable(tf.truncated_normal([4*4*L4,L5], dtype=tf.float32, stddev=0.1))
B5= tf.Variable(tf.constant(0.1, tf.float32, [L5]))
W6= tf.Variable(tf.truncated_normal([L5,10], dtype=tf.float32, stddev=0.1))
B6= tf.Variable(tf.constant(0.1, tf.float32, [10]))

# PLACEHOLDERS
X= tf.placeholder(tf.float32, [None, 28, 28, 1])
Y= tf.placeholder(tf.float32,[None,10])
lrate= tf.placeholder(tf.float32)
pkeep= tf.placeholder(tf.float32)

# CREATE MODEL
stride= 1 # 28x28
Y1= tf.nn.relu(tf.nn.conv2d(X,W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
# Y1= tf.nn.dropout(Yt, pkeep)
stride= 2 # 14x14
Y2= tf.nn.relu(tf.nn.conv2d(Y1,W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
# Y2= tf.nn.dropout(Yt, pkeep)
stride= 2 # 7x7
Y3= tf.nn.relu(tf.nn.conv2d(Y2,W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
# Y3= tf.nn.dropout(Yt, pkeep)
stride= 2 # 4x4
Y4= tf.nn.relu(tf.nn.conv2d(Y3,W4, strides=[1, stride, stride, 1], padding='SAME') + B4)
# Y4= tf.nn.dropout(Yt, pkeep)
YY= tf.reshape(Y4, shape=[-1,4*4*L4])
Y5= tf.nn.relu(tf.matmul(YY,W5)+B5)
YY5 = tf.nn.dropout(Y5, pkeep)
Ylogits= tf.matmul(YY5,W6)+B6
Y6= tf.nn.softmax(Ylogits)

bsize= 100
cross_entropy= tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y)
cross_entropy= tf.reduce_mean(cross_entropy)*bsize

is_correct= tf.equal(tf.argmax(Y6,1),tf.argmax(Y,1))
accuracy= tf.reduce_mean(tf.cast(is_correct,tf.float32))

max_learn_rate= 0.003
min_learn_rate= 0.0001
decay_speed= 5000.0
optimizer= tf.train.GradientDescentOptimizer(0.003)
train_step= optimizer.minimize(cross_entropy)

init= tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)

for i in range(1,10000+1):
    batch_X,batch_Y= mnist.train.next_batch(bsize)
    learn_rate= min_learn_rate + (max_learn_rate-min_learn_rate)* math.exp(-i/decay_speed)
    
    run_data= {X: batch_X, Y: batch_Y, lrate: learn_rate, pkeep: 0.7}
    sess.run(train_step, feed_dict=run_data)
        
    if i%200==0:
        train_data={X: batch_X, Y: batch_Y, pkeep: 1.0}
        a,c= sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print("{0:05d}".format(i) + ": accuracy: " + "{0:.4f}".format(a) + " loss: " + "{0:5.3f}".format(c) + " (lr:" + "{0:.5f}".format(learn_rate) + ")")
        print_acc(i,5000, mnist.test.images.shape[0],sess)
        print()       
        
print("--- %s minutes ---" % ((time.time() - start_time)/60.0))    

    