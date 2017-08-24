import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import math
import numpy as np
import sys,time
import matplotlib.pyplot as plt
plt.ion()
start_time=time.time()
tf.set_random_seed(0)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg= tf.train.ExponentialMovingAverage(0.9999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon= 1e-7
    if convolutional: mean, variance= tf.nn.moments(Ylogits, [0, 1, 2])
    else: mean, variance= tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m= tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v= tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn= tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages
    
def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

def print_acc(b,bsize,nbatches,sess):
    a,c= 0.0,0.0
    for i in range(0,nbatches,bsize):
        a+= accuracy.eval(session=sess, feed_dict={X: mnist.test.images[i:i+bsize], Y: mnist.test.labels[i:i+bsize], is_test: False, pkeep: 1.0, pkeep2: 1.0})
        c+= cross_entropy.eval(session=sess, feed_dict={X: mnist.test.images[i:i+bsize], Y: mnist.test.labels[i:i+bsize], is_test: False, pkeep: 1.0, pkeep2: 1.0})
    print("{0:05d}".format(b) + ": accuracy: " + "{0:.4f}".format(a/(nbatches/float(bsize))) + " loss: " + "{0:5.3f}".format(c/(nbatches/float(bsize))) + " (lr:" + "{0:.5f}".format(learn_rate) + ")")

mnist= read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
L1, L2= 24, 48 # LAYER SIZES
L3, L4, L5= 62, 86, 200

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
B6= tf.Variable(tf.constant(0.0, tf.float32, [10]))

# PLACEHOLDERS
X= tf.placeholder(tf.float32, [None, 28, 28, 1])
Y= tf.placeholder(tf.float32,[None,10])
is_test= tf.placeholder(tf.bool)
iteration= tf.placeholder(tf.int32)
lrate= tf.placeholder(tf.float32)
pkeep= tf.placeholder(tf.float32)
pkeep2= tf.placeholder(tf.float32)

# CREATE MODEL
stride= 1 # 28x28
Y1cv= tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1bn, ema1= batchnorm(Y1cv, is_test, iteration, B1, convolutional=True)
Y1ru= tf.nn.relu(Y1bn)

stride= 2 # 14x14
Y2cv= tf.nn.conv2d(Y1ru, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2bn, ema2= batchnorm(Y2cv, is_test, iteration, B2, convolutional=True)
Y2ru= tf.nn.relu(Y2bn)

stride= 2 # 7x7
Y3cv= tf.nn.conv2d(Y2ru, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3bn, ema3= batchnorm(Y3cv, is_test, iteration, B3, convolutional=True)
Y3ru= tf.nn.relu(Y3bn)

stride= 2 # 4x4
Y4cv= tf.nn.conv2d(Y3ru, W4, strides=[1, stride, stride, 1], padding='SAME')
Y4bn, ema4= batchnorm(Y4cv, is_test, iteration, B4, convolutional=True)
Y4ru= tf.nn.relu(Y4bn)
Y4= tf.nn.dropout(Y4ru, pkeep2, compatible_convolutional_noise_shape(Y4ru))

YY= tf.reshape(Y4, shape=[-1,4*4*L4])
Y5m= tf.matmul(YY,W5)
Y5bn, ema5= batchnorm(Y5m, is_test, iteration, B5)
Y5ru= tf.nn.relu(Y5bn)
Y5= tf.nn.dropout(Y5ru, pkeep)
Ylogits= tf.matmul(Y5,W6)+B6
Y6= tf.nn.softmax(Ylogits)
update_ema= tf.group(ema1,ema2,ema3,ema4,ema5)

bsize= 100
cross_entropy= tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y)
cross_entropy= tf.reduce_mean(cross_entropy)*bsize

is_correct= tf.equal(tf.argmax(Y6,1),tf.argmax(Y,1))
accuracy= tf.reduce_mean(tf.cast(is_correct,tf.float32))

max_learn_rate= 0.03
min_learn_rate= 0.0001
decay_speed= 2000.0
train_step= tf.train.AdamOptimizer(lrate).minimize(cross_entropy)

init= tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)

for i in range(1,10000+1):
    batch_X,batch_Y= mnist.train.next_batch(bsize)
    learn_rate= min_learn_rate + (max_learn_rate-min_learn_rate)* math.exp(-i/decay_speed)
    
    run_data= {X: batch_X, Y: batch_Y, lrate: learn_rate, iteration: i, is_test: False, pkeep: 0.8, pkeep2: 0.95}
    sess.run(train_step, feed_dict=run_data)
    sess.run(update_ema, feed_dict=run_data)
        
    if i%200==0:
        train_data={X: batch_X, Y: batch_Y, is_test: False,  pkeep: 1.0, pkeep2: 1.0}
        a,c= sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print("{0:05d}".format(i) + ": accuracy: " + "{0:.4f}".format(a) + " loss: " + "{0:5.3f}".format(c) + " (lr:" + "{0:.5f}".format(learn_rate) + ")")
        print_acc(i,5000, mnist.test.images.shape[0],sess)
        print()       
        
print("--- %s minutes ---" % ((time.time() - start_time)/60.0))    

    