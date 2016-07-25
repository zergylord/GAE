import tensorflow as tf
import numpy as np
eps = 1e-10
def linear(in_,out_dim,name,activation_fn=None,bias=True,bias_value=None):
    in_dim = in_.get_shape()[1]
    with tf.variable_scope(name):
        W = tf.get_variable('W',[in_dim,out_dim],tf.float32)
        #W = tf.random_uniform([in_dim,out_dim], minval=-0.05, maxval=0.05)
        out = tf.matmul(in_,W)
        if bias:
            b = tf.get_variable('b',[out_dim],tf.float32)
            #b = tf.constant(0.01, shape=[out_dim])
            if bias_value != None:
                print('manual bias')
                b = b.assign(tf.constant(bias_value,shape=[out_dim]))
            out = out + b
        if activation_fn != None:
            '''
            mean,variance = tf.nn.moments(out,[0,1])
            beta = tf.Variable(tf.constant(0.0,shape=[out_dim]))
            gamma = tf.Variable(tf.constant(1.0,shape=[out_dim]))
            out = tf.nn.batch_normalization(out,mean,variance,beta,gamma,eps)
            '''
            out = activation_fn(out)
    return out
def norm_pdf(x,mu=0.0,sigma=1.0):
    #var = tf.square(sigma)
    var = tf.clip_by_value(tf.square(sigma),eps,float('inf'))
    return tf.rsqrt(2.0*np.pi*var)*tf.exp(-tf.square(x-mu)/(2.0*var))
    
def compute_return(rewards,gamma):
    length = len(rewards)
    R = np.zeros((length,))
    for t in reversed(range(length)):
        R[:t+1] = R[:t+1]*gamma + rewards[t]
    return list(R)
