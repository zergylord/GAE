import tensorflow as tf
import numpy as np
def linear(in_,out_dim,name,activation_fn=None,bias=True):
    in_dim = in_.get_shape()[1]
    with tf.variable_scope(name):
        W = tf.get_variable('W',[in_dim,out_dim],tf.float32)
        out = tf.matmul(in_,W)
        if bias:
            b = tf.get_variable('b',[out_dim],tf.float32)
            out = out + b
        if activation_fn != None:
            return activation_fn(out)
        else:
            return out
eps = 1e-8
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
