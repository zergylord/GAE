import tensorflow as tf
from .ops import *
class Baseline(object):
    def __init__(self,state):
        self.baseline = linear(state,1,'baseline')
        tf.scalar_summary('baseline',tf.reduce_mean(self.baseline))
    def get_advantage(self,R_):
        return R_-tf.stop_gradient(self.baseline)
    def get_loss(self,R_):
        tf.scalar_summary('baseline - R',tf.reduce_mean(R_-self.baseline))
        return tf.reduce_mean(tf.square(R_-self.baseline))
class TD_Residual(Baseline):
    def get_advantage(self,R_):
        pass
