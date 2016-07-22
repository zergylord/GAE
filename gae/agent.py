import os
import tensorflow as tf
import numpy as np
import time
from .ops import *

class GAE_Agent(object):
    def __init__(self,env,sess,writer):     
        self.cur_time = time.clock()
        self.writer = writer
        self.x_dim = env.observation_space.shape[0]
        self.u_dim = 1
        self.hid_dim = 256
        self.sess = sess
        self.env = env
        self.gamma = .99
        self.build_model()
        self.merged = tf.merge_all_summaries()
        self.episodes_per_update = 20
        self.max_steps_per_episode = 1000
        self.reset_hist()
    def reset_hist(self):
        self.x_hist = []
        self.u_hist = []
        self.R_hist = []
        self.episode_count = 0
    def train(self):
        self.sess.run(tf.initialize_all_variables())
        self.total_updates = 0
        x = self.env.reset()
        rewards = []
        step = 0
        for i in range(int(1e8)):
            u = self.sample_action(x)
            self.x_hist.append(x)
            self.u_hist.append(u)
            x,reward,term,_ = self.env.step(u)
            step+=1
            rewards.append(reward)
            if term or step == self.max_steps_per_episode:
                x = self.env.reset() #only on term?
                self.R_hist = self.R_hist + compute_return(rewards,self.gamma)
                step = 0
                rewards = []
                self.episode_count+=1
                if self.episode_count == self.episodes_per_update:
                    feed_dict={}
                    feed_dict[self.x_] = np.asarray(self.x_hist)
                    feed_dict[self.u_] = np.asarray(self.u_hist)
                    feed_dict[self.R_] = np.expand_dims(np.asarray(self.R_hist),1)
                    _,summary_str = self.sess.run([self.train_step,self.merged],feed_dict=feed_dict)
                    self.writer.add_summary(summary_str,self.total_updates)
                    self.total_updates +=1
                    if self.total_updates % int(1e2) == 0:
                        print(self.total_updates,time.clock()-self.cur_time)
                        self.cur_time = time.clock()
                    self.reset_hist()

        
    def test(self):
        pass
    def sample_action(self,x):
        x = np.expand_dims(x,0)
        cur_mu = self.sess.run(self.mu,feed_dict={self.x_:x})
        cur_mu = np.squeeze(cur_mu,0)
        cur_sigma = 1.0 #network could learn this as well as mu
        return cur_mu+np.random.randn(self.u_dim)*cur_sigma
    def build_model(self):
        #inputs
        self.x_ = tf.placeholder('float32',[None,self.x_dim])
        self.u_ = tf.placeholder('float32',[None,self.u_dim])
        self.R_ = tf.placeholder('float32',[None,1])
        tf.histogram_summary('Returns',self.R_)
        tf.scalar_summary('avg Returns',tf.reduce_mean(self.R_))
        tf.histogram_summary('actions',self.u_)
        #network definition
        self.hid1 = linear(self.x_,self.hid_dim,'hid1',tf.nn.relu)
        self.last_hid = linear(self.hid1,self.hid_dim,'hid2')
        self.mu = linear(self.last_hid,self.u_dim,'mu')
        self.baseline = linear(self.last_hid,1,'baseline')
        self.log_p = tf.log(tf.clip_by_value(norm_pdf(self.u_,self.mu),eps,float('inf')))
        tf.histogram_summary('log p',self.log_p)

        self.b_loss = tf.reduce_mean(tf.square(self.R_-self.baseline))
        tf.scalar_summary('baseline error',self.b_loss)
        self.u_loss = tf.reduce_mean(self.log_p*(self.R_-tf.stop_gradient(self.baseline)))
        tf.scalar_summary('action prob * A',self.u_loss)
        self.net_loss = self.b_loss+self.u_loss
        tf.scalar_summary('loss',self.net_loss)
        #optimizer
        self.optim = tf.train.AdamOptimizer(1e-4)
        self.grads = self.optim.compute_gradients(self.net_loss)
        [tf.histogram_summary(v.name,g) if g is not None else '' for g,v in self.grads]
        self.train_step = self.optim.apply_gradients(self.grads)

            

        






