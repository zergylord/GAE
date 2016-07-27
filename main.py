import tensorflow as tf
import numpy as np
import time
import gym
from gae.agent import *
class say_zero():
    state = np.random.randn(1)
    def reset(self):
        return self.state
    def render(self):
        pass
    def step(self,a):
        #reward = -abs(a[0])
        target = self.state[0]
        reward = -abs(a[0])
        self.state = np.random.randn(1)
        return self.state,reward,False,False
#env = say_zero()
env = gym.make('Pendulum-v0')
sess = tf.Session()
#tensor board stuff------
flags = tf.app.flags
FLAGS = flags.FLAGS
dir_path = '/tmp/gae_logs'
if len(sys.argv) > 1:
    print('custom directory')
    dir_path += '/'+sys.argv[1]
flags.DEFINE_string('summary_dir',dir_path, 'Summaries directory')
if tf.gfile.Exists(FLAGS.summary_dir):
    tf.gfile.DeleteRecursively(FLAGS.summary_dir)
tf.gfile.MakeDirs(FLAGS.summary_dir)

writer = tf.train.SummaryWriter(FLAGS.summary_dir,sess.graph)
agent = GAE_Agent(env,sess,writer) 
agent.train()


