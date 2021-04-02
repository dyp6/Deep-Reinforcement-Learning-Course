# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 23:08:38 2020

@author: postd
"""

from maml_env import HalfCheetahDirecBulletEnv
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.specs import array_spec
import tensorflow_probability as tfp
from tf_agents.policies import greedy_policy
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import actor_policy
import tensorflow.keras.losses as kls
from tf_agents.metrics import tf_metrics
keras_backend.set_floatx('float32')
import pandas as pd

import random
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from tf_agents.networks import network

class policyNet(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden1 = keras.layers.Dense(40, activation="relu",input_shape=(1,),name = "Inner Net Input")
        self.hidden2 = keras.layers.Dense(40, activation = "relu", name = "Inner Net Hidden")
        self.out = keras.layers.Dense(6,activation="tanh", name = "Inner Net Ouput")
        
    def call(self, x):
        output = self.hidden1(x)
        output = self.hidden2(output)
        output = self.out(output)
        return output
    
class ActionNet(network.Network):

    def __init__(self, input_tensor_spec, output_tensor_spec):
        super(ActionNet, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),name='ActionNet')
        self._output_tensor_spec = output_tensor_spec
        self._sub_layers = [
            tf.keras.layers.Dense(30, activation = tf.nn.relu),
            tf.keras.layers.Dense(30,activation = tf.nn.relu),
            tf.keras.layers.Dense(
                6, activation=tf.nn.tanh),
    ]
    
    def call(self, observations, step_type=(),network_state=()):

        output = tf.cast(observations, dtype=tf.float32)
        
        for layer in self._sub_layers:
            output = layer(output)
        
        actions = tf.reshape(output, [-1] + self._output_tensor_spec.shape.as_list())
        return actions, network_state

class ValueNet(network.Network):

    def __init__(self, input_tensor_spec, output_tensor_spec):
        super(ValueNet, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),name='ValueNet')
        self._output_tensor_spec = output_tensor_spec
        self._sub_layers = [
            tf.keras.layers.Dense(26, activation = tf.nn.relu),
            tf.keras.layers.Dense(30,activation = tf.nn.relu),
            tf.keras.layers.Dense(1,activation=None),
    ]
    
    def call(self, observations, step_type=(),network_state=()):

        output = tf.cast(observations, dtype=tf.float32)
        
        for layer in self._sub_layers:
            output = layer(output)
        
        values = tf.reshape(output, [-1] + self._output_tensor_spec.shape.as_list())
        return values, network_state
    
class ActionDistributionNet(ActionNet):
    def call(self, observations):
        action_means, network_state = super(ActionDistributionNet, self).call(
                observations)

        action_std = tf.Variable(tf.ones_like(action_means),dtype=tf.float32,name="Inner Sigma")
        return tfp.distributions.MultivariateNormalDiag(action_means, action_std), network_state
    
class agent():
    def __init__(self,input_tensor_spec,action_spec,value_specs):
        self.input_tensor_spec = input_tensor_spec
        self.action_spec = action_spec
        self.value_specs = value_specs
        self.policy = ActionDistributionNet(self.input_tensor_spec,self.action_spec)
        self.critic = ValueNet(self.input_tensor_spec,self.value_specs)
        self.a_opt = keras.optimizers.Adam(learning_rate=0.01)
        self.c_opt = keras.optimizers.Adam(learning_rate=0.01)
        self.clip_pram = 0.2
        
    def act(self,state):
        dist,_ = self.policy(state)
        action = dist.sample()
        mult = tf.constant(2.,dtype=tf.float32)
        sub = tf.constant(1.,dtype=tf.float32)
        action = tf.subtract(
                    tf.multiply(mult,
                        tf.divide(
                            tf.subtract(
                                action, 
                                tf.reduce_min(action)
                                ), 
                            tf.subtract(
                                tf.reduce_max(action), 
                                tf.reduce_min(action)
                                )
                            )
                        ),sub
                            
                    )
        return action
            
    def learn_pg(self,states,actions,rewards):
        with tf.GradientTape() as tape:
            dist , _ = self.policy(states)
            rewards = tf.reshape(rewards,(len(rewards),1))
            actions = tf.reshape(actions,(len(actions),6))
            logps = dist.log_prob(actions)
            loss = tf.math.negative(tf.reduce_sum(tf.math.multiply(rewards,logps)))
            
        grads = tape.gradient(loss,self.policy.trainable_variables)
        self.a_opt.apply_gradients(zip(grads,self.policy.trainable_variables))
        return loss
    
    def preprocess_ppo(self, states, actions, rewards, values):
        g = 0
        lmbda = 1
        returns = []
        new_vals = []
        for i in reversed(range(len(rewards))):
            if i+1 % 201 == 0:
                pass
            delta = rewards[i] + values[i + 1] - values[i]
            g = delta + lmbda * g
            returns.append(g + values[i])
            new_vals.append(values[i])

        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - new_vals
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        return states, actions, returns, adv
    
    def learn_ppo(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),1))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            dist, _ = self.policy(states)
            p = dist.log_prob(actions)
            entropy = tf.math.negative(tf.reduce_sum(tf.math.multiply(discnt_rewards,p)))
            sur1 = []
            sur2 = []
            for pb, t, op in zip(p, adv, old_probs):
                        t =  tf.constant(t)
                        op =  tf.constant(op)
                        ratio = tf.math.divide(pb,op)
                        s1 = tf.math.multiply(ratio,t)
                        s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
                        sur1.append(s1)
                        sur2.append(s2)
            sr1 = tf.stack(sur1)
            sr2 = tf.stack(sur2)
            v, _ = self.critic(states)
            v = tf.reshape(v, (len(v),1))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - c_loss + 0.001 * entropy)
            
        grads1 = tape1.gradient(a_loss, self.policy.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.policy.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss
    
    def actor_loss_ppo(self, probs, actions, adv, old_probs, closs):
        
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        sur1 = []
        sur2 = []
        
        for pb, t, op in zip(probability, adv, old_probs):
            t =  tf.constant(t)
            op =  tf.constant(op)
            ratio = tf.math.divide(pb,op)
            s1 = tf.math.multiply(ratio,t)
            s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        return loss
    
