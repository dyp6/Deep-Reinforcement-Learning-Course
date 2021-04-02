# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 23:47:45 2020

@author: postd
"""

from agent import agent
from agent import ActionDistributionNet
from agent import ValueNet
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

class Tasks:
    def __init__(self, *task_configs):
        self.tasks = [i for i in task_configs]

    def sample_tasks(self, batch_size):
        return random.choices(self.tasks, k=batch_size)

def copy_actor(policy, x, input_tensor_spec,action_spec):
    '''Copy model weights to a new model.
    
    Args:
        model: model to be copied.
        x: An input example. This is used to run
            a forward pass in order to add the weights of the graph
            as variables.
    Returns:
        A copy of the model.
    '''
    policy(tf.convert_to_tensor(x))
    copied_model_actor = ActionDistributionNet(input_tensor_spec,action_spec)
    # If we don't run this step the weights are not "initialized"
    # and the gradients will not be computed.
    copied_model_actor(tf.convert_to_tensor(x))
   
    copied_model_actor.set_weights(policy.get_weights())
    
    return copied_model_actor

def copy_critic(critic, x, input_tensor_spec,value_spec):
    '''Copy model weights to a new model.
    
    Args:
        model: model to be copied.
        x: An input example. This is used to run
            a forward pass in order to add the weights of the graph
            as variables.
    Returns:
        A copy of the model.
    '''
    critic(tf.convert_to_tensor(x))
    copied_model_critic = ValueNet(input_tensor_spec,value_spec)
    # If we don't run this step the weights are not "initialized"
    # and the gradients will not be computed.
    copied_model_critic(tf.convert_to_tensor(x))
   
    copied_model_critic.set_weights(critic.get_weights())
    
    return copied_model_critic

def main(args):
    # Sample one trajectory for task_i and collect the transitions (old_state,action,new_state,reward)
    tasks = Tasks(("Forward", True), ("Backward", False))
    task_config = tasks.sample_tasks(1)
    task_name, env_args = task_config[0], task_config[1:]
    env = HalfCheetahDirecBulletEnv(*env_args)
    input_tensor_spec = tensor_spec.TensorSpec((26,), tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec((6,),tf.float32,minimum=-1,maximum=1)
    value_spec = tensor_spec.TensorSpec((1,),tf.float32)
    # Instantiate initial agent with random policy theta
    outer_agent = agent(input_tensor_spec,action_spec,value_spec)
    # Update temp outer agent with trajectories sampled from each updated policy in env. t_i
    temp_outer_agent = agent(input_tensor_spec,action_spec,value_spec)
    K = 1
    meta_iterations = args.meta_iteration
    num_adapt_steps=args.num_adapt_steps
    # Sample tasks from task distribution to train policy on
    avg_meta_return=[]
    meta_iter_policies=[]
    meta_iter_critics=[]
    task_list = []
    for meta_iter in range(meta_iterations):
        # Update actual outer agent policy with temp outer agent after learning with theta`_i from each task
        if meta_iter_policies:
            outer_agent.critic= meta_iter_critics[-1]
            outer_agent.policy=meta_iter_policies[-1]
        avg_task_return = []
        for task_config in tasks.sample_tasks(args.meta_batch_size):
            task_name, env_args = task_config[0], task_config[1:]
            task_list.append(task_name)
            env = HalfCheetahDirecBulletEnv(*env_args)
            # Create a copy of the model to sample trajectories with new policy theta'
            actor_copy = copy_actor(outer_agent.policy,env.reset().reshape(1,26),
                                            input_tensor_spec,action_spec)
            # Create the inner agent to update the policy using the copy of the model made above
            inner_agent = agent(input_tensor_spec,action_spec,value_spec)
            inner_agent.policy=actor_copy
            if num_adapt_steps==1 | num_adapt_steps==2:
            # Sample K trajectories from outer policy
                Ss = []
                As = []
                Rs = []
                for _ in range(K):
                    state = env.reset()
                    h = 0 
                    # Sample K trajectories from initial policy on task_i
                    while True:
                        Ss.append(state)
                        action = outer_agent.act(state.reshape(1,26))
                        As.append(action)
                        state, reward, done, _ = env.step(action.numpy().flatten())
                        Rs.append(reward)
                        h += 1
                        if h == 200:
                            break
                 # Calculate the loss and update the inner agent
                task_loss = inner_agent.learn_pg(np.array(Ss,dtype=np.float32),As,np.array(Rs,dtype=np.float32))
            if num_adapt_steps==2:
                # Sample K trajectories from outer policy
                Ss = []
                As = []
                Rs = []
                for _ in range(K):
                    state = env.reset()
                    h = 0 
                    # Sample K trajectories from initial policy on task_i
                    while True:
                        Ss.append(state)
                        action = outer_agent.act(state.reshape(1,26))
                        As.append(action)
                        state, reward, done, _ = env.step(action.numpy().flatten())
                        Rs.append(reward)
                        h += 1
                        if h == 200:
                            break
                task_loss2 = inner_agent.learn_pg(np.array(Ss,dtype=np.float32),As,np.array(Rs,dtype=np.float32))
            # Sample trajectories using new policy theta' with inner agent
            avg_return = []
            new_Ss = []
            new_As = []
            new_Rs = []
            values = []
            probs = []
            for _ in range(K):
                state = env.reset()
                h = 0
                init_value,_ = inner_agent.critic(state.reshape(1,26))
                values.append(init_value.numpy()[0])
                # Sample K trajectories from updated policy on task_i
                while True:
                    new_Ss.append(state)
                    action = inner_agent.act(state.reshape(1,26))
                    new_As.append(action)
                    state, reward, done, _ = env.step(action.numpy().flatten())
                    new_Rs.append(reward)
                    actor_dist,_ = inner_agent.policy(state.reshape(1,26))
                    probs.append(actor_dist.log_prob(action))
                    value,_ = inner_agent.critic(state.reshape(1,26))
                    values.append(value.numpy()[0])
                    h += 1
                    if h == 200:
                        avg_return.append(sum(new_Rs[-200:]))
                        break
            states, actions,returns, adv  = inner_agent.preprocess_ppo(new_Ss, new_As, new_Rs, values)
        
            # Update temp outer agent with trajectories sampled from each updated policy in env. t_i
            temp_outer_agent = agent(input_tensor_spec,action_spec,value_spec)
            temp_outer_agent.policy=actor_copy
            temp_outer_agent.critic=(copy_critic(outer_agent.critic,env.reset().reshape(1,26),
                                              input_tensor_spec,value_spec))
            actor_loss,critic_loss = temp_outer_agent.learn_ppo(np.array(new_Ss,dtype=np.float32),
                                                            new_As,adv,probs,np.array(new_Rs,dtype=np.float32))
            avg_task_return.append(sum(avg_return)/len(avg_return))
        # Update outer agent policy without in a copy of the model, so the inner_agent isn't affected
        meta_iter_critics.append(copy_critic(temp_outer_agent.critic,env.reset().reshape(1,26),input_tensor_spec,value_spec))
        meta_iter_policies.append(copy_actor(temp_outer_agent.policy,env.reset().reshape(1,26),input_tensor_spec,value_spec))
        avg_meta_return.append(sum(avg_task_return)/len(avg_task_return))
        print("Meta Iteration Complete")
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--meta_iteration", default=100, type=int)
    parser.add_argument("--meta_batch_size", default=20, type=int)
    parser.add_argument("--horizon", "-H", default=200, type=int)
    parser.add_argument("--num_adapt_steps", default=1, type=int)

    args = parser.parse_args()

    main(args)