from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import gym
import pybullet_envs
import numpy as np
import tensorflow as tf
import abc
import tensorflow_probability as tfp
import tf_agents

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.networks import network
from tf_agents.policies import tf_policy
from tf_agents.policies import actor_policy
from tf_agents.trajectories import time_step as ts

from agent import ActionDistributionNet
from agent import Agent_ppo

def params():
    action_spec = tensor_spec.BoundedTensorSpec((8,), tf.float32, minimum = -1.0,maximum = 1.0)
    input_tensor_spec = tensor_spec.TensorSpec((28,), tf.float32)
    action_distribution_net = ActionDistributionNet(input_tensor_spec, action_spec)
    num_iterations = 500
    collect_episodes_per_iteration = 1 
    replay_buffer_capacity = 2000
    fc_layer_params = (28,)
    learning_rate = 1e-3 
    log_interval = 10 
    num_eval_episodes = 10 
    eval_interval = 10 
    return [action_distribution_net,
            num_iterations,num_eval_episodes,
            collect_episodes_per_iteration,replay_buffer_capacity,
            fc_layer_params,learning_rate,log_interval,num_eval_episodes,
            eval_interval]

def REINFORCEagent(tf_env,action_distribution_net,
                        optimizer,train_step_counter):
    agent = reinforce_agent.ReinforceAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_network=action_distribution_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    return agent

def REINFORCEwbAgent(tf_env,action_distribution_net,value_net,
                        optimizer,train_step_counter):
    agent = reinforce_agent.ReinforceAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_network=action_distribution_net,
        value_network = value_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    return agent

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def collect_episode(environment, policy, 
                    num_episodes,step_count,
                    replay_buffer):
    episode_counter = 0
    environment.reset()
    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)
        
        step_count += 1
        if traj.is_boundary():
            episode_counter += 1
            print("Step Count: " + str(step_count))
    return step_count

def test_reward(env,ag):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = ag.act(state=state)[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    return total_reward

def preprocess1(states, actions, rewards, done, values, gamma):
    g = 0
    lmbda = 1
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
        g = delta + gamma * lmbda * done[i] * g
        returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv

def main(args):
    pList = params()
    if args.algo != "ppo":
        env = suite_gym.load(args.env)
        tf_env = tf_py_environment.TFPyEnvironment(env)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=pList[7])
        train_step_counter = tf.compat.v2.Variable(0)

        if args.algo == "pg":
            r_agent = REINFORCEagent(tf_env,pList[0],optimizer,train_step_counter)
            r_agent.initialize()
    
        if args.algo == "pgb":
            value_net = tf_agents.networks.value_network.ValueNetwork(tf_env.observation_spec())
            r_agent = REINFORCEwbAgent(tf_env,pList[0],value_net,optimizer,train_step_counter)
            r_agent.initialize()

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=r_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=pList[4])
        r_agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(tf_env, r_agent.policy, pList[8])
        returns = [avg_return]
        curr_step_count = 0
        while curr_step_count < 500001:

            # Collect a few episodes using collect_policy and save to the replay buffer.
            curr_step_count = collect_episode(
                                tf_env, r_agent.collect_policy,
                                args.epoch, curr_step_count,
                                replay_buffer)

            # Use data from the buffer and update the agent's network.
            experience = replay_buffer.gather_all()
            train_loss = r_agent.train(experience)
            replay_buffer.clear()

            step = r_agent.train_step_counter.numpy()

            if step % pList[7] == 0:
                print('Episode = {0}: loss = {1}'.format(step, train_loss.loss))

            if step % pList[9] == 0:
                avg_return = compute_avg_return(tf_env, r_agent.policy, pList[8])
                print('Episode = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
    
    else:
        env= gym.make(args.env)
        agent_ppo = Agent_ppo()
        steps = 500
        avg_rewards_list = []


        for s in range(steps):
            done = False
            state = env.reset()
            rewards = []
            states = []
            actions = []
            probs = []
            dones = []
            values = []
            print("new episode")

            for e in range(args.epoch):
   
                action = agent_ppo.act(state)[0]
                value = agent_ppo.critic(np.array([state])).numpy()
                next_state, reward, done, _ = env.step(action)
                dones.append(1-done)
                rewards.append(reward)
                states.append(state)
                #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
                actions.append(action)
                prob = agent_ppo.actor(np.array([state])).numpy()
                probs.append(prob[0])
                values.append(value[0][0])
                state = next_state
                if done:
                    env.reset()
  
            value = agent_ppo.critic(np.array([state])).numpy()
            values.append(value[0][0])
            np.reshape(probs, (len(probs),8))
            probs = np.stack(probs, axis=0)

            states, actions,returns, adv  = preprocess1(states, actions, rewards, dones, values, 1)

            for epocs in range(10):
                al,cl = agent_ppo.learn(states, actions, adv, probs, returns)   

            avg_reward = np.mean([test_reward(env,agent_ppo) for _ in range(10)])
            print(f"total test reward is {avg_reward}")
            avg_rewards_list.append(avg_reward)
            env.reset()

        env.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="AntBulletEnv-v0")
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--algo", required=True, type=str, help="Name of algorithm. It should be one of [pg, pgb, ppo]")

    args = parser.parse_args()
    main(args)
