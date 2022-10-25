import gym
from gym import spaces
from gym.envs.registration import register

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

import tf_agents

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import os

from tqdm import tqdm
#create gym



class PerceptualLearning(gym.Env):

    # np.random.seed(1234)

    metadata = {"render.modes":["human", "rgb_array"],"video.frames_per_second": 1}

    state_elements = 4


    #define the participant model

    def __init__(self):


        self.initial_accuracy = 0
        self.total_accuracy = []

        self.initial_rotation = 3 #3 #np.random.randint(1,8)
        self.initial_noise = 3 #3 #np.random.randint(1,8)
        # self.total_n = []

        self.initial_response = 0

        self.action_space = spaces.Discrete(10) # 0: increase difficulty, 1: decrease difficulty, 2: same, 3: random action

        self.low = np.array([1,1,0,0],dtype=np.float32)
        self.high = np.array([7,7,1,1], dtype=np.float32)
        self.observation_space = spaces.Box(low= self.low, high= self.high)

        # self.state_log = []


        self.trials = 1

        self.reset()

    def predict_parameter(self,action):
        # decrease rotation permutation
        if action == 0:
            self.rotation -= 1
            self.noise -= 1
            if self.rotation == 0:
                self.rotation = 1
            if self.noise == 0:
                self.noise = 1


        if action == 1:
            self.rotation -= 1
            self.noise = self.noise
            if self.rotation == 0:
                self.rotation = 1

        if action == 2:
            self.rotation -= 1
            self.noise += 1
            if self.rotation == 0:
                self.rotation = 1
            if self.noise == 8:
                self.noise = 7



# constant rotation permutation
        if action == 3:
            self.rotation = self.rotation
            self.noise -= 1

            if self.noise == 0:
                self.noise = 1

        if action == 4:
            self.rotation = self.rotation
            self.noise = self.noise

        if action == 5:
            self.rotation = self.rotation
            self.noise += 1

        if self.noise == 8:
            self.noise = 7

#increase rotation permutation
        if action == 6:
            self.rotation += 1
            self.noise -= 1

            if self.noise == 0:
                self.noise = 1

            if self.rotation == 8:
                self.rotation = 7

        if action == 7:
            self.rotation += 1
            self.noise = self.noise

            if self.rotation == 8:
                self.rotation = 7

        if action == 8:
            self.rotation += 1
            self.noise += 1

            if self.rotation == 8:
                self.rotation = 7

            if self.noise == 8:
                self.noise = 7

        # random action
        if action == 9:
            self.predict_parameter(action = np.random.randint(0,9))



    def step(self,action):
        self.predict_parameter(action)



        self.rotation = self.rotation
        self.noise = self.noise
        # self.total_n.append(self.n)

        # np.argmax(PerceptualLearning.participant.predict([self.n]))

        dir = "C:/YNC/YNC/SIRP/Data/curateOS_normalised_filtered.csv"

        df = pd.read_csv(dir)

        df["correct"] = df["correct"].astype(int)

        # df = df.fillna(0)

        df2 = df[(df.rotation == self.rotation) & (df.noise == self.noise)]

        if len(df2) == 0:
            self.response = np.random.randint(2)
        else:
            sampling = df2.sample(1)["correct"]
            self.response = sampling.tolist()[0]



        # self.response = np.argmax(PerceptualLearning.participant.predict(tf.expand_dims(np.array([self.rotation,self.noise]),axis=0))) #replace with prediction afterwards using self.n
        self.total_response.append(self.response)

        self.accuracy = sum(self.total_response)/(self.current_step+1)

        self.total_accuracy.append(self.accuracy)

        self.current_step +=1

        # if round(self.accuracy,2) == 0.8:
        #     self.reward = 0
        # else:


        # if self.accuracy == 0.8:
        #     self.done = True
        # else:
        #     self.done = False

        # for testing




        self.info = {}
        self.info["Difficulty"] = self.state
        self.info["Accuracy"] = self.accuracy
        self.info["Response"] = self.response
        self.info["Trial"] = self.trials

        self.state[0] = self.rotation
        self.state[1] = self.noise
        self.state[2] = self.accuracy
        self.state[3] = self.response


        self.reward = round(-abs((0.8 - self.accuracy)*100),3)
        self.reward = self.reward/(99-self.trials)

        self.trials += 1

        if self.trials == 99:
            self.done = True
        else:
            self.done = False
         #for testing



        return self.state, self.reward, self.done, self.info


    def reset(self):
        self.reward = 0
        self.done = False
        self.info = {}

        self.rotation = 3 #3 #np.random.randint(1,8)
        self.noise = 3 #3 #np.random.randint(1,8)

        self.current_step = 0
        # self.predictions = []
        self.total_response = []
        self.trials = 0
        self.accuracy = 0
        self.total_accuracy = []

        self.response = self.initial_response
        # self.total_n = []

        self.state = [0] * PerceptualLearning.state_elements

        return np.array(self.state)

    def render(self, mode = "human"):
        s = "Difficulty: {} Reward: {} info: {}"
        print(s.format(self.state, self.reward, self.info))


register(
id ="perceptual-learning-v1",
entry_point=f"{__name__}:PerceptualLearning",
)



env = gym.make("perceptual-learning-v1")

# env.action_space.sample()
#
# env.observation_space.sample()


episode = 1

for episode in range(1, episode+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()

        action = env.action_space.sample()
        state, reward, done, info = env.step(action)


#Creating a RL Model

#hyper parameter

num_iterations = 80000
initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 128
learning_rate = 1e-4
log_interval = 200

num_eval_episodes = 10
eval_interval = 1000

#hyper parameter end


train_py_env = suite_gym.load("perceptual-learning-v1")
test_py_env = suite_gym.load("perceptual-learning-v1")

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
test_env = tf_py_environment.TFPyEnvironment(test_py_env)

fc_layer_params = (200,)

cat_q_net = categorical_q_network.CategoricalQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params = fc_layer_params
)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network = cat_q_net,
    optimizer =optimizer,
    td_errors_loss_fn = common.element_wise_squared_loss,
    train_step_counter= train_step_counter,
    n_step_update = 1,
    max_q_value = 0,
    min_q_value = -80,
    epsilon_greedy = 0.2,
    gamma = 0.9)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

def compute_avg_return (environment, policy, num_episodes = 10):

    total_return = 0.0
    total_acc = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        episode_acc = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

        if time_step.is_last():

            acc = np.array(time_step[3])
            acc = acc.tolist()[0][2]
            episode_acc += acc
        total_acc += episode_acc

    avg_return = total_return / num_episodes
    avg_acc = total_acc / num_episodes
    return avg_return.numpy()[0], avg_acc

#
# time_step = test_env.reset()
#
# acc = np.array(time_step[3])
#
# acc = acc.tolist()
#
# acc[0][2]

compute_avg_return(test_env, random_policy, num_eval_episodes)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size= train_env.batch_size,
    max_length = replay_buffer_max_length)

# agent.collect_data_spec
# agent.collect_data_spec._fields

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)

    # print("add trajectory") #debugging

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)
    #
    #     print ("collect step")
    #
    # print("collect data")

collect_data(train_env, random_policy, replay_buffer, steps = 100)

# print ("collect data d")


dataset = replay_buffer.as_dataset(
    num_parallel_calls = 3,
    sample_batch_size = batch_size,
    num_steps = 2).prefetch(3)

iterator = iter(dataset)
# print(iterator)
# iterator.next()
#
# agent.train = common.function(agent.train)
#
# agent.train_step_counter.assign(0)


avg_return, accuracy = compute_avg_return(test_env, agent.policy, num_eval_episodes)
returns = [avg_return]

acc_container = [accuracy]

loss_container = []


for _ in tqdm(range(num_iterations)):

    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent.collect_policy, replay_buffer)

        # print("collect step")

    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    # print("convert to numpy")

    if step % log_interval == 0:
        print("step = {0}: loss = {1}".format(step, train_loss))

    if step % eval_interval == 0:
        avg_return, accuracy = compute_avg_return(test_env, agent.policy, num_eval_episodes)
        print("step = {0}: Average Return = {1}, Average Accuracy = {2}".format(step,avg_return, accuracy))
        returns.append(avg_return)
        acc_container.append(accuracy)
        loss_container.append(train_loss)



print("done")
variance = list(map(abs,returns))

variance = np.divide(variance,-1).tolist()

iterations = range(0, num_iterations +1, eval_interval)
plt.plot(iterations,variance)
plt.title("Categorical Dqn Agent Reward")
plt.ylabel("Rewards")
plt.xlabel("Iterations")
# plt.ylim(top=-150)


# loss_container = np.array(loss_container).tolist()
diff = np.subtract(acc_container,0.8)
diff = np.multiply(diff,100)
diff = list(map(abs,diff))

plt.plot(iterations, diff)
plt.title("CAT DQN Variance from 80%")
plt.ylabel("Variance from 80%")
plt.xlabel("Iterations")



accuracy2 = np.multiply(acc_container,100)
mean = np.mean(accuracy2)

plt.plot(iterations, accuracy2)
plt.title("CAT DQN Accuracy Plot, mean = {}".format(mean))
plt.axhline(y = mean, color = "black", linestyle = "-", label = "mean")
plt.axhline(y = 80, color = "red", linestyle = "--", label = "80% line")
plt.ylabel("Accuracy (%)")
plt.xlabel("Iterations")

plt.ylim(50,100)
plt.legend()

agent.policy



policy_dir = "C:/YNC/YNC/SIRP/Model/New Perceptual Learning Model Lookup_CatDQN2_newreward2_fixed_perf_4Obs_80k"

tf_policy_saver = policy_saver.PolicySaver(agent.policy)

tf_policy_saver.save(policy_dir)


tf_agents.policies.policy_saver.specs_from_collect_data_spec(agent)

agent2 =tf.compat.v2.saved_model.load(policy_dir)

time_step

agent2.get_initial_state(batch_size = 3)

time_step = test_env.reset()
action_step = agent2.action(time_step)
time_step = test_env.step(action_step.action)

action_step
time_step

train_env.current_time_step



#debug



    time_step = test_env.reset()
    print (time_step)

    while not time_step.is_last():
        action_step = agent2.action(time_step)
        time_step = test_env.step(action_step)
        print (time_step)

agent2.action(tf_agents.trajectories.time_step.TimeStep(
    step_type = tf.convert_to_tensor(np.array([1]),np.int32),
    reward = tf.convert_to_tensor(np.array([-20]),np.float32),
    discount = tf.convert_to_tensor(np.array([1]),np.float32),
    observation = tf.convert_to_tensor(np.array([1,2,0.4]),np.float32)

))


step_type = tf.convert_to_tensor(np.array([1]),np.int32)

step_type
# step_type = tf.TensorSpec.from_tensor(step_type, name = "step_type")


reward = tf.convert_to_tensor(np.array([-20]),np.float32)
# reward = tf.TensorSpec.from_tensor(reward, name = "reward")

discount = tf.convert_to_tensor(np.array([1]),np.float32)
# discount = tf.TensorSpec.from_tensor(discount, name = "discount")

observation = tf.convert_to_tensor(np.array([1,2,0.4]),np.float32)
observation = tf.reshape(observation, [1,3])
# observation = tf.TensorSpec.from_tensor(observation, name = "observation")

observation

#
#
# step_type = tf.cast(tf.constant(np.array([1]), dtype = tf.int32), dtype = tf.int32, name = "step_type")
#
# reward = tf.cast(tf.constant(np.array([-20]),dtype = tf.float32),dtype = tf.float32, name = "reward")
#
# discount = tf.cast(tf.constant(np.array([1]), dtype = tf.float32), dtype = tf.float32, name = "discount")
#
# observation = tf.cast(tf.constant(np.array([1.0,2.0,0.4]), dtype = tf.float32), dtype = tf.float32, name = "observation")
# observation = tf.reshape(observation, [1,3])


# tf.TensorSpec.from_tensor(tf.constant(np.array([1]), dtype = tf.int32), name = "step_type")

timestep = tf_agents.trajectories.time_step.transition(
    # step_type= step_type,
    reward= reward,
    # discount=discount,
    observation=observation
    )

tf.TensorSpec.from_tensor(tf.constant(np.array([1]), dtype = tf.int32))

agent2.action(timestep)

tf.constant(1, dtype = tf.float32)

tf_agents.trajectories.time_step.time_step_spec(
    observation_spec = np.array([1,2,0.4]),
    reward_spec = np.array([-20])
)

agent2.action(tf_agents.trajectories.time_step.time_step_spec(
    observation_spec = np.array([1,2,0.4]),
    reward_spec = np.array([-20])
))



from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec

from tf_agents.trajectories import time_step as ts

input_tensor_spec = tensor_spec.TensorSpec((3,),tf.float32)
time_step_spec = ts.time_step_spec(input_tensor_spec)

input_tensor_spec

time_step_spec

batch_size = 1
#
# observation = tf.constant([1,2,0.8])
# reward = tf.constant(-20.0)
#
# time_step = ts.transition(observation = observation, reward = reward)

observation = tf.ones([1]+ time_step_spec.observation.shape.as_list())

observation.shape

time_step_spec.observation.shape.as_list()

time_step = ts.restart(observation, batch_size = batch_size )


observation
time_step

agent2.action(time_step)


















# env.time_step_spec().observation
