# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 21:58:16 2018

@author: efthi
"""



import random

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
env = gym.make("Taxi-v2")

#Implementing network
tf.reset_default_graph()

#Establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,500],dtype=tf.float32)#hold data for later use
W = tf.Variable(tf.random_uniform([500,6],0,0.1))#shared variable
Qout = tf.matmul(inputs1,W)#the multiplication of the two matrices
predict = tf.argmax(Qout,1)

#Obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,6],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)


init = tf.initialize_all_variables()#Training the network

finished_games = 0#Number of finished games
gamma = 0.99# importance we want to give to future rewards.
epsilon = 0.1#percentage to choose a random step
num_episodes =1000

 

rList = []
#initialize a tensorflow session
with tf.Session() as sess:
    
    sess.run(init)
    for episode in range(num_episodes):
        #Reset environment and get first new state(observation)
        state = env.reset()
        done = False
        finished_games = 0
        j=0
        
        while j<101: 
            j+=1
            
           
            if random.uniform(0, 1) >= epsilon:
                #exploitation: choose an action from the Q-network
                a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(500)[state:state+1]})
            else:
                a[0] = env.action_space.sample() # exploration with a new option with probability epsilon
                 
            state1,rew,done,info = env.step(a[0]) #Get all the needen infos(new state,reward,done and info)
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(500)[state1:state1+1]}) #get the Q values
            maxQ1 = np.max(Q1) #Obtain maxQ1 and set our target value for chosen action.
            targetQ = allQ
            targetQ[0,a[0]] = rew + gamma*maxQ1
            #train the network
            info,Weight1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(500)[state:state+1],nextQ:targetQ})
            finished_games += rew#add all the rewards to finished_games
            state = state1
           
            if done == True :#if game is solved lower epsilon
                #as we train the model we reduce the random step percentage
                epsilon -=0.001
        #every 10 episodes report games finished
        if episode % 100 == 0 and episode != 0 :
            rList.append(finished_games)
            print('Episode '+str(episode)+' Taxi game solved: '+str(finished_games)+' times')
        
            
        
print ("Percent of succesful episodes: " + str(finished_games/num_episodes) + "%")
  


rew=0.
state= env.reset()
done=False
while (done!= True):
    action = np.argmax(Weight1[state])
    state, rew, done, info = env.step(action) #take step using selected action
    env.render()
    print("Reward: ",rew)
    
print("Reward: ",rew) 
#print the value given to each state
 
print(Weight1) 


plt.plot(rList)
plt.xlabel('Number of times used data')
plt.ylabel('Solved Games(every 100games)')
plt.show()





