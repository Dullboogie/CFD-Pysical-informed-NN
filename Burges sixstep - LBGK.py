# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:13:55 2018

@author: Administrator
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


nx = 401  #x方向网格节点数量
dx = 2*np.pi/(nx-1)  #空间网格步长，x方向总长度2
nt = 100 # 时间步数
dt = 0.005  # 时间步长

loss=[]

# 指定初始条件

x = np.linspace(0,2*np.pi,nx,dtype="float32")

u = np.zeros(nx,dtype="float32")
phi=np.exp(-x**2/0.28)+np.exp(-(x-2*np.pi)**2/0.28)
u=(x/phi)*np.exp((-x*x)/0.28)+((x-2*np.pi)/phi)*np.exp((-(x-2*np.pi)**2)/0.28)+4


#开始秀起来
train_X=x.reshape(len(x),1)
train_Y=u.reshape(len(u),1)

X=tf.placeholder(tf.float32)   
Y=tf.placeholder(tf.float32)

#下面是迭代时间步后速度的神经元
l=tf.contrib.layers.variance_scaling_initializer(
    factor=1.0,
    mode='FAN_AVG',
    uniform=False,
    seed=None,
    dtype=tf.float32)

W31=tf.Variable(l([1,50]),name="weight")#一个神经元
b31=tf.Variable(tf.zeros([1,50]),name="weight")#对应一个神经元的偏差
W32=tf.Variable(l([50,50]),name="weight")#一个神经元
b32=tf.Variable(tf.zeros([1,50]),name="weight")#对应一个神经元的偏差
W33=tf.Variable(l([50,50]),name="weight")#一个神经元
b33=tf.Variable(tf.zeros([1,50]),name="weight")#对应一个神经元的偏差
W34=tf.Variable(l([50,50]),name="weight")#一个神经元
b34=tf.Variable(tf.zeros([1,50]),name="weight")#对应一个神经元的偏差
W35=tf.Variable(l([50,1]),name="weight")#一个神经元
b35=tf.Variable(tf.zeros([1]),name="weight")


#下面是迭代时间步后速度的神经元solution
h31=tf.nn.tanh(tf.matmul(X,W31)+b31)
h32=tf.nn.tanh(tf.matmul(h31,W32)+b32)
h33=tf.nn.tanh(tf.matmul(h32,W33)+b33)
h34=tf.nn.tanh(tf.matmul(h33,W34)+b34)
Unz=tf.matmul(h34,W35)+b35
#下面是计算参数设定 
u_x=tf.gradients(Unz,X)[0]
u_xx=tf.gradients(u_x,X)[0]  
cost=tf.square(Unz[0]-Unz[-1])+tf.reduce_mean(tf.square(Unz-Y+dt*(Unz*u_x-tf.multiply(0.07,u_xx))))

#LBGSB
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost, method = 'L-BFGS-B', 
                                                   options = {'maxiter': 5000,
                                                              'maxfun': 5000,
                                                              'maxcor': 50,
                                                              'maxls': 50,
                                                              'ftol' : 1.0 * np.finfo(float).eps})
#Adam
Adam_optimizer=tf.train.AdamOptimizer().minimize(cost)
training_epochs=1000   
init=tf.global_variables_initializer()
sess.run(init)


for i in range(nt):
    tf_dict ={X:train_X,Y:train_Y}
    for epoch in range(training_epochs):
        sess.run(Adam_optimizer,feed_dict=tf_dict)
     
    optimizer.minimize(sess,feed_dict = tf_dict,fetches = [cost])
    Uf=sess.run(Unz,feed_dict=tf_dict)
    u_add2=Uf
    train_Y=u_add2.reshape(len(u_add2),1) 
    
plt.plot(x, Uf, 'r',lw =3,label= 'current')

plt.plot(x,u, 'b',lw=3, label = 'init')



#theory
phi_2=np.exp((-(x-0.56*np.pi)**2)/(0.28*(0.14*np.pi+1)))+np.exp((-(x-2.56*np.pi)**2)/(0.28*(0.14*np.pi+1)))
u_2=((x-0.56*np.pi)/(phi_2*(0.14*np.pi+1)))*np.exp((-(x-0.56*np.pi)**2)/(0.28*(0.14*np.pi+1)))+((x-2.56*np.pi)/(phi_2*(0.14*np.pi+1)))*np.exp((-(x-2.56*np.pi)**2)/(0.28*(0.14*np.pi+1)))+4
plt.plot(x,u_2, 'b--',lw=3, label = 'Theory')

plt.legend() 












