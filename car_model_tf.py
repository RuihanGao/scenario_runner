'''
Two-layer feed-forward to approximate car model, for Gazebo simulation, tensorflow framework
ref: https://github.com/854768750/car_model
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv

start_time = time.time()
loss_data = 0
iterations = 200001
hidden_neuron_num1 = 50
hidden_neuron_num2 = 500
train_size = 90000
test_size = 5000
predict_size = 93000
plot = True
train = 2 # 1 for first time train, 2 for loading parameter and not training, 3 for loading and keep training
layer_num = 2
project_dir = "/home/lion/car_model"
print("iterations:",iterations)
lowest_loss = 100
batch_size  = 100

'''
load data from the second line, the first line of the txt is data name
array delta_ori stores the difference between two orientations and is limited between -2pi~2pi
command_data includes time step and torque
input_data includes command data and previous state
state_data includes current state
state_data does not include init_state
'''

f = open(project_dir+"/state_command_data_7.txt", "r")
next(f)
result = []
for line in f:
    result.append(map(float,line.split(' ')))
for i in range(0, len(result)-1):
    result[i][0] = result[i+1][0] - result[i][0]
result = np.array(result)

command_data = result[0:-1,9:11].reshape([-1,2])
quaternion_t = result[0:-1,5:7].reshape([-1,2])
v_t = result[0:-1,7:8].reshape([-1,1])
theta_t = np.zeros((len(result)-1,1))
for i in range(0, len(result)-1):
    theta_t[i][0] = math.atan2(2*quaternion_t[i][0]*quaternion_t[i][1], pow(quaternion_t[i][1],2)-pow(quaternion_t[i][0],2))
input_data = np.concatenate((theta_t,v_t,command_data[:,:].reshape([-1,2])),axis=1)


delta_x = np.zeros((len(result)-1,1))
delta_y = np.zeros((len(result)-1,1))
delta_quaternion = np.zeros((len(result)-1,2))
v_t_1 = np.delete(result[:,7:8].reshape([-1,1]), 0, 0)
for i in range(0, len(result)-1):
    delta_x[i][0] = result[i+1][1] - result[i][1]
    delta_y[i][0] = result[i+1][2] - result[i][2]
    delta_quaternion[i,:] = result[i+1,5:7] - result[i,5:7]
output_data = np.concatenate((delta_x,delta_y),axis=1)

x_data = input_data[0:train_size,:]
y_data = output_data[0:train_size,:]

test_x_data = input_data[train_size:train_size+test_size,:]
test_y_data = output_data[train_size:train_size+test_size,:]

input_size = 4
output_size = 2
xs = tf.placeholder(tf.float32, [None, input_size])
ys = tf.placeholder(tf.float32, [None, output_size])

'''
weights1 = tf.Variable(tf.random_normal([input_size,hidden_neuron_num1],seed=1))
biases1 = tf.Variable(tf.random_normal([1,hidden_neuron_num1],seed=1))
Wx_plus_b1 = tf.matmul(xs, weights1) + biases1
output1 = tf.nn.sigmoid(Wx_plus_b1)
weights2 = tf.Variable(tf.random_normal([hidden_neuron_num1,output_size],seed=1))
biases2 = tf.Variable(tf.random_normal([1,output_size],seed=1))
Wx_plus_b2 = tf.matmul(output1, weights2) + biases2
output2 = (Wx_plus_b2)
prediction = output2
'''


if train==1:
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.999, staircase = False)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver=tf.train.Saver(max_to_keep=1)

    for i in range(1,iterations): 
        feed_start = (i*batch_size)%train_size
        feed_end = ((i+1)*batch_size)%train_size
        x_feed = x_data 
        y_feed = y_data
        sess.run(train_step, feed_dict={xs: x_feed, ys: y_feed})
        loss_data = sess.run(loss, feed_dict={xs: test_x_data, ys: test_y_data})
        if loss_data<lowest_loss:
            lowest_loss = loss_data
            print("lowest loss:", lowest_loss)
            saver.save(sess, project_dir+'/ckpt_v5/model.ckpt', global_step=global_step)
        if i % (1000) == 0: 
            print("step: ", sess.run(global_step), loss_data, "learning_rate: ", sess.run(learning_rate))

elif train==2:
    global_step = tf.Variable(0)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    learning_rate = tf.train.exponential_decay(0.05, global_step, 100, 0.999, staircase = False)
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
    saver=tf.train.Saver()
    sess = tf.InteractiveSession()
    model_file=tf.train.latest_checkpoint(project_dir+'/ckpt_v5')
    saver.restore(sess,model_file)

elif train==3:
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 100, 0.999, staircase = False)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
    saver=tf.train.Saver()
    sess = tf.InteractiveSession()
    model_file=tf.train.latest_checkpoint(project_dir+'/ckpt_v5')
    saver.restore(sess,model_file)
    saver=tf.train.Saver(max_to_keep=1)

    for i in range(iterations): 
        x_feed = x_data 
        y_feed = y_data
        sess.run(train_step, feed_dict={xs: x_feed, ys: y_feed})
        loss_data = sess.run(loss, feed_dict={xs: test_x_data, ys: test_y_data})
        if loss_data<lowest_loss:
            lowest_loss = loss_data
            print("lowest loss:", lowest_loss)
            saver.save(sess, project_dir+'/ckpt_v5/model.ckpt', global_step=global_step)
        if i % (1000) == 0: 
            print("step: ", sess.run(global_step), loss_data, "learning_rate: ", sess.run(learning_rate))




start = 0
current_state = result[start,1:3].reshape([-1,2])
predict_state = np.zeros((predict_size,2))
for i in range(0,predict_size,1):
    if(i%5000==0):
        current_state = result[start+i,1:3].reshape([-1,2])
    current_input = np.concatenate((theta_t[start+i,:].reshape([-1,1]),v_t[start+i,:].reshape([-1,1]), command_data[start+i,:].reshape([-1,2])),axis=1)
    current_output = sess.run(prediction, feed_dict={xs: current_input})
    current_state = current_output + current_state#result[start+i,1:3].reshape([-1,2])
    #print("current_input:", current_input)
    predict_state[i,:] = current_state
    #print("state_data:", state_data[i+start,0:6])
    #print("prediction:", predict_state[i,0:6])

    

if plot:
    plt.plot(result[start, 1], result[start, 2], marker='*', color='black')
    plt.plot(result[start+1:start+1+predict_size, 1], result[start+1:start+1+predict_size, 2], marker='*', color='red')
    plt.plot(predict_state[0:predict_size, 0], predict_state[0:predict_size, 1], marker='*', color='green')
    #plt.plot(predict_state[train_size:predict_size, 0], predict_state[train_size:predict_size, 1], marker='*', color='yellow')
    plt.pause(0.000001)

print(sess.run(tf.reduce_mean(tf.sqrt(tf.square(result[start+1:start+1+predict_size, 1] - predict_state[0:predict_size, 0])  \
    + tf.square(result[start+1:start+1+predict_size, 2] - predict_state[0:predict_size, 1])),reduction_indices=[0])))


save_data = True
if save_data:
    #out = open('plot_data.csv','a', newline='')
    #csv_write = csv.writer(out,dialect='excel')
    #csv_write.writerow(stu1)
    np.savetxt('/home/lion/plot_data.csv', np.concatenate((result[start+1:start+1+predict_size, 1:3],predict_state[0:predict_size, 0:2]),axis=1), delimiter = ' ')
    print ("write over")


#print(sess.run(weights1))
#print(sess.run(biases1))
end_time = time.time()
print("cost_time:",end_time-start_time)
print("train_size:",train_size)
print("test_size:",test_size,"loss_data:",sess.run(loss, feed_dict={xs: test_x_data, ys: test_y_data}))
print("predict_size:",predict_size)
raw_input("press any button to exit")