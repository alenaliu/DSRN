import tensorflow as tf
import numpy as np
import os
from tensorflow.python.ops import control_flow_ops
import tensorflow.contrib.slim as slim
import scipy.io as sio
import matplotlib.pyplot as plt
import skimage.io as skio
import operation

phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')



def batch_norm(x, phase_train):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
        affn:      whether to affn-transform outputs
    Return:
        normed:      batch-normalized maps
    Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
    """
    name = 'batch_norm'
    with tf.variable_scope(name):
        phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
        n_out = int(x.get_shape()[3])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                           name=name + '/beta', trainable=True, dtype=x.dtype)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                            name=name + '/gamma', trainable=True, dtype=x.dtype)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = control_flow_ops.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


model_dir = os.path.join('SavedModel', 'final')
if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
    os.makedirs(model_dir)


def weight_variable(shape,name):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
    return tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.01),regularizer=regularizer)



def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def doubleConv(input,batch_size,input_maps,input_w,input_h,num_filters,filter_size,kernel_size,name):

    W=weight_variable([num_filters,filter_size,filter_size,input_maps],name)
    filter_offset=filter_size-kernel_size+1
    times=filter_offset**2
    W_shape=(num_filters*times,input_maps,kernel_size,kernel_size)
    filter = tf.reshape(tf.eye(np.prod(W_shape[1:])), (  np.prod(W_shape[1:]), W_shape[1], W_shape[2], W_shape[3]    ))

    filter=tf.transpose(filter,perm=[2,3,1,0])

    W_effective=tf.nn.conv2d(W,filter,strides=[1, 1, 1, 1], padding='VALID')
    W_effective=tf.transpose(W_effective,perm=[0,3,1,2])
    W_effective=tf.reshape(tf.transpose(W_effective,perm=[0,2,3,1]),W_shape)
    W_effective=tf.transpose(W_effective,perm=[2,3,1,0])
    output=tf.nn.conv2d(input,W_effective,strides=[1,1,1,1],padding='SAME')
    output=tf.transpose(output,perm=[0,3,1,2])
    this_shape=(batch_size,num_filters,times,input_w,input_h)
    output=tf.reshape(output,this_shape)
    output=tf.reduce_max(output, axis=2)
    return  tf.transpose(output,perm=[0,2,3,1])


x = tf.placeholder(tf.float32, shape=[None, 60, 60, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 38])

x_image = x


b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(batch_norm(doubleConv(x_image,64,1,60,60,32,4,3,'w_conv1'), phase_train=phase_train_placeholder) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(batch_norm(doubleConv(h_pool1,64,32,30,30,64,4,3,'w_conv2'), phase_train=phase_train_placeholder) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(batch_norm(doubleConv(h_pool2,64,64,15,15,128,4,3,'w_conv3'), phase_train=phase_train_placeholder) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(batch_norm(doubleConv(h_pool3,64,128,8,8,256,4,3,'w_conv4'), phase_train=phase_train_placeholder) + b_conv4)
h_pool4 = h_conv4




h_pool2_flat = tf.reshape(h_pool4, [-1, 64,256])
W_fc1= weight_variable([256 , 128],'w_fc2')

b_fc1 = bias_variable([128])

h_fc1 = tf.cos(tf.tensordot(h_pool2_flat,W_fc1,axes = [[2], [0]])+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1 = tf.nn.dropout(h_fc1, keep_prob)


h_fc1_flat=tf.reshape(h_fc1, [-1, 64*128])



W_fc4 = weight_variable([128*64, 100],'w_fc4')
b_fc4 = bias_variable([100])
h_fc4=tf.matmul(h_fc1_flat, W_fc4) + b_fc4


y_conv = slim.fully_connected(h_fc4, 38, activation_fn=None,
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              weights_regularizer=slim.l2_regularizer(0.001),
                              scope='out', reuse=False)

sess = tf.InteractiveSession()

regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(y_conv, y_)), 1))
total_loss = tf.add_n([loss] + regularization_losses, name='total_loss')
train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

sess.run(tf.initialize_all_variables())
#ckpt = tf.train.get_checkpoint_state('SavedModel/final/')
#saver.restore(sess, ckpt.model_checkpoint_path)








def uncrop(label_predict,bbox_te):
    for i in range(4386):
        for j in range(19):
            label_predict[i,j]=label_predict[i,j]*(bbox_te[i,1]-bbox_te[i,0])+bbox_te[i,0]
        for j in range(19,38):
            label_predict[i,j]=label_predict[i,j]*(bbox_te[i,3]-bbox_te[i,2])+bbox_te[i,2]
    return label_predict

def recover(sess,x,y_,keep_prob,phase_train_placeholder,y_conv):
    img_te = sio.loadmat('img_testing')['data'][:, :, :, np.newaxis]
    data_te = sio.loadmat('label_testing')['data']
    bbox_te = sio.loadmat('bbox_testing')['data']

    label_predict=np.zeros((4386,38))
    for i in range(68):
        image = img_te[i * 64:(i * 64 + 64), :, :, :]
        label = data_te[i * 64:i * 64 + 64, :]

        yp = sess.run(y_conv, feed_dict={x: image, y_: label, keep_prob: 1.0, phase_train_placeholder: False})
        label_predict[i*64:(i+1)*64,:]=yp
    image = img_te[-64:, :, :, :]
    label = data_te[-64:, :]
    yp = sess.run(y_conv, feed_dict={x: image, y_: label, keep_prob: 1.0, phase_train_placeholder: False})
    label_predict[-34:, :] = yp[-34:,:]
    return uncrop(label_predict,bbox_te)




img=sio.loadmat('img_training')['data'][:,:,:,np.newaxis]
data=sio.loadmat('label_training')['data']
bbox=sio.loadmat('bbox_training')['data']

num=0
for e in range(10000):
    bbox, data, img = operation.shuffle(bbox, data, img)#

    for i in range(312):
        image=img[i*64:(i*64+64),:,:,:]
        labels=data[i*64:(i*64+64),:]
        train_step.run(feed_dict={x: image, y_: labels, keep_prob: 1.0, phase_train_placeholder: True})

        if num % 100 == 0:


            m=ALE(sess,x, y_, keep_prob,phase_train_placeholder,y_conv)


        num += 1
