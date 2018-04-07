import tensorflow as tf
import numpy as np
import random
import copy

# hyperparameter
BATCH_SIZE = 128
ITER = 391
EPOCH = 164
MOMENTUM = tf.Variable(0.9)
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0001
TOTAL_DETPH = 110
N = int((TOTAL_DETPH-2)/6)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None,32,32,3],name='input_x')
ys = tf.placeholder(tf.float32, [None,10],name='input_y')

# set saver
IS_TRAIN = False
CHECKPOINT_STEPS = ITER
CHECKPOINT_DIR = "saver/Lab1_"+str(TOTAL_DETPH)+"/save_net.ckpt"

def compute_accuracy(x,y,output,sess):
    y_pre = sess.run(output, feed_dict={xs: x})
    result = tf.contrib.metrics.accuracy(labels = tf.argmax(y,1),predictions = tf.argmax(y_pre,1))
    return sess.run(result)

def weight_variable(shape,name,initial = tf.contrib.layers.xavier_initializer()):
    regular = tf.contrib.layers.l2_regularizer(scale = WEIGHT_DECAY)
    weight = tf.get_variable(name = "weights"+name,shape = shape,initializer = initial,regularizer=regular)
    return weight

def bias_variable(shape,name):
    initial = tf.contrib.layers.xavier_initializer()
    regular = tf.contrib.layers.l2_regularizer(scale = WEIGHT_DECAY)
    bias = tf.get_variable(name = "bias"+name,shape = shape,initializer = initial,regularizer=regular)
    return bias

# shuffle data
def shuffle_unison(x,y):
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)

# augmentation
def augmentation(img):
    temp = np.pad(img,((0,0),(4,4),(4,4),(0,0)),'constant')
    for i in range(img.shape[0]):
        # translation
        shift1 = random.randint(0,8)
        shift2 = random.randint(0,8)
        img[i]=temp[i][shift1:32+shift1,shift2:32+shift2][:]
        # flip
        if random.randint(0,1)==0:
            img[i]=np.flip(img[i],1)
    return img

# batch norm
def batch_norm(data):
    axis = list(range(len(data.get_shape()) - 1))
    fc_mean, fc_var = tf.nn.moments(data,axes = axis)
    dimension = data.get_shape().as_list()[-1]
    shift = tf.Variable(tf.zeros([dimension]))
    scale = tf.Variable(tf.ones([dimension]))     
    epsilon = 0.001
    return tf.nn.batch_normalization(data, fc_mean, fc_var, shift, scale, epsilon)
                                                                                        
# get labels
def get_labels(y):
    len = y.shape[0]
    labels = [[0 for x in range(10)] for y in range(len)]
    for i in range(len):
        labels[i][int(y[i])]=1
    return labels

# create weights
def create_weights():
    weight=[[] for x in range(5)]
    weight[0]=weight_variable([3,3,3,16],'0')

    for i in range(2*N):
        if i==2*N-1:
            weight[1].append(weight_variable([3,3,16,32],'1'+str(i)))
        else:
            weight[1].append(weight_variable([3,3,16,16],'1'+str(i)))

    for i in range(2*N):
        if i==2*N-1:
            weight[2].append(weight_variable([3,3,32,64],'2'+str(i)))   
        else:
            weight[2].append(weight_variable([3,3,32,32],'2'+str(i)))
    for i in range(2*N):
        weight[3].append(weight_variable([3,3,64,64],'3'+str(i)))
    weight[4] = weight_variable([4*4*64,10],'4')
    return weight

# create bias
def create_bias():
    bias = bias_variable([10],'bias')
    return bias

def network():
    weight=create_weights()
    bias=create_bias()

    # conv layer
    # layer: [128,32,32,16]
    layer = tf.nn.conv2d(input = xs,filter = weight[0],strides = [1,1,1,1],padding = 'SAME')
    pre_layer = tf.nn.relu(batch_norm(layer))


    for i in range(N):
        # layer: [128,32,32,16]
        layer = tf.nn.conv2d(input = tf.nn.relu(batch_norm(layer)),filter = weight[1][2*i],strides = [1,1,1,1],padding = 'SAME')
        if i==N-1:
            # layer: [128,16,16,32]
            layer = tf.nn.conv2d(input = tf.nn.relu(batch_norm(layer)),filter = weight[1][2*i+1],strides = [1,2,2,1],padding = 'SAME')
            pre_layer=tf.nn.avg_pool(pre_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
            pre_layer=tf.pad(pre_layer,[[0,0],[0,0],[0,0],[8,8]])
        else:
            # layer: [128,32,32,16]
            layer = tf.nn.conv2d(input = tf.nn.relu(batch_norm(layer)),filter = weight[1][2*i+1],strides = [1,1,1,1],padding = 'SAME')
        layer = layer+pre_layer
        pre_layer = layer

    for i in range(N):
        # layer: [128,16,16,32]
        layer = tf.nn.conv2d(input = tf.nn.relu(batch_norm(layer)),filter = weight[2][2*i],strides = [1,1,1,1],padding = 'SAME')
        if i==N-1:
            # layer: [128,8,8,64]
            layer = tf.nn.conv2d(input = tf.nn.relu(batch_norm(layer)),filter = weight[2][2*i+1],strides = [1,2,2,1],padding = 'SAME')
            pre_layer=tf.nn.avg_pool(pre_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
            pre_layer=tf.pad(pre_layer,[[0,0],[0,0],[0,0],[16,16]])   
        else:
            # layer: [128,16,16,32]
            layer = tf.nn.conv2d(input = tf.nn.relu(batch_norm(layer)),filter = weight[2][2*i+1],strides = [1,1,1,1],padding = 'SAME')
        layer = layer+pre_layer
        pre_layer = layer

    for i in range(N):
        # layer: [128,8,8,64]
        layer = tf.nn.conv2d(input = tf.nn.relu(batch_norm(layer)),filter = weight[3][2*i],strides = [1,1,1,1],padding = 'SAME')
        # layer: [128,8,8,64]
        layer = tf.nn.conv2d(input = tf.nn.relu(batch_norm(layer)),filter = weight[3][2*i+1],strides = [1,1,1,1],padding = 'SAME')
        layer = layer+pre_layer
        pre_layer = layer

    # fc layer
    # pool size: [128,4,4,64]
    pool = tf.nn.avg_pool(value = tf.nn.relu(batch_norm(layer)),ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
   
    # flatten size: [128,4*4*64]
    flatten = tf.contrib.layers.flatten(inputs = pool)


    # fc size: [128,10] 
    fc = tf.matmul(flatten,weight[4])+bias

    # output size: [128,10] 
    output = tf.nn.softmax(fc)

    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc,labels = ys))
    tf.summary.scalar('loss', loss)

    # optimizer
    train_step = tf.train.MomentumOptimizer(learning_rate = LEARNING_RATE, momentum = MOMENTUM).minimize(loss)
    
    return output,train_step

def run_network():
    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    train_labels = get_labels(y_train)
    test_labels = get_labels(y_test)
    output,train_step=network()
    sess = tf.Session()
    merged = tf.summary.merge_all()
    if IS_TRAIN: 
        writer = tf.summary.FileWriter("logs/Lab1_"+str(TOTAL_DETPH),sess.graph)

    # initial saver
    saver = tf.train.Saver()

    # initialize variable
    init = tf.global_variables_initializer()
    sess.run(init)

    # run network
    if IS_TRAIN:
        for i in range(EPOCH):
            shuffle_unison(x_train,train_labels)
            img = augmentation(copy.deepcopy(x_train))
            print("EPOCH:",i)
            global LEARNING_RATE
            if i==81 or i==122:
                LEARNING_RATE/=10
            for j in range(ITER):   
                start = j*128 if (j*128+128) < 50000 else (50000-128)
                end = (start+128)
                sess.run(train_step, feed_dict={xs: img[start:end], ys: train_labels[start:end]})        
                if (j+1) % CHECKPOINT_STEPS == 0:
                    saver.save(sess,CHECKPOINT_DIR)

            result_test = compute_accuracy(x_test,test_labels,output,sess)
            print ("Test accuracy:",result_test)
            summary = tf.Summary(value=[tf.Summary.Value(tag="error", simple_value=(1.0-result_test))])
            result_train = compute_accuracy(x_train[:128],train_labels[:128],output,sess)
            print ("Train accuracy:",result_train)
            rs = sess.run(merged, feed_dict = {xs: x_test, ys: test_labels})
            writer.add_summary(rs,i)
            writer.add_summary(summary,i)
        writer.close()
    else:
        saver.restore(sess,CHECKPOINT_DIR)
        result_test = compute_accuracy(x_test,test_labels,output,sess)
        print ("Test accuracy:",result_test)    

def main():
    run_network()

if __name__=='__main__':
    main()




