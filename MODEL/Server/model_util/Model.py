from MODEL.Server.model_util.layers import *
import tensorflow as tf
import numpy as np

mu = 0
sigma = 0.1

class B_VGGNet(object):

    def __init__(self, num_class=10):
        self.num_class = num_class

    def model(self, x, is_train):
        # conv layer 1
        with tf.variable_scope("baseline"):
            self.conv1 = Conv2d(x, filters=64, k_size=3, stride=1,name='conv1')
            self.conv1 = BN(self.conv1, phase_train=is_train,name='conv1_bn')
            self.conv1 = Relu(self.conv1,name='conv1_relu')
            self.conv2 = Conv2d(self.conv1, filters=64, k_size=3, stride=1,name='conv2')
            self.conv2 = BN(self.conv2, phase_train=is_train,name='conv2_bn')
            self.conv2 = Relu(self.conv2,name='conv2_relu')
            self.max_pool1 = max_pooling(self.conv2, k_size=2, stride=2,name='block1_maxpool')

            self.conv3 = Conv2d(self.max_pool1, filters=128, k_size=3, stride=1,name='conv3')
            self.conv3 = BN(self.conv3, phase_train=is_train,name='conv3_bn')
            self.conv3 = Relu(self.conv3,name='conv3_relu')
            self.conv4 = Conv2d(self.conv3, filters=128, k_size=3, stride=1,name='conv4')
            self.conv4 = BN(self.conv4, phase_train=is_train,name='conv4_bn')
            self.conv4 = Relu(self.conv4,name='conv4_relu')
            self.max_pool2 = max_pooling(self.conv4, k_size=2, stride=2,name='block2_maxpool')
            #-----------------------------------EXIT
            self.conv5 = Conv2d(self.max_pool2, filters=256, k_size=3, stride=1,name='conv5')
            self.conv5 = BN(self.conv5, phase_train=is_train,name='conv5_bn')
            self.conv5 = Relu(self.conv5,name='conv5_relu')
            self.conv6 = Conv2d(self.conv5, filters=256, k_size=3, stride=1,name='conv6')
            self.conv6 = BN(self.conv6, phase_train=is_train,name='conv6_bn')
            self.conv6 = Relu(self.conv6,name='conv6_relu')
            self.conv7 = Conv2d(self.conv6, filters=256, k_size=3, stride=1,name='conv7')
            self.conv7 = BN(self.conv7, phase_train=is_train,name='conv7_bn')
            self.conv7 = Relu(self.conv7,name='conv7_relu')
            self.max_pool3 = max_pooling(self.conv7, k_size=2, stride=2,name='block3_maxpool')
            # ----------------------------------EXIT
            self.conv8 = Conv2d(self.max_pool3, filters=512, k_size=3, stride=1,name='conv8')
            self.conv8 = BN(self.conv8, phase_train=is_train,name='conv8_bn')
            self.conv8 = Relu(self.conv8,name='conv8_relu')
            self.conv9 = Conv2d(self.conv8, filters=512, k_size=3, stride=1,name='conv9')
            self.conv9 = BN(self.conv9, phase_train=is_train,name='conv9_bn')
            self.conv9 = Relu(self.conv9,name='conv9_relu')
            self.conv10 = Conv2d(self.conv9, filters=512, k_size=3, stride=1,name='conv10')
            self.conv10 = BN(self.conv10, phase_train=is_train,name='conv10_bn')
            self.conv10 = Relu(self.conv10,name='conv10_relu')
            self.max_pool4 = max_pooling(self.conv10, k_size=2, stride=2,name='block4_maxpool')
            #-----------------------------------EXIT
            self.conv11 = Conv2d(self.max_pool4, filters=512, k_size=3, stride=1,name='conv11')
            self.conv11 = BN(self.conv11, phase_train=is_train,name='conv11_bn')
            self.conv11 = Relu(self.conv11,name='conv11_relu')
            self.conv12 = Conv2d(self.conv11, filters=512, k_size=3, stride=1,name='conv12')
            self.conv12 = BN(self.conv12, phase_train=is_train,name='conv12_bn')
            self.conv12 = Relu(self.conv12,name='conv12_relu')
            self.conv13 = Conv2d(self.conv12, filters=512, k_size=3, stride=1,name='conv13')
            self.conv13 = BN(self.conv13, phase_train=is_train,name='conv13_bn')
            self.conv13 = Relu(self.conv13,name='conv13_relu')

            self.fc1 = Flatten(self.conv13)
            self.fc1 = fc_layer(self.fc1, 4096,name='fc3')
            self.fc1 = Relu(self.fc1,name='fc3_relu')
            self.fc1 = Drop_out(self.fc1, 0.2, training=is_train)
            self.fc2 = fc_layer(self.fc1, 4096,name='fc4')
            self.fc2 = Relu(self.fc2,name='fc4_relu')
            self.fc2 = Drop_out(self.fc2, 0.2, training=is_train)
            logits_exit3 = fc_layer(self.fc2, self.num_class,name='logits_exit3')

        with tf.variable_scope("exit0"):
            #tf.reset_default_graph()
            """self.exit0 = Conv2d(self.max_pool2, filters=256, k_size=3, stride=2,name='conv1')
            self.exit0 = BN(self.exit0, phase_train=is_train,name='conv1_bn_exit0')
            self.exit0 = Relu(self.exit0,name='conv1_relu')
            self.exit0 = Conv2d(self.exit0, filters=256, k_size=3, stride=2,name='conv2')
            self.exit0 = BN(self.exit0, phase_train=is_train,name='conv2_bn')
            self.exit0 = Relu(self.exit0,name='conv2_relu')
            """
            self.exit0 = max_pooling(self.max_pool2, k_size=2, stride=2,name='maxpool1')
            #self.exit0 = max_pooling(self.exit0, k_size=2, stride=2,name='maxpool2')

            #self.exit0 = max_pooling(self.exit0, k_size=2, stride=2,name='maxpool3')
            self.exit0 = Flatten(self.exit0)
            self.exit0 = fc_layer(self.exit0, 4096,name='fc1')
            self.exit0 = Relu(self.exit0,name='fc1_relu')
            self.exit0 = Drop_out(self.exit0, 0.2, training=is_train)
            #self.exit0 = fc_layer(self.exit0, 4096,name='fc2')
            #self.exit0 = Relu(self.exit0,name='fc2_relu')
            #self.exit0 = Drop_out(self.exit0, 0.2, training=is_train)
            logits_exit0 = fc_layer(self.exit0, self.num_class,name='logits_exit0')

        with tf.variable_scope("exit1"):
            #tf.reset_default_graph()
            """self.exit0 = Conv2d(self.max_pool2, filters=256, k_size=3, stride=2,name='conv1')
            self.exit0 = BN(self.exit0, phase_train=is_train,name='conv1_bn_exit0')
            self.exit0 = Relu(self.exit0,name='conv1_relu')
            self.exit0 = Conv2d(self.exit0, filters=256, k_size=3, stride=2,name='conv2')
            self.exit0 = BN(self.exit0, phase_train=is_train,name='conv2_bn')
            self.exit0 = Relu(self.exit0,name='conv2_relu')
            """
            self.exit1 = max_pooling(self.max_pool3, k_size=2, stride=2,name='maxpool1')

            #self.exit0 = max_pooling(self.exit0, k_size=2, stride=2,name='maxpool3')
            self.exit1 = Flatten(self.exit1)
            self.exit1 = fc_layer(self.exit1, 4096,name='fc1')
            self.exit1 = Relu(self.exit1,name='fc1_relu')
            self.exit1 = Drop_out(self.exit1, 0.2, training=is_train)
            #self.exit1 = fc_layer(self.exit1, 4096,name='fc2')
            #self.exit1 = Relu(self.exit1,name='fc2_relu')
            #self.exit1 = Drop_out(self.exit1, 0.2, training=is_train)
            logits_exit1 = fc_layer(self.exit1, self.num_class,name='logits_exit1')

        with tf.variable_scope("exit2"):
            """self.exit1 = Conv2d(self.max_pool4, filters=512, k_size=3, stride=2,name='conv1')
            self.exit1 = BN(self.exit1, phase_train=is_train,name='conv1_bn')
            self.exit1 = Relu(self.exit1,name='conv1_relu')"""
            self.exit2 = max_pooling(self.max_pool4, k_size=2, stride=2,name='maxpool1')

            self.exit2 = Flatten(self.exit2)
            self.exit2 = fc_layer(self.exit2, 4096,name='fc1')
            self.exit2 = Relu(self.exit2,name='fc1_relu')
            self.exit2 = Drop_out(self.exit2, 0.2, training=is_train)
            self.exit2 = fc_layer(self.exit2, 4096,name='fc2')
            self.exit2 = Relu(self.exit2,name='fc2_relu')
            self.exit2 = Drop_out(self.exit2, 0.2, training=is_train)
            logits_exit2 = fc_layer(self.exit2, self.num_class,name='logits_exit2')



        return [logits_exit0, logits_exit1, logits_exit2, logits_exit3]

'''
def B_AlexNet_(x, is_train, num_class=10):
    # conv layer 1
    conv1 = Conv2d(x, filters=32, k_size=5, stride=1)
    conv1 = BN(conv1, phase_train=is_train)
    conv1 = Relu(conv1)
    conv1 = max_pooling(conv1, k_size=2, stride=2)

    # conv layer 2
    conv2 = Conv2d(conv1, filters=64, k_size=5, stride=1)
    conv2 = BN(conv2, phase_train=is_train)
    conv2 = Relu(conv2)
    conv2 = max_pooling(conv2, k_size=2, stride=2)

    # exit0
    exit0 = Conv2d(conv2, filters=32, k_size=5, stride=1)
    exit0 = BN(exit0, phase_train=is_train)
    exit0 = Relu(exit0)
    exit0 = Conv2d(exit0, filters=32, k_size=3, stride=1)
    exit0 = BN(exit0, phase_train=is_train)
    exit0 = Relu(exit0)
    exit0 = max_pooling(exit0, k_size=2, stride=2)
    logits_exit0 = Flatten(exit0)
    logits_exit0 = fc_layer(logits_exit0, num_class)

    # continue baseline, conv layer 3
    conv3 = Conv2d(conv2, filters=96, k_size=3, stride=1)
    conv3 = BN(conv3, phase_train=is_train)
    conv3 = Relu(conv3)

    # exit1
    exit1 = Conv2d(conv3, filters=32, k_size=3, stride=1)
    exit1 = BN(exit1, phase_train=is_train)
    exit1 = Relu(exit1)
    logits_exit1 = Flatten(exit1)
    logits_exit1 = fc_layer(logits_exit1, num_class)

    # continue baseline layer 4
    conv4 = Conv2d(conv3, filters=96, k_size=3, stride=1)
    conv4 = BN(conv4, phase_train=is_train)
    conv4 = Relu(conv4)

    # layer 5
    conv5 = Conv2d(conv4, filters=64, k_size=3, stride=1)
    conv5 = BN(conv5, phase_train=is_train)
    conv5 = Relu(conv5)
    conv5 = max_pooling(conv5, k_size=2, stride=2)

    # fc1
    fc1 = Flatten(conv5)
    fc1 = fc_layer(fc1, 256)
    fc1 = Relu(fc1)
    fc1 = Drop_out(fc1, 0.1, training=is_train)

    # fc2
    fc2 = fc_layer(fc1, 128)
    fc2 = Relu(fc2)
    fc2 = Drop_out(fc2, 0.1, training=is_train)

    # logits
    logits_exit2 = fc_layer(fc2, num_class)

    return [logits_exit0, logits_exit1, logits_exit2]
'''
