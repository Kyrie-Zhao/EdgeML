from model_util.layers import *
import tensorflow as tf


mu = 0
sigma = 0.1

class B_VGGNet(object):

    def __init__(self, num_class=10):
        self.num_class = num_class

    def model(self, x, is_train):
        # conv layer 1
        with tf.name_scope("Inference_1"):
            self.conv1 = Conv2d(x, filters=64, k_size=3, stride=1)
            self.conv1 = BN(self.conv1, phase_train=is_train)
            self.conv1 = Relu(self.conv1)
            self.max_pool1 = max_pooling(self.conv1, k_size=2, stride=2)

        with tf.name_scope("Inference_2"):
            self.conv2 = Conv2d(self.max_pool1, filters=128, k_size=3, stride=1)
            self.conv2 = BN(self.conv2, phase_train=is_train)
            self.conv2 = Relu(self.conv2)
            self.max_pool2 = max_pooling(self.conv2, k_size=2, stride=2)

        with tf.name_scope("exit_1"):
            self.exit0 = Conv2d(self.max_pool2, filters=256, k_size=3, stride=2)
            self.exit0 = BN(self.exit0, phase_train=is_train)
            self.exit0 = Relu(self.exit0)
            self.exit0 = Conv2d(self.exit0, filters=256, k_size=3, stride=2)
            self.exit0 = BN(self.exit0, phase_train=is_train)
            self.exit0 = Relu(self.exit0)
            self.exit0 = max_pooling(self.exit0, k_size=2, stride=2)
            self.exit0 = Flatten(self.exit0)
            self.exit0 = fc_layer(self.exit0, 4096)
            self.exit0 = Relu(self.exit0)
            self.exit0 = Drop_out(self.exit0, 0.2, training=is_train)
            self.exit0 = fc_layer(self.exit0, 4096)
            self.exit0 = Relu(self.exit0)
            self.exit0 = Drop_out(self.exit0, 0.2, training=is_train)
            logits_exit0 = fc_layer(self.exit0, self.num_class)

        with tf.name_scope("Inference_3"):
            self.conv3 = Conv2d(self.max_pool2, filters=256, k_size=3, stride=1)
            self.conv3 = BN(self.conv3, phase_train=is_train)
            self.conv3 = Relu(self.conv3)
            self.conv4 = Conv2d(self.conv3, filters=256, k_size=3, stride=1)
            self.conv4 = BN(self.conv4, phase_train=is_train)
            self.conv4 = Relu(self.conv4)
            self.max_pool3 = max_pooling(self.conv4, k_size=2, stride=2)

        with tf.name_scope("Inference_4"):
            self.conv5 = Conv2d(self.max_pool3, filters=512, k_size=3, stride=1)
            self.conv5 = BN(self.conv5, phase_train=is_train)
            self.conv5 = Relu(self.conv5)
            self.conv6 = Conv2d(self.conv5, filters=512, k_size=3, stride=1)
            self.conv6 = BN(self.conv6, phase_train=is_train)
            self.conv6 = Relu(self.conv6)
            self.max_pool4 = max_pooling(self.conv6, k_size=2, stride=2)

        with tf.name_scope("exit_2"):
            self.exit1 = Conv2d(self.max_pool4, filters=512, k_size=3, stride=2)
            self.exit1 = BN(self.exit1, phase_train=is_train)
            self.exit1 = Relu(self.exit1)
            self.exit1 = Flatten(self.exit1)
            self.exit1 = fc_layer(self.exit1, 4096)
            self.exit1 = Relu(self.exit1)
            self.exit1 = Drop_out(self.exit1, 0.2, training=is_train)
            self.exit1 = fc_layer(self.exit1, 4096)
            self.exit1 = Relu(self.exit1)
            logits_exit1 = fc_layer(self.exit1, self.num_class)

        with tf.name_scope("Inference_5"):
            self.conv7 = Conv2d(self.max_pool4, filters=512, k_size=3, stride=1)
            self.conv7 = BN(self.conv7, phase_train=is_train)
            self.conv7 = Relu(self.conv7)
            self.conv8 = Conv2d(self.conv7, filters=512, k_size=3, stride=1)
            self.conv8 = BN(self.conv8, phase_train=is_train)
            self.conv8 = Relu(self.conv8)

        with tf.name_scope("exit_3"):
            self.fc1 = Flatten(self.conv8)
            self.fc1 = fc_layer(self.fc1, 4096)
            self.fc1 = Relu(self.fc1)
            self.fc1 = Drop_out(self.fc1, 0.2, training=is_train)
            self.fc2 = fc_layer(self.fc1, 4096)
            self.fc2 = Relu(self.fc2)
            self.fc2 = Drop_out(self.fc2, 0.2, training=is_train)
            logits_exit2 = fc_layer(self.fc2, self.num_class)

        return [logits_exit0, logits_exit1, logits_exit2]

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
