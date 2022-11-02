import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from MODEL.Server.model_util.loss import *
from MODEL.Server.file_server import SocketServer
from MODEL.Server.model_util.augementation import *
from MODEL.Server.model_util.Model import B_VGGNet
from MODEL.Server.model_util.utils import read_all_batches, read_val_data, write_pickle
tf.set_random_seed(123)
np.random.seed(123)

# from AlexNet import B_VGGNet

# 一些参数用来使GPU可以使用
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#from AlexNet import B_AlexNet
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Solver(object):
    """
        这个class主要干的事：
        1： 给environment提供one_step的api用来调用
        """
    def __init__(self):
        # build hp
        self.build_hp()
        # build client
        print("1")
        self.build_socket(host="192.168.0.19", port=12344)
        # build MODEL
        self.build_model()
        # 进行固定切分点的调用
        # 切分点为参数传进去
        print("2")
        self.pic = 0
        #self.one_step(action=[0.1, 0.05, 1, 5])

        #for i in range(3):
        #    self.test_data_batch, self.test_label_batch = self._get_val_batch(self.val_data, self.val_label, self.test_batch_size)


    def build_hp(self):
        """
        建立相关的超参数
        :return: 无
        """
        self.data_root = './cifar-10-batches-py'
        self.checkpoint_path = './model_checkpoints'
        self.num_class = 10
        self.input_w = 32
        self.input_h = 32
        self.input_c = 3
        # Training parameter
        # batch size
        self.train_batch_size = 128
        self.test_batch_size = 1
        # lr
        self.lr = 0.0001
        self.momentum = 0.9
        # epoch and step
        self.epochs = 50
        self.baseline_epoch = 30
        # early exit
        self.earlyexit_lossweights = [0.0, 0.0, 0.0, 1.0]
        self.earlyexit_thresholds = [0.01, 0.05, 0.05]
        # to store
        self.exit0_crrect = 0
        self.exit1_crrect = 0
        self.exit2_crrect = 0
        self.exit3_crrect = 0

    def build_socket(self, host, port):
        """
        建立socket模型
        :param host: host的ip，为string
        :param port: port， 为int
        :return: 无
        """
        self.socket = SocketServer(host, port)

    def load_data(self):
        """
        将所有的data加载进内存中
        :return: 看下面的return
        """
        train_data, train_label = read_all_batches(self.data_root, 5, [self.input_w, self.input_h, self.input_c])
        self.train_step = len(train_data) // self.train_batch_size
        val_data, val_label = read_val_data(self.data_root, [self.input_w, self.input_h, self.input_c], shuffle=False)
        self.test_step = len(val_data) // self.test_batch_size
        return (train_data, train_label), (val_data, val_label)

    def build_model(self):
        """
        建立推断的模型
        :return: 无
        """
        # create placeholder
        self.img_placeholder = tf.placeholder(dtype=tf.float32,
                                              shape=[self.test_batch_size, self.input_w, self.input_h, self.input_c])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.test_batch_size])
        self.training_flag = tf.placeholder(dtype=tf.bool, shape=[])
        self.earlyexit_lossweights_placeholder = tf.placeholder(dtype=tf.float32,
                                                                shape=[len(self.earlyexit_lossweights)])

        # create MODEL and build graph
        self.B_VGGNet_instance = B_VGGNet(num_class=self.num_class)
        [self.logits_exit0, self.logits_exit1, self.logits_exit2, self.logits_exit3] = self.B_VGGNet_instance.model(self.img_placeholder, is_train=self.training_flag)

        # prediction from branches
        self.pred0 = tf.nn.softmax(self.logits_exit0)
        self.pred1 = tf.nn.softmax(self.logits_exit1)
        self.pred2 = tf.nn.softmax(self.logits_exit2)
        self.pred3 = tf.nn.softmax(self.logits_exit3)

        # logits of branches
        #print(logits_exit0.shape, logits_exit1.shape, logits_exit2.shape)
        self.loss_exit0 = cross_entropy(self.logits_exit0, self.label_placeholder)
        self.loss_exit1 = cross_entropy(self.logits_exit1, self.label_placeholder)
        self.loss_exit2 = cross_entropy(self.logits_exit2, self.label_placeholder)
        self.loss_exit3 = cross_entropy(self.logits_exit3, self.label_placeholder)
        self.total_loss = tf.reduce_sum(tf.multiply(self.earlyexit_lossweights_placeholder, [self.loss_exit0,
                                                                                        self.loss_exit1,
                                                                                        self.loss_exit2,
                                                                                        self.loss_exit3]))

        # accuracy from brach
        self.train_acc0 = top_k_error(self.pred0, self.label_placeholder, 1)
        self.train_acc1 = top_k_error(self.pred1, self.label_placeholder, 1)
        self.train_acc2 = top_k_error(self.pred2, self.label_placeholder, 1)
        self.train_acc3 = top_k_error(self.pred3, self.label_placeholder, 1)

        # Initialize MODEL and create session
        self.sess = tf.Session()

        # Construct saver and restore graph
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.checkpoint_path, 'B_VGG.ckpt'))

        # load all data in memory
        #_, (self.val_data, self.val_label) = self.load_data()

    def test(self):
        """
        测试一次
        :return:
        """
        # inference / test
        test_data_batch, test_label_batch = self._get_val_batch(self.val_data, self.val_label, self.test_batch_size)

        # cut point0
        max_pool1_out = self.sess.run(self.B_VGGNet_instance.max_pool1, feed_dict={self.img_placeholder:test_data_batch,
                                                                                    self.label_placeholder:test_label_batch,
                                                                                    self.training_flag: False})

        # cut point1
        max_pool2_out = self.sess.run(self.B_VGGNet_instance.max_pool1,
                                      feed_dict={self.B_VGGNet_instance.max_pool1: max_pool1_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        # early exit 0
        exit0_pred, test_acc0 = self.sess.run([self.pred0, self.train_acc0],
                                              feed_dict={self.B_VGGNet_instance.max_pool2: max_pool2_out,
                                                         self.label_placeholder: test_label_batch,
                                                         self.training_flag: False})

        # eval exit 0
        print("cross entropy 0:{}".format(self._entropy_cal(exit0_pred)))
        if self._entropy_cal(exit0_pred) < self.earlyexit_thresholds[0]:
            print('Exit from branch 0: ', self._entropy_cal(exit0_pred))
            if test_acc0 == 1:
                self.exit0_crrect += 1
            return

        # cut point2
        max_pool3_out = self.sess.run(self.B_VGGNet_instance.max_pool3,
                                      feed_dict={self.B_VGGNet_instance.max_pool2: max_pool2_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        # cut point3
        max_pool4_out = self.sess.run(self.B_VGGNet_instance.max_pool4,
                                      feed_dict={self.B_VGGNet_instance.max_pool3: max_pool3_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        # early exit 1
        exit1_pred, test_acc1 = self.sess.run([self.pred1, self.train_acc1],
                                              feed_dict={self.B_VGGNet_instance.max_pool4: max_pool4_out,
                                                         self.label_placeholder: test_label_batch,
                                                         self.training_flag: False})

        # eval exit 1
        print("cross entropy 1:{}".format(self._entropy_cal(exit1_pred)))
        if self._entropy_cal(exit1_pred) < self.earlyexit_thresholds[1]:
            print('Exit from branch 1: ', self._entropy_cal(exit1_pred))
            if test_acc1 == 1:
                self.exit1_crrect += 1
            return

        # early exit 2
        exit2_pred, test_acc2 = self.sess.run([self.pred2, self.train_acc2],
                                              feed_dict={self.B_VGGNet_instance.max_pool4: max_pool4_out,
                                                         self.label_placeholder: test_label_batch,
                                                         self.training_flag: False})

        # eval exit 2
        print('Exit from branch 2 !!!')
        if test_acc2 == 1:
            self.exit2_crrect += 1
        return

    def one_step(self, action):
        """
        给ECRT提供one_step的api用来调用
        :param: action: (threshold_1, threshold_2, patition, round)
        :return: observation_next
        """
        # threshold partition and round
        thresholds = [action[0], action[1],action[2]]
        cut_point = int(action[3])
        round = action[4]
        print(round)
        print(cut_point)
        print(thresholds)
        print("fuck!!!")
        # send cut_point and thresholds
        self.socket.send_data(data=cut_point, name="cut_point")
        self.socket.send_data(data=thresholds, name="thresholds")

        # receive data from client
        self.test_network_power()
        data, name = self.socket.recv_data()
        if (data[-1] != None):
            np.array(data[-1])
        #print(data)
        #print(name)
        #print("wocaonima")
        # init
        infer_time_server = 0
        infer_time_client = 0
        recive_time = 0
        power = 0
        out_point = None

        #print("this is data!!!!!!!!!!!!!")
        #print(data)
        #print(data[0])


        if name == "cut_point:0":
            [network, power, infer_time_client] = data[1:4]
            [out_point, infer_time_server, acc] = self.inference_server(cut_point=0, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:1":
            [network, power, infer_time_client] = data[1:4]
            [out_point, infer_time_server, acc] = self.inference_server(cut_point=1, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:2":
            [network, power, infer_time_client] = data[1:4]
            [out_point, infer_time_server, acc] = self.inference_server(cut_point=2, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:3":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=3, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:4":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=4, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:5":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=5, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:6":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=6, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:7":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=7, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:8":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=8, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:9":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=9, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:10":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=10, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:11":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=11, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:12":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=12, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:13":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=13, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:14":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=14, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:15":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=15, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:16":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=16, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "cut_point:17":
            [network, power, infer_time_client] = data[1:4]
            [out_point,  infer_time_server, acc] = self.inference_server(cut_point=17, thresholds=thresholds, data=data[0], label = data[-1])
            # receive receive time from client
            recive_time, _ = self.socket.recv_data()
            band_w,_ = self.socket.recv_data()
        elif name == "exit:0":
            out_point = 0
            [network, acc, power, infer_time_client] = data[0:4]
            band_w,_ = self.socket.recv_data()
        elif name == "exit:1":
            out_point = 1
            [network, acc, power, infer_time_client] = data[0:4]
            band_w,_ = self.socket.recv_data()
        elif name == "exit:2":
            out_point = 2
            [network, acc, power, infer_time_client] = data[0:4]
            band_w,_ = self.socket.recv_data()
        elif name == "exit:final":
            out_point = 3
            [network, acc, power, infer_time_client] = data[0:4]
            band_w,_ = self.socket.recv_data()

        #数量级都在0.00XX
        network = band_w / 1024
        latency = infer_time_client + recive_time
        times = [infer_time_client, recive_time, infer_time_server]
        print("\n\ncut_point:{}".format(cut_point))
        print("name:{}".format(name))
        print("out_point:{}".format(out_point))
        print("acc:{}".format(acc))
        print("infer_time_client:{}".format(infer_time_client))
        print("Air:{}".format(recive_time))
        print("infer_time_server:{}".format(infer_time_server))
        print("latency :{}".format(latency))
        print("network:{}".format(network))
        print("recive_time:{}".format(recive_time))
        print("power:{}".format(power))

        print("\n")

        #latency = infer_time_client + infer_time_server + recive_time


        print("one step over")
        return [network], [acc, power, latency], out_point, times


    def test_network_power(self):
        """
        对网络状态和power状态进行测试
        :return: 如下
        """
        _, _ = self.socket.recv_data()

        # read power
        #power = 0
        #self.power_message_write('b')
        #self.power_message_write('e')
        #power = self.power_read()
        #name = "power"
        #self.socket.send_data(data=power, name=name)

    def inference_server(self, cut_point, thresholds, data, label):
        """
        用于server端的推断
        :param cut_point: 0，1，2，3
        :param thresholds: 给的thresh
        :param data: 输入的data，对应cut point
        :return: [out_point, acc, infer_time]
        """

        self.pic += 1
        print("inference server begins")
        print(cut_point)
        print(thresholds)


        # inference / test
        #test_data_batch, test_label_batch = self.test_data_batch, self.test_label_batch
        #test_data_batch, test_label_batch = self._get_val_batch(self.val_data, self.val_label, self.test_batch_size)
        test_label_batch = label
        print("test_label_batch")
        print(test_label_batch)
        # infer start
        infer_start = time.time()
        out_point = None



        if cut_point == 0:
            conv2_out = self.sess.run(self.B_VGGNet_instance.conv2,
                                          feed_dict={self.B_VGGNet_instance.conv1: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})

        if cut_point == 1:
            max_pool1_out = self.sess.run(self.B_VGGNet_instance.max_pool1,
                                      feed_dict={self.B_VGGNet_instance.conv2: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 1:
            max_pool1_out = self.sess.run(self.B_VGGNet_instance.max_pool1,
                                      feed_dict={self.B_VGGNet_instance.conv2: conv2_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 2:
            conv3_out = self.sess.run(self.B_VGGNet_instance.conv3,
                                      feed_dict={self.B_VGGNet_instance.max_pool1: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 2:
            conv3_out = self.sess.run(self.B_VGGNet_instance.conv3,
                                      feed_dict={self.B_VGGNet_instance.max_pool1: max_pool1_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 3:
            conv4_out = self.sess.run(self.B_VGGNet_instance.conv4,
                                      feed_dict={self.B_VGGNet_instance.conv3: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 3:
            conv4_out = self.sess.run(self.B_VGGNet_instance.conv4,
                                      feed_dict={self.B_VGGNet_instance.conv3: conv3_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 4:
            max_pool2_out = self.sess.run(self.B_VGGNet_instance.max_pool2,
                                      feed_dict={self.B_VGGNet_instance.conv4: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 4:
            max_pool2_out = self.sess.run(self.B_VGGNet_instance.max_pool2,
                                      feed_dict={self.B_VGGNet_instance.conv4: conv4_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})


        if cut_point == 5:
            # early exit 0
            exit0_pred, test_acc0 = self.sess.run([self.pred0, self.train_acc0],
                                                  feed_dict={self.B_VGGNet_instance.max_pool2: data,
                                                             self.label_placeholder: test_label_batch,
                                                             self.training_flag: False})

            print("cross entropy 0:{}".format(self._entropy_cal(exit0_pred)))
            if self._entropy_cal(exit0_pred) < thresholds[0]:
                print('Exit from branch 0: ', self._entropy_cal(exit0_pred))
                if test_acc0 == 1:
                    self.exit0_crrect += 1
                out_point = 0
                return [out_point, time.time() - infer_start, float(test_acc0)]

            # cut point 5
            conv5_out = self.sess.run(self.B_VGGNet_instance.conv5,
                                      feed_dict={self.B_VGGNet_instance.max_pool2: data,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        elif cut_point < 5:
            # early exit 0
            exit0_pred, test_acc0 = self.sess.run([self.pred0, self.train_acc0],
                                                  feed_dict={self.B_VGGNet_instance.max_pool2: max_pool2_out,
                                                             self.label_placeholder: test_label_batch,
                                                             self.training_flag: False})

            # eval exit 0
            print("cross entropy 0:{}".format(self._entropy_cal(exit0_pred)))
            if self._entropy_cal(exit0_pred) < thresholds[0]:
                print('Exit from branch 0: ', self._entropy_cal(exit0_pred))
                if test_acc0 == 1:
                    self.exit0_crrect += 1
                out_point = 0
                return [out_point, time.time() - infer_start, float(test_acc0)]

            conv5_out = self.sess.run(self.B_VGGNet_instance.conv5,
                                      feed_dict={self.B_VGGNet_instance.max_pool2: max_pool2_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 6:
            conv6_out = self.sess.run(self.B_VGGNet_instance.conv6,
                                      feed_dict={self.B_VGGNet_instance.conv5: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 6:
            conv6_out = self.sess.run(self.B_VGGNet_instance.conv6,
                                      feed_dict={self.B_VGGNet_instance.conv5: conv5_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 7:
            conv7_out = self.sess.run(self.B_VGGNet_instance.conv7,
                                      feed_dict={self.B_VGGNet_instance.conv6: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 7:
            conv7_out = self.sess.run(self.B_VGGNet_instance.conv7,
                                      feed_dict={self.B_VGGNet_instance.conv6: conv6_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 8:
            max_pool3_out = self.sess.run(self.B_VGGNet_instance.max_pool3,
                                      feed_dict={self.B_VGGNet_instance.conv7: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 8:
            max_pool3_out = self.sess.run(self.B_VGGNet_instance.max_pool3,
                                      feed_dict={self.B_VGGNet_instance.conv7: conv7_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 9:
            # early exit 0
            exit1_pred, test_acc1 = self.sess.run([self.pred1, self.train_acc1],
                                                  feed_dict={self.B_VGGNet_instance.max_pool3: data,
                                                             self.label_placeholder: test_label_batch,
                                                             self.training_flag: False})

            print("cross entropy 1:{}".format(self._entropy_cal(exit1_pred)))
            if self._entropy_cal(exit1_pred) < thresholds[1]:
                print('Exit from branch 1: ', self._entropy_cal(exit1_pred))
                if test_acc1 == 1:
                    self.exit1_crrect += 1
                out_point = 1
                return [out_point, time.time() - infer_start, float(test_acc1)]

            # cut point 5
            conv8_out = self.sess.run(self.B_VGGNet_instance.conv8,
                                      feed_dict={self.B_VGGNet_instance.max_pool3: data,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        elif cut_point < 9:
            # early exit 0
            exit1_pred, test_acc1 = self.sess.run([self.pred1, self.train_acc1],
                                                  feed_dict={self.B_VGGNet_instance.max_pool3: max_pool3_out,
                                                             self.label_placeholder: test_label_batch,
                                                             self.training_flag: False})

            # eval exit 0
            print("cross entropy 1:{}".format(self._entropy_cal(exit1_pred)))
            if self._entropy_cal(exit1_pred) < thresholds[1]:
                print('Exit from branch 1: ', self._entropy_cal(exit1_pred))
                if test_acc1 == 1:
                    self.exit1_crrect += 1
                out_point = 1
                return [out_point, time.time() - infer_start, float(test_acc1)]

            conv8_out = self.sess.run(self.B_VGGNet_instance.conv8,
                                      feed_dict={self.B_VGGNet_instance.max_pool3: max_pool3_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 10:
            conv9_out = self.sess.run(self.B_VGGNet_instance.conv9,
                                      feed_dict={self.B_VGGNet_instance.conv8: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 10:
            conv9_out = self.sess.run(self.B_VGGNet_instance.conv9,
                                      feed_dict={self.B_VGGNet_instance.conv8: conv8_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 11:
            conv10_out = self.sess.run(self.B_VGGNet_instance.conv10,
                                      feed_dict={self.B_VGGNet_instance.conv9: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 11:
            conv10_out = self.sess.run(self.B_VGGNet_instance.conv10,
                                      feed_dict={self.B_VGGNet_instance.conv9: conv9_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 12:
            max_pool4_out = self.sess.run(self.B_VGGNet_instance.max_pool4,
                                      feed_dict={self.B_VGGNet_instance.conv10: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 12:
            max_pool4_out = self.sess.run(self.B_VGGNet_instance.max_pool4,
                                      feed_dict={self.B_VGGNet_instance.conv10: conv10_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 13:
            # early exit 0
            exit2_pred, test_acc2 = self.sess.run([self.pred2, self.train_acc2],
                                                  feed_dict={self.B_VGGNet_instance.max_pool4: data,
                                                             self.label_placeholder: test_label_batch,
                                                             self.training_flag: False})

            print("cross entropy 2:{}".format(self._entropy_cal(exit2_pred)))
            if self._entropy_cal(exit2_pred) < thresholds[2]:
                print('Exit from branch 2: ', self._entropy_cal(exit2_pred))
                if test_acc2 == 1:
                    self.exit2_crrect += 1
                out_point = 2
                return [out_point, time.time() - infer_start, float(test_acc2)]

            conv11_out = self.sess.run(self.B_VGGNet_instance.conv11,
                                      feed_dict={self.B_VGGNet_instance.max_pool4: data,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        elif cut_point < 13:
            # early exit 0
            exit2_pred, test_acc2 = self.sess.run([self.pred2, self.train_acc2],
                                                  feed_dict={self.B_VGGNet_instance.max_pool4: max_pool4_out,
                                                             self.label_placeholder: test_label_batch,
                                                             self.training_flag: False})

            # eval exit 0
            print("cross entropy 2:{}".format(self._entropy_cal(exit2_pred)))
            if self._entropy_cal(exit2_pred) < thresholds[2]:
                print('Exit from branch 2: ', self._entropy_cal(exit2_pred))
                if test_acc2 == 1:
                    self.exit2_crrect += 1
                out_point = 2
                return [out_point, time.time() - infer_start, float(test_acc2)]

            conv11_out = self.sess.run(self.B_VGGNet_instance.conv11,
                                      feed_dict={self.B_VGGNet_instance.max_pool4: max_pool4_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 14:
            conv12_out = self.sess.run(self.B_VGGNet_instance.conv12,
                                      feed_dict={self.B_VGGNet_instance.conv11: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 14:
            conv12_out = self.sess.run(self.B_VGGNet_instance.conv12,
                                      feed_dict={self.B_VGGNet_instance.conv11: conv11_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 15:
            conv13_out = self.sess.run(self.B_VGGNet_instance.conv13,
                                      feed_dict={self.B_VGGNet_instance.conv12: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 15:
            conv13_out = self.sess.run(self.B_VGGNet_instance.conv13,
                                      feed_dict={self.B_VGGNet_instance.conv12: conv12_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 16:
            fc1_out = self.sess.run(self.B_VGGNet_instance.fc1,
                                      feed_dict={self.B_VGGNet_instance.conv13: data,
                                                     self.label_placeholder: test_label_batch,
                                                     self.training_flag: False})
        elif cut_point < 16:
            fc1_out = self.sess.run(self.B_VGGNet_instance.fc1,
                                      feed_dict={self.B_VGGNet_instance.conv13: conv13_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 17:
            exit3_pred, test_acc3 = self.sess.run([self.pred3, self.train_acc3],
                                                  feed_dict={self.B_VGGNet_instance.fc1: data,
                                                             self.label_placeholder: test_label_batch,
                                                             self.training_flag: False})

            print('Exit from final exit!!!')
            if test_acc3 == 1:
                self.exit3_crrect += 1
            out_point = 3
            return [out_point, time.time() - infer_start, float(test_acc3)]

        elif cut_point < 17:
            exit3_pred, test_acc3 = self.sess.run([self.pred3, self.train_acc3],
                                                  feed_dict={self.B_VGGNet_instance.fc1: fc1_out,
                                                             self.label_placeholder: test_label_batch,
                                                             self.training_flag: False})

            print('Exit from final exit!!!')
            if test_acc3 == 1:
                self.exit3_crrect += 1
            out_point = 3
            return [out_point, time.time() - infer_start, float(test_acc3)]



    def _entropy_cal(self, x):
        '''
        Function to calculate entropy as exit condition
        '''
        # print(x)
        return np.abs(np.sum(x * np.log(x + 1e-10)))

    def _get_val_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(10000 - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset + vali_batch_size, ...]
        vali_data_batch = whitening_image(vali_data_batch)
        vali_label_batch = vali_label[offset:offset + vali_batch_size]
        return vali_data_batch, vali_label_batch

    def _get_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(10000 - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset + train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, padding_size=2)
        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset + self.train_batch_size]

        return batch_data, batch_label


def main():
    solver = Solver()


if __name__ == '__main__':
    main()
