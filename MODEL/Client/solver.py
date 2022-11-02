import os
import logging
import psutil
import time
import sys
import numpy as np
import argparse
import tensorflow.compat.v1 as tf
import subprocess
import threading
tf.disable_v2_behavior()


from model_util.loss import *
from file_client import SocketClient
from model_util.augementation import *
from model_util.Model import B_VGGNet
from model_util.utils import read_all_batches, read_val_data, write_pickle

tf.set_random_seed(123)
#np.random.seed(123)

# from AlexNet import B_VGGNet

# 一些参数用来使GPU可以使用
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES']='0'

class Solver(object):
    """
    这个class主要干的事：
    1： 直接运行，作为客户端的主程序
    """
    def __init__(self):
        self.signal = 0
        self.gpu_value = 0
        self.GPU_pointer = subprocess.Popen(["tegrastats","--interval", "1"],stdout = subprocess.PIPE)
        # 建立超参数
        self.build_hp()
        # 创建socket
        self.build_socket(host="192.168.0.19", port=12344)
        # 加载模型
        self.build_model()

        #self.GPU_pointer = subprocess.Popen("tegrastats",stdout = subprocess.PIPE)
        # 推断
        # for i in range(2):
        #     self.test_data_batch, self.test_label_batch = self._get_val_batch(self.val_data, self.val_label, self.test_batch_size)
        while True:
            # 在与server交互的时候（ecrt）
            self.one_step()


    def gpu(self):
        gpu_value_tmp = 0
        self.signal = 0
        tmp = 0
        ls = []

        #GPU_pointer, err = GPU_pointer.communicate()
        while(self.signal == 0):
            tmp += 1
            #time.sleep(0.1)
            #print("reading gpu")

            tegrastats_value = self.GPU_pointer.stdout.readline().decode()
            #print(tegrastats_value)
            gpu_value_tmp = float(tegrastats_value.split("GR3D_FREQ")[1].split("%")[0])/10
            ls.append(gpu_value_tmp)

            #print(gpu_value)
            #print(ls)
        #ls = [tmp for tmp in ls if tmp>0 ]
        self.gpu_value = float(np.mean(ls))
        print("fuk11111111111111")
        print(self.gpu_value)
        #print(ls)
        print("fuk222222222222222")


    def gpu_old(self):
        print("reading gpu")
        tegrastats_value = self.GPU_pointer.stdout.readline().decode()
        #print(tegrastats_value)
        gpu_value = float(tegrastats_value.split("GR3D_FREQ")[1].split("%")[0])/10
        print(gpu_value)
        return gpu_value

    def build_hp(self):
        """
        建立相关的超参数
        :return: 无
        """
        self.data_root = '../../cifar-10-batches-py'
        self.checkpoint_path = '../../model_checkpoints'
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

    def build_socket(self, host, port):
        """
        建立socket模型
        :param host: host的ip，为string
        :param port: port， 为int
        :return: 无
        """
        self.socket = SocketClient(host, port)

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
        self.img_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.test_batch_size, self.input_w, self.input_h, self.input_c])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.test_batch_size])
        self.training_flag = tf.placeholder(dtype=tf.bool, shape=[])
        self.earlyexit_lossweights_placeholder = tf.placeholder(dtype=tf.float32, shape=[len(self.earlyexit_lossweights)])
        #self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

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
        self.sess = tf.Session(config=config)

        # Construct saver and restore graph
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.checkpoint_path, 'B_VGG.ckpt'))

        # load all data in memory
        _, (self.val_data, self.val_label) = self.load_data()

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
        max_pool2_out = self.sess.run(self.B_VGGNet_instance.max_pool2,
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

    def one_step(self):
        """
        作为class的主函数
        :return:
        """
        print("One_step")
        # receive cut point and thresholds
        #time.sleep(5)
        cut_point, _ = self.socket.recv_data()
        thresholds, _ = self.socket.recv_data()
        #print(cut_point)
        #print("damn1")
        #print(thresholds)
        #print("damn")
        # inference to cut point or exit
        inference_out = self.inference_client(cut_point=cut_point, thresholds=thresholds)
        print("get inference out")
        # data to send
        name = inference_out[0]
        data = inference_out[1:6]


        #print("this is data!!!!")
        #print(data)
        print(name)
        #print(data)
        #print(data)
        print("fuking bug")
        # if send cut point
        if name[0] is "c":
            send_over = self.socket.send_data(data=data, name=name)
            print("data")
            self.socket.send_data(data=send_over, name="send_time")
            print(sys.getsizeof(data))
            print(sys.getsizeof(data)/send_over)
            self.socket.send_data(data=sys.getsizeof(data)/send_over, name="band")
        # if send exit
        else:
            send_time = self.socket.send_data(data=data, name=name)
            print(sys.getsizeof(data))
            print(sys.getsizeof(data)/send_time)
            self.socket.send_data(data=sys.getsizeof(data)/send_time, name="band")

        print("One Step Stop")



    def test_network_power(self):
        """
        对网络状态和power状态进行测试
        :return: 如下
        """
        send_time = self.socket.send_data(data="test", name="test")

        #power_now, _ = self.socket.recv_data()

        return send_time


    def power_message_write(self, to_send, write_path="/home/nvidia/project/test/pipeline.in"):
        # def power message
        wf = os.open(write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)
        os.write(wf, '{}'.format(to_send).encode())
        os.close(wf)

    def power_read(self, read_path="/home/nvidia/project/test/pipeline.out"):
        rf = os.open(read_path, os.O_RDONLY)
        s = os.read(rf, 1024)
        os.close(rf)
        return float(s.decode())

    """
    How to use:
    # power start
    power_message_write('b', write_path)
    # power stop
    power_message_write('e', write_path)
    # read power
    power_to_send = power_read(read_path)
    """

    def inference_client(self, cut_point, thresholds):
        """
        根据cut和thresh进行inference, 在服务器的时候取消掉注释
        :param cut_point: 传入的cut
        :param thresholds: 传入的thresh
        :return: [name, data, acc, power, infer_time]
        """


        # inference / test
        # test_data_batch, test_label_batch = self.test_data_batch, self.test_label_batch
        test_data_batch, test_label_batch = self._get_val_batch(self.val_data, self.val_label, self.test_batch_size)
        print("test_label_batch")
        print(test_label_batch)
        latency_stop = 0
        # infer start
        infer_start = time.time()
        # init power write
        #power = 0
        power = 0
        cpu_list = []
        p1 = psutil.Process(os.getpid())
        #self.power_message_write('b')
        #print(psutil.cpu_percent(None))
        #print("bitch")

        #power = self.gpu()
        #power = (psutil.cpu_percent(None))
        # cut point 0
        #print(thresholds)
        #print("fuck!!!")
        infer_start = time.time()
        t = threading.Thread(target=self.gpu)
        t.start()
        conv1_out = self.sess.run(self.B_VGGNet_instance.conv1,
                                      feed_dict={self.img_placeholder: test_data_batch,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})


        if cut_point == 0:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 0")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:0", conv1_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]


        # cut point1
        conv2_out = self.sess.run(self.B_VGGNet_instance.conv2,
                                      feed_dict={self.B_VGGNet_instance.conv1: conv1_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})


        if cut_point == 1:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 1")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:1", conv2_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # cut point 2
        max_pool1_out = self.sess.run(self.B_VGGNet_instance.max_pool1,
                                      feed_dict={self.B_VGGNet_instance.conv2: conv2_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 2:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 2")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:2", max_pool1_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # cut point3
        conv3_out = self.sess.run(self.B_VGGNet_instance.conv3,
                                      feed_dict={self.B_VGGNet_instance.max_pool1: max_pool1_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 3:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 3")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:3", conv3_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # cut point4
        conv4_out = self.sess.run(self.B_VGGNet_instance.conv4,
                                      feed_dict={self.B_VGGNet_instance.conv3: conv3_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 4:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 4")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:4", conv4_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # cut point5
        max_pool2_out = self.sess.run(self.B_VGGNet_instance.max_pool2,
                                      feed_dict={self.B_VGGNet_instance.conv4: conv4_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 5:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 5")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:5", max_pool2_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # early exit 0
        exit0_pred, test_acc0 = self.sess.run([self.pred0, self.train_acc0],
                                              feed_dict={self.B_VGGNet_instance.max_pool2: max_pool2_out,
                                                         self.label_placeholder: test_label_batch,
                                                         self.training_flag: False})


        print("cross entropy 0:{}".format(self._entropy_cal(exit0_pred)))
        if self._entropy_cal(exit0_pred) < thresholds[0]:
            latency_stop = time.time()-infer_start
            print('Exit from branch 0: ', self._entropy_cal(exit0_pred))
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["exit:0", self.test_network_power(), float(test_acc0), power, latency_stop, None]

        # cut point6
        conv5_out = self.sess.run(self.B_VGGNet_instance.conv5,
                                      feed_dict={self.B_VGGNet_instance.max_pool2: max_pool2_out,
                                                 self.label_placeholder: test_label_batch,
                                                 self.training_flag: False})

        if cut_point == 6:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 6")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:6", conv5_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]


        # cut point 7
        conv6_out = self.sess.run(self.B_VGGNet_instance.conv6,
                                  feed_dict={self.B_VGGNet_instance.conv5: conv5_out,
                                             self.label_placeholder: test_label_batch,
                                             self.training_flag: False})

        if cut_point == 7:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 7")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:7", conv6_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # cut point 8
        conv7_out = self.sess.run(self.B_VGGNet_instance.conv7,
                                  feed_dict={self.B_VGGNet_instance.conv6: conv6_out,
                                             self.label_placeholder: test_label_batch,
                                             self.training_flag: False})

        if cut_point == 8:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 8")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:8", conv7_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # cut point 9
        max_pool3_out = self.sess.run(self.B_VGGNet_instance.max_pool3,
                                  feed_dict={self.B_VGGNet_instance.conv7: conv7_out,
                                             self.label_placeholder: test_label_batch,
                                             self.training_flag: False})

        if cut_point == 9:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 9")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:9", max_pool3_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]


        # early exit 1
        exit1_pred, test_acc1 = self.sess.run([self.pred1, self.train_acc1],
                                              feed_dict={self.B_VGGNet_instance.max_pool3: max_pool3_out,
                                                         self.label_placeholder: test_label_batch,
                                                         self.training_flag: False})


        print("cross entropy 1:{}".format(self._entropy_cal(exit1_pred)))
        if self._entropy_cal(exit1_pred) < thresholds[1]:
            latency_stop = time.time()-infer_start
            print('Exit from branch 1: ', self._entropy_cal(exit1_pred))
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)

            return ["exit:1", self.test_network_power(), float(test_acc1), power, latency_stop,None]


        # cut point 10
        conv8_out = self.sess.run(self.B_VGGNet_instance.conv8,
                                  feed_dict={self.B_VGGNet_instance.max_pool3: max_pool3_out,
                                             self.label_placeholder: test_label_batch,
                                             self.training_flag: False})

        if cut_point == 10:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 10")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = float(psutil.cpu_percent(None))/10
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:10", conv8_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]


        # cut point 11
        conv9_out = self.sess.run(self.B_VGGNet_instance.conv9,
                                  feed_dict={self.B_VGGNet_instance.conv8: conv8_out,
                                             self.label_placeholder: test_label_batch,
                                             self.training_flag: False})

        if cut_point == 11:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 11")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:11", conv9_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # cut point 12
        conv10_out = self.sess.run(self.B_VGGNet_instance.conv10,
                                  feed_dict={self.B_VGGNet_instance.conv9: conv9_out,
                                             self.label_placeholder: test_label_batch,
                                             self.training_flag: False})

        if cut_point == 12:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 12")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:12", conv10_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # cut point 13
        max_pool4_out = self.sess.run(self.B_VGGNet_instance.max_pool4,
                                  feed_dict={self.B_VGGNet_instance.conv10: conv10_out,
                                             self.label_placeholder: test_label_batch,
                                             self.training_flag: False})

        if cut_point == 13:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 13")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:13", max_pool4_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]


        # early exit 2
        exit2_pred, test_acc2 = self.sess.run([self.pred2, self.train_acc2],
                                              feed_dict={self.B_VGGNet_instance.max_pool4: max_pool4_out,
                                                         self.label_placeholder: test_label_batch,
                                                         self.training_flag: False})


        print("cross entropy 2:{}".format(self._entropy_cal(exit2_pred)))
        if self._entropy_cal(exit2_pred) < thresholds[2]:
            latency_stop = time.time()-infer_start
            print('Exit from branch 2: ', self._entropy_cal(exit2_pred))
            # self.power_message_write('e')
            # power = self.power_read()
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["exit:2", self.test_network_power(), float(test_acc2), power, latency_stop,None]

        # cut point 14
        conv11_out = self.sess.run(self.B_VGGNet_instance.conv11,
                                  feed_dict={self.B_VGGNet_instance.max_pool4: max_pool4_out,
                                             self.label_placeholder: test_label_batch,
                                             self.training_flag: False})

        if cut_point == 14:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 14")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:14", conv11_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # cut point 15
        conv12_out = self.sess.run(self.B_VGGNet_instance.conv12,
                                  feed_dict={self.B_VGGNet_instance.conv11: conv11_out,
                                             self.label_placeholder: test_label_batch,
                                             self.training_flag: False})

        if cut_point == 15:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 15")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:15", conv12_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # cut point 16
        conv13_out = self.sess.run(self.B_VGGNet_instance.conv13,
                                  feed_dict={self.B_VGGNet_instance.conv12: conv12_out,
                                             self.label_placeholder: test_label_batch,
                                             self.training_flag: False})

        if cut_point == 16:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 16")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1
            print(power)
            print(latency_stop)
            return ["cut_point:16", conv13_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]

        # cut point 17
        fc1_out = self.sess.run(self.B_VGGNet_instance.fc1,
                                  feed_dict={self.B_VGGNet_instance.conv13: conv13_out,
                                             self.label_placeholder: test_label_batch,
                                             self.training_flag: False})

        if cut_point == 17:
            latency_stop = time.time()-infer_start
            print("exit at cut_point 17")
            # self.power_message_write('e')
            # power = self.power_read()
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1

            print(power)
            print(latency_stop)
            return ["cut_point:17", fc1_out.tolist(), self.test_network_power(), power, latency_stop, test_label_batch.tolist()]


        if cut_point == 18:
                # final exit
            exit3_pred, test_acc3 = self.sess.run([self.pred3, self.train_acc3],
                                                  feed_dict={self.B_VGGNet_instance.fc1: fc1_out,
                                                             self.label_placeholder: test_label_batch,
                                                             self.training_flag: False})

            latency_stop = time.time()-infer_start
            print('Exit from final !!!')
            # self.power_message_write('e')
            # power = self.power_read()
            #power = self.gpu()
            power = self.gpu_value
            self.signal = 1
            print(power)
            print(latency_stop)
            return ["exit:final", self.test_network_power(), float(test_acc3), power, latency_stop,None]



    def _entropy_cal(self, x):
        ''' Function to calculate entropy as exit condition
        '''
        #print(x)
        return np.abs(np.sum(x * np.log(x+1e-10)))

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
        vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
        vali_data_batch = whitening_image(vali_data_batch)
        vali_label_batch = vali_label[offset:offset+vali_batch_size]
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

        batch_data = train_data[offset:offset+train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, padding_size=2)
        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset+self.train_batch_size]

        return batch_data, batch_label


def main():
    solver = Solver()

if __name__ == '__main__':
    main()
