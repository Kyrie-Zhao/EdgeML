import socket
import time
import json
import re


class SocketServer(object):
    """用来建立sockt链接"""
    def __init__(self, host, port):
        """
        初始化的时候就建立好socket
        :param host: 是一个String，放目标的ip地址
        :param port: 是一个int， 放目标端口
        """
        self.host = host
        self.port = port
        self.buffer = 4028
        # 建立socket
        self.sock = socket.socket(family=socket.AF_INET,
                                  type=socket.SOCK_STREAM)

        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定
        self.sock.bind((host, port))
        # 开始监听目标端口
        self.sock.listen(0)
        # 接受链接
        self.connection, address = self.sock.accept()


    def send_data(self, data, name):
        """
        用来发送数据
        :param data: 发送的数据，必须可以被json序列化
        :param name: 发送的名字
        :return: send_time 纯发送的时间
        """
        # 序列化data
        to_send = json.dumps(data)
        #print(to_send)
        #print("tosend")
        self.connection.send("{{name: {} ,length: {} }}".format(name, len(to_send)).encode())
        if self.connection.recv(len(name.encode())).decode() == name:
            # 开始计时
            time_begin = time.time()
            self.connection.send(to_send.encode())

        if self.connection.recv(len("done".encode())).decode() == "done":
            # 结束计时
            #send_time = time.time() - time_begin
            # 返回发送时间
            send_time = time.time() - time_begin
            return send_time
        else:
            raise IOError

    def recv_data(self):
        """
        收取服务端发送的数据
        :return: data，name， data是数据，name是信息
        """
        print("reciving")
        message = self.connection.recv(self.buffer)
        #print(message)
        if message == "":
            print("wrong happend")
            message = self.connection.recv(self.buffer)
        # print(message.decode())
        name = str(message.decode().split(" ")[1])
        length = int(message.decode().split(" ")[3])
        # print("length: {}".format(length))
        self.connection.send(name.encode())

        if length <= self.buffer:
            data = self.connection.recv(length)

        else:
            data = self.connection.recv(self.buffer)
            while len(data) != length:
                data = data + self.connection.recv(self.buffer)
        #print("data_length: {}".format(len(data)))

        data = json.loads(data.decode())
        self.connection.send("done".encode())
        return data, name


if __name__ == '__main__':
    sock = SocketServer("127.0.0.1", 1234)
    data, name = sock.recv_data()
    print(data, name)
    time.sleep(1)
    data, name = sock.recv_data()
    print(data, name)
