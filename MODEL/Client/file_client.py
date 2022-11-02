import socket
import time
import json


class SocketClient(object):
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
        # 链接目标
        self.sock.connect((host, port))

    def send_data(self, data, name):
        """
        用来发送数据
        :param data: 发送的数据，必须可以被json序列化
        :param name: 发送的名字
        :return: send_time 纯发送的时间
        """
        # 序列化data
        to_send = json.dumps(data)
        # 开始发送流程
        self.sock.send("{{name: {} ,length: {} }}".format(name, len(to_send)).encode())
        if self.sock.recv(len(name.encode())).decode() == name:
            # 开始计时
            time_begin = time.time()
            self.sock.send(to_send.encode())

        if self.sock.recv(len("done".encode())).decode() == "done":
            # 结束计时
            send_time = time.time() - time_begin
            # 返回发送时间
            return send_time
        else:
            raise IOError

    def recv_data(self):
        """
        收取服务端发送的数据
        :return: data，name， data是数据，name是信息
        """
        #print("socket receiving")
        message = self.sock.recv(self.buffer)
        #print(message)
        #print(len(message))
        name = str(message.decode().split(" ")[1])
        length = int(message.decode().split(" ")[3])
        self.sock.send(name.encode())

        if length <= self.buffer:
            data = self.sock.recv(length)

        else:
            data = self.sock.recv(self.buffer)
            while len(data) != length:
                data = data + self.sock.recv(self.buffer)

        data = json.loads(data.decode())
        self.sock.send("done".encode())
        return data, name


if __name__ == '__main__':
    sock = SocketClient("127.0.0.1", 1234)
    sock.send_data(list(range(65537)), "hello")
    print('hello')
    time.sleep(2)
    sock.send_data("data", "hello2")
