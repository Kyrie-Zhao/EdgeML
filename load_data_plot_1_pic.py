import pandas as pd
import numpy as np
import pickle
from MODEL.Server.model_util.augementation import *
from MODEL.Server.model_util.utils import read_all_batches, read_val_data, write_pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
plt.style.use("ggplot")
"""
要统计的变量：
1. bandwidth -> epoch step
2. latency -> epoch step
3. power -> epoch step
4. accuracy -> epoch step
5. output_point -> epoch step
6. partition_point -> epoch
7. threshold -> epoch
8. round -> epoch
9. reward -> epoch

8. 用户的latency_bound
9. 用户的acc_bound

要计算的量:
flops


要改变的变量：
1. 用户的latency_bound
2. 用户的power_bound

3. rolling的值-reward
4. rolling的值-次方n

baseline的时候
5. 固定出点
6. 固定branch

7. bandwidth

8. input图片变化或不变
"""
np.random.seed(123)

step = []
bandwidth = []
latency = []
power = []
accuracy = []
output_point = []
times = []

epoch = []
bandwidth_epoch = []
latency_epoch = []
power_epoch = []
accuracy_epoch = []
output_point_epoch = []
partition_point = []
threshold = []
round = []
reward = []


def load_data():
    """
    将所有的data加载进内存中
    :return: 看下面的return
    """
    train_data, train_label = read_all_batches("./cifar-10-batches-py", 5, [32, 32, 3])
    train_step = len(train_data) // 64
    val_data, val_label = read_val_data("./cifar-10-batches-py", [32, 32, 3], shuffle=False)
    test_step = len(val_data) // 1
    return (train_data, train_label), (val_data, val_label)

def _get_val_batch(vali_data, vali_label, vali_batch_size):
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
    # vali_data_batch = whitening_image(vali_data_batch)
    vali_label_batch = vali_label[offset:offset + vali_batch_size]
    return vali_data_batch, vali_label_batch


def plot_data(x_data, y_data, title=' ', x_label=' ', y_label=' ', save=True):
    """plot data"""
    # new figure
    plt.figure()
    line_style = ["-",":","-."]
    color = ["r", "g", "b"]

    for i in range(2):
        # plot, set x, y label and title
        plt.plot(x_data[i*300:(i+1)*300], y_data[i*300:(i+1)*300], linestyle=line_style[i], color=color[i])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        red_patch = Line2D([],[],linestyle=line_style[0], color='r', label='first')
        green_patch = Line2D([],[],linestyle=line_style[1], color='g', label='second')
        # blue_patch = Line2D([],[],linestyle=line_style[2], color='b', label='third')

        plt.legend(handles=[red_patch, green_patch])

        # if save then save
        if save:
            plt.savefig('./{}.eps'.format(title))

    # show the fig
    plt.show()
    plt.close(0)


(train_data, train_label), (val_data, val_label) = load_data()

for i in range(3):
    val_batch_target, val_label_target = _get_val_batch(val_data, val_label, 1)
    if i == 2:
        image = np.cast[int](np.squeeze(np.array(val_batch_target)))
        plt.imshow(image)
        plt.show()


file = open("./data_store/data_high_high_num_2_touse.pk", "rb")
data = pickle.load(file)

step.append(data["step"]["step"])
bandwidth.append(data["step"]["band_width"])
latency.append(data["step"]["latency"])
power.append(data["step"]["power"])
accuracy.append(data["step"]["accuracy"])
output_point.append(data["step"]["output_point"])
times.append(data["step"]["times"])

epoch.append(data["epoch"]["epoch"])
bandwidth_epoch.append(data["epoch"]["band_width"])
latency_epoch.append(data["epoch"]["latency"])
power_epoch.append(data["epoch"]["power"])
accuracy_epoch.append(data["epoch"]["accuracy"])
output_point_epoch.append(data["epoch"]["output_point"])
partition_point.append(data["epoch"]["partition_point"])
threshold.append(data["epoch"]["threshold"])
round.append(data["epoch"]["round"])
reward.append(data["epoch"]["reward"])

# config:
latency_bound = data["config"]["latency_bound"]
power_bound = data["config"]["power_bound"]
rolling_reward = data["config"]["rolling_reward"]
rolling_n = data["config"]["rolling_n"]
fix_partition = data["config"]["fix_partition"]
fix_branch = data["config"]["fix_branch"]

file.close()

file = open("./data_store/data_low_low_num_2_touse.pk", "rb")
data = pickle.load(file)

step.append(data["step"]["step"])
bandwidth.append(data["step"]["band_width"])
latency.append(data["step"]["latency"])
power.append(data["step"]["power"])
accuracy.append(data["step"]["accuracy"])
output_point.append(data["step"]["output_point"])

epoch.append(data["epoch"]["epoch"])
bandwidth_epoch.append(data["epoch"]["band_width"])
latency_epoch.append(data["epoch"]["latency"])
power_epoch.append(data["epoch"]["power"])
accuracy_epoch.append(data["epoch"]["accuracy"])
output_point_epoch.append(data["epoch"]["output_point"])
partition_point.append(data["epoch"]["partition_point"])
threshold.append(data["epoch"]["threshold"])
round.append(data["epoch"]["round"])
reward.append(data["epoch"]["reward"])

# config:
latency_bound = data["config"]["latency_bound"]
power_bound = data["config"]["power_bound"]
rolling_reward = data["config"]["rolling_reward"]
rolling_n = data["config"]["rolling_n"]
fix_partition = data["config"]["fix_partition"]
fix_branch = data["config"]["fix_branch"]

file.close()

step = np.array(step).flatten().tolist()
bandwidth = np.array(bandwidth).flatten().tolist()
latency = np.array(latency).flatten().tolist()
power = np.array(power).flatten().tolist()
accuracy = np.array(accuracy).flatten().tolist()
output_point = np.array(output_point).flatten().tolist()

epoch = np.array(epoch).flatten()
print(np.shape(epoch))
epoch_2 = epoch[300:600]+300
epoch[300:600] = epoch_2

bandwidth_epoch = np.array(bandwidth_epoch).flatten().tolist()
latency_epoch = np.array(latency_epoch).flatten().tolist()
power_epoch = np.array(power_epoch).flatten().tolist()
accuracy_epoch = np.array(accuracy_epoch).flatten().tolist()
output_point_epoch = np.array(output_point_epoch).flatten().tolist()
partition_point = np.array(partition_point).flatten().tolist()
print(np.shape(threshold))
threshold = np.array(threshold).flatten().tolist()
threshold = np.reshape(threshold,[600,2])
round = np.array(round).flatten().tolist()
reward = np.array(reward).flatten().tolist()


count = 0
for i in reward[0:300]:
    if i > 0:
        count += 1
print(count/ 300)

count = 0
for i in reward[300:600]:
    if i > 0:
        count += 1
print(count/ 300)



# plot all data
plot_data(epoch, bandwidth_epoch, title="bandwidth_epoch")
plot_data(epoch, latency_epoch, title="latency_epoch")
plot_data(epoch, power_epoch, title="power_epoch")
plot_data(epoch, accuracy_epoch, title="accuracy_epoch")
plot_data(epoch, output_point_epoch, title="output_point_epoch")
plot_data(epoch, partition_point, title="partition_point")
plot_data(epoch, threshold, title="threshold")
plot_data(epoch, round, title="round")
plot_data(epoch, reward, title="reward")
# plot_data(step, bandwidth, title="bandwidth")
# plot_data(step, latency, title="latency")
# plot_data(step, power, title="power")
# plot_data(step, accuracy, title="accuracy")
# plot_data(step, output_point, title="output_point")



"""
step = 0
bandwidth = []
latency = []
power = []
accuracy = []
output_point = []


epoch = 0
bandwidth_epoch = []
latency_epoch = []
power_epoch = []
accuracy_epoch = []
output_point_epoch = []
partition_point = []
threshold = []
round = []
reward = []

# config:
latency_bound = 0
power_bound = 0
rolling_reward = 0
rolling_n = 0
fix_partition = False
fix_branch = False


step = {
    "step"         : step,
    "band_width"   : bandwidth,
    "latency"      : latency,
    "power"        : power,
    "accuracy"     : accuracy,
    "output_point" : output_point,
}

epoch = {
    "epoch"          : epoch,
    "band_width"     : bandwidth_epoch,
    "latency"        : latency_epoch,
    "power"          : power_epoch,
    "accuracy"       : accuracy_epoch,
    "output_point"   : output_point_epoch,
    "partition_point": partition_point,
    "threshold"      : threshold,
    "round"          : round,
    "reward"         : reward
}

config = {
    "latency_bound" : latency_bound,
    "power_bound"   : power_bound,
    "rolling_reward": rolling_reward,
    "rolling_n"     : rolling_n,
    "fix_partition" : fix_partition,
    "fix_branch"    : fix_branch
}

all = {
    "step"  : step,
    "epoch" : epoch,
    "config": config
}
"""
