import serial as serial
import time
import thread
import os
import numpy

# init averagePower to retreive
global averagePower

def retreivePower():
    tmp = averagePower
    averagePower = 0
    return tmp

def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

# serial
ser = serial.Serial('/dev/ttyACM0', 9600)

# path
read_path = "/home/nvidia/project/test/pipeline.in"
write_path = "/home/nvidia/project/test/pipeline.out"

# if already exsist: remove
if os.path.exists(read_path):
    os.remove(read_path)
if os.path.exists(write_path):
    os.remove(write_path)

# init pipe
os.mkfifo(write_path)
os.mkfifo(read_path)

# file handle
rf = os.open(read_path, os.O_RDONLY | os.O_NONBLOCK)
wf = os.open(write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)

# init power list
powerList = []
# init read data tmp
tmp = ""
# data = ""

# serial offset remove
# Todo: fix some problem
while(1):
    tmp = ser.read(1)
    if(tmp == "W"):
      print("offset removed")
      print(ser.read(1))
      break

# main
while(1):
    # init counter and trigger
    count = 0
    append_trigger = False
    
    # for every round
    while(1):
        # read data from serial
        data = ser.read(12)
        # convert data into float
        power = float(data[-6:-2])
        print(power)
        print(powerList)
        print(np.shape(powerList))

        # read user command
        try:
            judge = os.read(rf, 1)
        except OSError:
            print("OSError")
            time.sleep(0.2)
            pass
        time.sleep(0.1)

        print("command :{}".format(judge))
        
        # if command == b
        if(judge == "b"):
              append_trigger = True

        # if command == e
        elif(judge == "e"):
          append_trigger = False
          averagePower = averagenum(powerList)
          os.write(wf, str(averagePower))
          powerList = []
          averagePower = 0
          break

        # accumulate to average
        if append_trigger:
          powerList.append(power)
          print(count)
          count+=1
# close file
os.close(rf)
os.close(wf)

