#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import threading
import time
import argparse
import math
from naoqi import ALProxy

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

def gazeMove(robotIP="192.168.43.18", PORT=9559, x=0, y=0, theta=0):
    motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)
    # Wake up robot
    motionProxy.wakeUp()
    # Send robot to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)
    # Example showing how Initialize move process.
    motionProxy.moveInit()
    # Example showing the moveTo command
    # The units for this command are meters and radians
    # x  = 0.2
    # y  = 0.2
    # theta  = math.pi/2
    motionProxy.moveTo(x, y, theta)
    # Will block until move Task is finished
    ########
    # NOTE #
    ########
    # If moveTo() method does nothing on the robot,
    # read the section about walk protection in the
    # Locomotion control overview page.
    # Go to rest position
    # motionProxy.rest()

def squatDown(robotIP="192.168.43.18", PORT=9559):
    motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    # Go to rest position
    motionProxy.rest()


def rotateHead(robotIP="192.168.43.18", PORT=9559, head_cmd=''):
    # yaw ->[-120°,120°]，ptich->[-39°,39°]

    if head_cmd == 'LookLeft':
        names = "HeadYaw"
        angleLists = [1.0, 0.0]
    elif head_cmd == 'LookRight':
        names = "HeadYaw"
        angleLists = [-1.0, 0.0]
    elif head_cmd == 'LookUp':
        names = "HeadPitch"
        angleLists = [-0.5, 0.0]
    elif head_cmd == 'LookDown':
        names = "HeadPitch"
        angleLists = [0.5, 0.0]
    else:
        return

    motion  = ALProxy("ALMotion", robotIP, PORT)
    # Set stiffness on for Head motors
    motion.setStiffnesses("Head", 1.0)
    # names = "HeadYaw"
     #各个关节的名称可以在sdk说明文档里找到  
    # angleLists = [1.0,0.0]     #关节要转动的角度  
    timeLists = [1.0,2.0]      #到达指定角度的指定时间  
    isAbsoulte = True        #true 代表绝对角度 
    motion.angleInterpolation(names,angleLists,timeLists,isAbsoulte)
    # Gently set stiff off for Head motors
    motion.setStiffnesses("Head", 0.0)

def parseCmd(cmd_str):
    print(cmd_str)
    if cmd_str == 'TurnLeft':
        print('action TurnLeft' )
        gazeMove(x=0,y=0,theta=math.pi/2)
    elif cmd_str == 'MoveForward':
        print('action MoveForward' )
        gazeMove(x=0.2,y=0,theta=0)
    elif cmd_str == 'TurnRight':
        print('action TurnRight' )
        gazeMove(x=0,y=0,theta=-math.pi/3)
    elif cmd_str == 'SquatDown':
        print('action SquatDown' )
        squatDown()
    else:
        rotateHead(head_cmd=cmd_str)
    # elif cmd_str is 'HeadYaw+':
    #     pass
    # elif cmd_str is 'HeadYaw-':
    #     pass
    # elif cmd_str is 'HeadPitch+':
    #     pass
    # elif cmd_str is 'HeadPitch-':
    #     pass
    # else cmd_str is 'HeadRest':
    #     pass


def tcplink(sock, addr):
    print('Accept new connection from %s:%s...' % addr)
    sock.send(b'Welcome!')
    while True:
        data = sock.recv(1024)
        # time.sleep(1)
        if not data or data.decode('utf-8') == 'exit':
            break
        # sock.send(('%s OK!' % data.decode('utf-8')).encode('utf-8'))
        parseCmd(str(data.decode('utf-8')))
    sock.close()
    print('Connection from %s:%s closed.' % addr)


def main(IP, PORT):
    # 创建一个socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 监听端口:
    s.bind((IP, PORT))

    s.listen(5)
    print('Waiting for connection...')

    while True:
        # 接受一个新连接:
        sock, addr = s.accept()
        # 创建新线程来处理TCP连接:
        t = threading.Thread(target=tcplink, args=(sock, addr))
        t.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.43.18",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9999,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)

