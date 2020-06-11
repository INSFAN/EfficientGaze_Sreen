#-*-coding:utf-8-*-
import sys
import os
import numpy as np
import time
# import subprocess
from subprocess import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QWidget, QPushButton, QApplication, QGridLayout, QLCDNumber, QLabel)
from PyQt5.QtGui import QCursor, QFont
import socket
import argparse 

class MyBtn(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.name = text
        self.upTime = QTimer()
        self.upTime.timeout.connect(self.update)

        # def mouseMoveEvent(self, e):
    #     print("ok")
    #
    # def mousePressEvent(self, e):
    #     print("Not Ok")

    def enterEvent(self, a0):
        # print(self.name)
        self.upTime.stop()
        self.upTime.start(2500)
        self.setStyleSheet("background-color: red")
        return super().enterEvent(a0)

    def update(self):
        global num
        global gaze_event
        # if num is not self.name:
        num = self.name
        gaze_event = True

        # print('time end ', num)


    def leaveEvent(self, a0):
        self.upTime.stop()
        self.setStyleSheet("background-color: white")
        return super().leaveEvent(a0)
    # def clicked(self):

class Example(QWidget):
    def __init__(self, posF, args):
        super().__init__()
        self.Init_UI()
        global num
        global gaze_event
        # num = ''
        gaze_event = False
        self.posF = posF
        self.total = 0
        self.correct_num = 0
        self.gaze_data = []
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建一个socket
        self.s.connect((args.ip, args.port))  # 建立连接
        num = self.s.recv(1024).decode('utf-8')  # 接收连接成功消息
        # print(self.s.recv(1024).decode('utf-8')) # 接收连接成功消息

    def Init_UI(self):
        grid = QGridLayout()
        self.setLayout(grid)
        self.setMouseTracking(True)
        self.setGeometry(0, 0, 1920, 1080)
        self.setWindowTitle('gaze ui')
        self.showFullScreen()
        self.setCursor(Qt.CrossCursor)

        self.time = QTimer(self)
        self.time.setInterval(100)
        self.time.timeout.connect(self.refresh)
        self.time.start()

        # self.x = 600
        # self.y = 160
        # self.move_avg_rate = 0.6

        self.time_xy = QTimer(self)
        self.time_xy.setInterval(83)
        self.time_xy.timeout.connect(self.setQCursorPos)
        self.time_xy.start()

        names = [
                'TurnLeft', 'MoveForward', 'TurnRight', 
                'LookLeft', '', 'LookRight', 
                'LookUp', 'SquatDown', 'LookDown' ]

        positions = [(i,j) for i in range(0,3) for j in range(0,3)]
        for position, name in zip(positions, names):
            if name == '':
                continue
            # button = QPushButton(name)
            button = MyBtn(name)
            button.setFixedSize(640, 350)
            button.setFont(QFont('微软雅黑', 50, QFont.Bold))
            grid.addWidget(button, *position)
            button.clicked.connect(self.Cli)
            # button.enterEvent(self.Time2s)
            # button.event(self, )

        # self.lcd = QLCDNumber()
        # self.lcd.setSegmentStyle(QLCDNumber.Filled)
        # grid.addWidget(self.lcd, 1, 1)
        # grid.setSpacing(10)
        # self.lcd.display('HELLO')

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont('微软雅黑', 50, QFont.Bold))
        self.label.setStyleSheet("color:red")
        grid.addWidget(self.label, 1, 1)
        self.label.setText('')
        self.show()

        # t = threading.Thread(target=tcplink, args=(sock, addr))
        # t.start()

    def Cli(self):
        sender = self.sender().text()

        self.lcd.display(sender)

    # def recvRobot():
    #     self.label.setText(self.s.recv(1024).decode('utf-8'))   

    def refresh(self):
        global gaze_event
        global num
        if gaze_event: 
            gaze_event = False
            # self.lcd.display(num)  
            self.label.setText(num) 
            self.gaze_data.append(num)
            self.s.send(num.encode('utf-8')) 
            self.label.setText(num + ' OK!')
            # self.label.setText(self.s.recv(1024).decode('utf-8'))   
            num = ''         


    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            gaze_np = np.array(self.gaze_data)
            print(gaze_np)
            np.savetxt('../output/gaze_data.txt', gaze_np)
            self.close()



    def setQCursorPos(self):
        line = self.posF.readline().decode('utf-8')
        xy = line.strip()
        xy = xy.split(" ")
        # print(len(xy))
        if len(xy) == 2:
            print(xy)
            # self.x = self.move_avg_rate * self.x + (1-self.move_avg_rate) * int(xy[0])
            # self.y = self.move_avg_rate * self.y + (1-self.move_avg_rate) * int(xy[1])
            # QCursor.setPos(self.x, self.y)
            QCursor.setPos(int(xy[0]), int(xy[1]))
        # QCursor.setPos(x, y)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.43.18",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9999,
                        help="Robot port number")

    proc = Popen("./EfficientGaze", bufsize=1024, stdin=PIPE, stdout=PIPE)  
    (fin, fout) = (proc.stdin, proc.stdout)
    time.sleep(4)

    args = parser.parse_args()
    app = QApplication(sys.argv)
    ex = Example(fout, args)
    app.exit(app.exec_())
