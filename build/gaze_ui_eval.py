#-*-coding:utf-8-*-
import sys
import os
import numpy as np
import time
# import subprocess
from subprocess import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QWidget, QPushButton, QApplication, QGridLayout, QLCDNumber)
from PyQt5.QtGui import QCursor, QFont

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
        if num is not self.name:
            num = self.name
            gaze_event = True

        # print('time end ', num)


    def leaveEvent(self, a0):
        self.upTime.stop()
        self.setStyleSheet("background-color: white")
        return super().leaveEvent(a0)
    # def clicked(self):

class Example(QWidget):
    def __init__(self, posF):
        super().__init__()
        self.Init_UI()
        global num
        global gaze_event
        num = ''
        gaze_event = False
        self.lcd_num = '0'
        self.posF = posF
        self.total = 0
        self.correct_num = 0
        self.gaze_data = []

    def Init_UI(self):
        grid = QGridLayout()
        self.setLayout(grid)
        self.setMouseTracking(True)
        self.setGeometry(0, 0, 1920, 1080)
        self.setWindowTitle('click show')
        self.showFullScreen()
        self.setCursor(Qt.CrossCursor)

        self.time = QTimer(self)
        self.time.setInterval(100)
        self.time.timeout.connect(self.refresh)
        self.time.start()

        self.x = 600
        self.y = 160
        self.move_avg_rate = 0.6

        self.time_xy = QTimer(self)
        self.time_xy.setInterval(83)
        self.time_xy.timeout.connect(self.setQCursorPos)
        self.time_xy.start()

        names = [
                '0', '1', '2', 
                '7', '', '3', 
                '6', '5', '4' ]

        positions = [(i,j) for i in range(0,3) for j in range(0,3)]
        for position, name in zip(positions, names):
            if name == '':
                continue
            # button = QPushButton(name)
            button = MyBtn(name)
            button.setFixedSize(640, 350)
            button.setFont(QFont('微软雅黑', 100, QFont.Bold))
            grid.addWidget(button, *position)
            button.clicked.connect(self.Cli)
            # button.enterEvent(self.Time2s)
            # button.event(self, )

        self.lcd = QLCDNumber()
        grid.addWidget(self.lcd, 1, 1)
        # grid.setSpacing(10)
        self.show()

    def Cli(self):
        sender = self.sender().text()

        self.lcd.display(sender)

    def refresh(self):
        global gaze_event
        global num
        if gaze_event: 
            if self.lcd_num is num:
                self.correct_num += 1 
            gaze_event = False
            self.gaze_data.append((int(self.lcd_num), int(num)))
            num = self.lcd_num
            self.lcd_num = str((int(self.lcd_num) + 1) % 8)
            self.lcd.display(self.lcd_num)
            self.total += 1
            
            


    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            gaze_np = np.array(self.gaze_data)
            print(gaze_np)
            np.save('../output/gaze_data.npy', gaze_np)
            self.close()

        # print(e.key())
        # print(x, y)

        # if e.key() == 85:
        #     y -= 100
        #     # print('up')
        # elif e.key() == 68:
        #     y += 100
        #     # print('down')
        # elif e.key() == 76:
        #     x -= 100
        #     # print('left')
        # elif e.key() == 82:
        #     x += 100
        #     # print('right')
        # QCursor.setPos(x, y)

    def setQCursorPos(self):
        line = self.posF.readline().decode('utf-8')
        xy = line.strip()
        xy = xy.split(" ")
        # print(len(xy))
        if len(xy) == 2:
            # print(xy)
            self.x = self.move_avg_rate * self.x + (1-self.move_avg_rate) * int(xy[0])
            self.y = self.move_avg_rate * self.y + (1-self.move_avg_rate) * int(xy[1])
            QCursor.setPos(self.x, self.y)
        # QCursor.setPos(x, y)


if __name__ == '__main__':

    proc = Popen("./EfficientGaze", bufsize=1024, stdin=PIPE, stdout=PIPE)  
    (fin, fout) = (proc.stdin, proc.stdout)
    time.sleep(4)
    app = QApplication(sys.argv)
    ex = Example(fout)
    app.exit(app.exec_())
