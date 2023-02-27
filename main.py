import sys
import cv2
import numpy as np
# from pandas import writer
from loadModel import Classifier
from time import time
from csv import *

# from PIL import Image
import align.detect_face
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
from matplotlib.figure import Figure

############################
import sys


matplotlib.use('Qt5Agg')
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal,Qt,QThread
from Display import Ui_MainWindow
from PyQt5.QtWidgets import QWidget, QApplication

###############################




plt.style.use('fivethirtyeight')

INPUT_IMAGE_SIZE = 96
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709

tf.compat.v1.disable_eager_execution()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
with sess.as_default():
    with sess.graph.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")

classifier = Classifier()
classifier.load_model()
_translate = QtCore.QCoreApplication.translate
class MainWindown(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = QWidget()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.ButtonStart.clicked.connect(self.start_capture_video)
        self.uic.ButtonStop.clicked.connect(self.stop_capture_video)
        self.thread = {}

    def closeEnvent(self, event):
        self.stop_capture_video()

    def start_capture_video(self):
        self.thread[1] = live_stream(index=1)

        self.thread[1].start()
        # self.time=time()
        self.thread[1].signal1.connect(self.show_wedcam)
        self.thread[1].signal2.connect(self.show_gh)
        self.thread[1].signal3.connect(self.show_Flas)

    def stop_capture_video(self):
        self.thread[1].stop()

    def show_wedcam(self, frame):
        qt_img = self.convert_cv_qt(frame)
        qt_img_graph = self.convert_cv_qt_gh(frame)
        self.uic.label.setPixmap(qt_img)
        self.uic.label_2.setPixmap(qt_img_graph)
    def show_Flas(self, flag):
        if flag == 'Bình thường':
            self.uic.label_4.setText(_translate("MainWindow", "Bình thường"))
        elif flag == 'Căng thẳng':
            self.uic.label_4.setText(_translate("MainWindow", "Căng thẳng"))
        elif flag == 'Có khả năng căng thẳng':
            self.uic.label_4.setText(_translate("MainWindow", "Có khả năng căng thẳng"))
    def show_gh(self, frame):
        qt_img_graph = self.convert_cv_qt_gh(frame)
        self.uic.label_2.setPixmap(qt_img_graph)

    def convert_cv_qt(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    def convert_cv_qt_gh(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(300, 300, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class live_stream(QThread):
    signal1 = pyqtSignal(np.ndarray)
    signal2 = pyqtSignal(np.ndarray)
    signal3 = pyqtSignal(str)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(live_stream, self).__init__()
        self.countNegative = 0
        self.countPositive = 0
        self.time = 0
        self.flag = None
        self.img_graph = None
        self.timeCount = 0

    def run(self):

        self.time = time()
        self.run_programer()

    def detect_face(self, frame):
        # frame = cv2.flip(frame, 1)
        det = None
        bb = None
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        faces_found = bounding_boxes.shape[0]
        try:
            # if faces_found > 1:
            #     cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #                 1, (255, 255, 255), thickness=1, lineType=2)
            # elif faces_found > 0:
            #     det = bounding_boxes[:, 0:4]
            #     bb = np.zeros((faces_found, 4), dtype=np.int32)
            if faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)
        except:
            pass
        return bb, det, faces_found

    def get_camera_stream(self):
        return cv2.VideoCapture(0)
    # def animationGraph(self, ):

    def run_programer(self):
        vd = self.get_camera_stream()
        total_fps = 0
        frame_count = 0
        arrNegative = []
        arrPositive = []
        timeCount = []
        vlue = True
        while True:
            start_time = time()
            ret, frame = vd.read()
            try:
                bb, det, faces_found = self.detect_face(frame)

                for i in range(faces_found):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]
                    # print(bb[i][3]-bb[i][1])
                    # print(frame.shape[0])
                    # print((bb[i][3]-bb[i][1])/frame.shape[0])
                    if ((bb[i][3] - bb[i][1]) / frame.shape[0]) > 0.25:
                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                            interpolation=cv2.INTER_CUBIC)
                        # frame_count += 0.1
                        if vlue:
                            self.timeCount = time()
                            vlue = False
                        timeNow = time() - self.timeCount
                        timeCount.append(timeNow)
                        # 2m * 60s = 120s     =>    20% Negative   80% Positive

                        # convert scaled to image
                        scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
                        name = classifier.predict(scaled)
                        # print(name)
                        # self.signal.emit(frame)
                        color = (255, 255, 255)
                        if name == "Angry" or name == "Disgust" or name == "Fear" or name == "Sad":
                            color = (0, 0, 255)
                            self.countNegative += 0.1
                            arrNegative.append(self.countNegative)
                            arrPositive.append(self.countPositive)
                            List = [f'{timeNow}', f'{self.countPositive}', f'{self.countNegative}']
                            with open(r'.\Data\dataEmotion.csv', 'a') as f_object:
                                writer_object = writer(f_object)
                                writer_object.writerow(List)
                                f_object.close()
                        elif name == "Happy" or name == "Surprise":
                            self.countPositive += 0.1
                            arrPositive.append(self.countPositive)
                            arrNegative.append(self.countNegative)
                            color = (255, 0, 0)
                            List = [f'{timeNow}', f'{self.countPositive}', f'{self.countNegative}']
                            with open(r'.\Data\dataEmotion.csv', 'a') as f_object:
                                writer_object = writer(f_object)
                                writer_object.writerow(List)
                                f_object.close()
                        else:
                            arrPositive.append(self.countPositive)
                            arrNegative.append(self.countNegative)
                            color = (0, 255, 0)
                        if self.countNegative > self.countPositive:
                            self.flag = False
                        else:
                            self.flag = True
                        # put name
                        cv2.putText(frame, name, (bb[i][0], bb[i][1] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, color, thickness=1, lineType=2)
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), color, 2)
                        fig = Figure(figsize=(5, 4), dpi=100)
                        canvas = FigureCanvasAgg(fig)
                        ax = fig.add_subplot(111)
                        ax.plot(timeCount, arrPositive)
                        ax.plot(timeCount, arrNegative)
                        canvas.draw()
                        buf = canvas.buffer_rgba()
                        # img = Image.frombuffer('RGBA', canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
                        X = np.asarray(buf)
                        self.img_graph = cv2.cvtColor(X, cv2.COLOR_RGBA2BGR)


                end_time = time()
                timerun = end_time - self.time
                # print('time', timerun)
                if timerun > 10:
                    frame_count = self.countPositive + self.countNegative
                    if self.countPositive > ((frame_count/100)*20):
                        self.flag = 'Bình thường'
                        # print('Bình thường')
                    elif self.countNegative == frame_count:
                        self.flag = 'Căng thẳng'
                        # print('Cang thang')
                    else:
                        self.flag = 'Có khả năng căng thẳng'
                    #     print('Có khả năng Cang thanh')
                    # print("countPositive", self.countPositive)
                    # print("countNegative", self.countNegative)
                    # print("frame_count", frame_count)
                # if timerun > 300:
                    # self.countNegative = 0
                    # self.countPositive = 0

                fps = 1 / (end_time - start_time)
                total_fps += fps
                # print(f"Frame Per Second: {round(fps, 1)}FPS")
                self.signal1.emit(frame)
                self.signal2.emit(self.img_graph)
                self.signal3.emit(self.flag)
            except:
                pass

    def stop(self):
        # print("stop threading", self.index)
        directory = "E:\AppEmotionInterface\Image"
        os.chdir(directory)
        cv2.imwrite("test.jpg", self.img_graph)
        self.terminate()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindown()
    main_win.show()
    sys.exit(app.exec())


