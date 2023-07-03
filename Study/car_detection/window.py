import sys
from PyQt5.QtWidgets import *
import torch
from torch import nn
import torchvision.models as models
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import serial
import time

#아두이노 통신 포트, 보드 레이트
py_serial = serial.Serial(

    # Window
    port='COM9',

    # 보드 레이트 (통신 속도)
    baudrate=9600,
)


class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.model = models.resnet18()
        self.model.fc = nn.Linear(512, 2)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.model(x)
        return x

PATH = './car.pth'
net = TheModelClass()
net.load_state_dict(torch.load(PATH))
net.eval()
class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CarDetection')
        self.move(300, 300)
        self.resize(400, 200)
        self.pushButton = QPushButton('Upload File')
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.pushButton)
        layout.addWidget(self.label)

        self.setLayout(layout)

    def pushButtonClicked(self):
        fname = QFileDialog.getOpenFileName(self)

        img_RGB = Image.open(fname[0]).convert('L')
        img_RGB = img_RGB.resize((224, 224))
        # torchvision.transforms.ToTensor
        tf_toTensor = ToTensor()
        # PIL to Tensor
        image = tf_toTensor(img_RGB).unsqueeze(dim=0)

        outputs = net(image)
        _, predicted = torch.max(outputs, 1)

        if predicted.item() ==0:
            text = "차량을 감지하지 못하였습니다."
            self.label.setText(text)

        else:
            text = "차량을 감지하였습니다."
            state = 'T'
            self.label.setText(text)
            py_serial.write(state.encode())

            if py_serial.readable():
                response = py_serial.readline()

                self.label.setText(response.decode())


if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   ex.show()
   sys.exit(app.exec_())