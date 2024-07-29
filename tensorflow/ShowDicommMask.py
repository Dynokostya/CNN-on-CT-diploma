import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QPoint
import os
import numpy as np
from PIL import Image


class ImageViewer(QMainWindow):
    def __init__(self, dicom_dir, mask_dir, min_size, max_size, num):
        super().__init__()
        self.dicom_dir = dicom_dir
        self.mask_dir = mask_dir
        self.min_size = min_size
        self.max_size = max_size
        self.num = num
        self.initUI()

    def initUI(self):
        # Set up the main window
        self.setWindowTitle('DICOM and Mask Viewer')
        self.setGeometry(100, 100, 800, 600)
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        self.imageLabel = QLabel(self)
        self.combinedLabel = QLabel(self)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.min_size)
        self.slider.setMaximum(self.max_size)
        self.slider.valueChanged.connect(self.update_images)
        hbox.addWidget(self.imageLabel)
        hbox.addWidget(self.combinedLabel)
        vbox.addLayout(hbox)
        vbox.addWidget(self.slider)

        container = QWidget()
        container.setLayout(vbox)
        self.setCentralWidget(container)

        self.update_images(self.slider.value())

    def find_max_slices(self):
        # Assume the number of DICOM files and mask files are the same
        return len([name for name in os.listdir(self.dicom_dir) if name.endswith('.png')])

    def update_images(self, slice_number):
        dicom_path = os.path.join(self.dicom_dir, f'I00{slice_number:03d}TIFF.tif')
        mask_path = os.path.join(self.mask_dir, f'1_00{self.num:02d}_mask_{slice_number:03d}.png')
        dicom_image = QImage(dicom_path)
        mask_image = QImage(mask_path)
        self.imageLabel.setPixmap(QPixmap.fromImage(dicom_image))
        combined_image = self.create_combined_image(dicom_image, mask_image, 0.75)  # 50% transparency for starting
        self.combinedLabel.setPixmap(combined_image)

    def create_combined_image(self, dicom_image, mask_image, alpha):
        dicom_image = dicom_image.convertToFormat(QImage.Format_ARGB32)
        mask_image = mask_image.convertToFormat(QImage.Format_ARGB32)
        result_image = QImage(dicom_image.size(), QImage.Format_ARGB32)
        painter = QPainter(result_image)
        painter.drawImage(QPoint(0, 0), dicom_image)
        painter.setOpacity(alpha)
        painter.drawImage(QPoint(0, 0), mask_image)
        painter.end()
        return QPixmap.fromImage(result_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    num = 15
    dicom_dir = f'PatientData/I_00{num:02d}/DICOMTIFF/'
    mask_dir = f'PatientData/I_00{num:02d}/Masks/'
    ex = ImageViewer(dicom_dir, mask_dir, 303, 316, num)
    ex.show()
    sys.exit(app.exec_())
