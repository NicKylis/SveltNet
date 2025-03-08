import sys
import numpy as np
import time
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QFont
from PyQt5.QtCore import Qt, QPoint
from cv2 import cv2 as cv
import tensorflow as tf

class DigitRecognizer(QWidget):
    def __init__(self):
        super().__init__()

        # Window settings
        self.setWindowTitle("Handwritten Digit Recognizer")
        self.setGeometry(100, 100, 500, 600)
        self.setMinimumSize(600, 400)

        # Canvas settings
        self.canvas_size = 280
        self.canvas = QPixmap(self.canvas_size, self.canvas_size)
        self.canvas.fill(Qt.white)

        # UI Elements
        self.label_title = QLabel("Draw Here", self)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setFont(QFont("Arial", 14, QFont.Bold))

        self.label_canvas = QLabel(self)
        self.label_canvas.setPixmap(self.canvas)
        self.label_canvas.setAlignment(Qt.AlignCenter)

        self.label_or = QLabel("Or", self)
        self.label_or.setAlignment(Qt.AlignCenter)
        self.label_or.setFont(QFont("Arial", 12))

        self.button_upload = QPushButton("Upload File", self)
        self.button_upload.clicked.connect(self.upload_image)
        self.button_upload.setFont(QFont("Arial", 12))

        self.button_clear = QPushButton("Clear", self)
        self.button_clear.clicked.connect(self.clear_canvas)
        self.button_clear.setFont(QFont("Arial", 12))

        self.button_predict = QPushButton("Predict", self)
        self.button_predict.clicked.connect(self.predict_digit)
        self.button_predict.setFont(QFont("Arial", 12))

        self.label_result = QLabel("Prediction: ?", self)
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setFont(QFont("Arial", 16, QFont.Bold))

        self.label_logs = QLabel("", self)  # Log display
        self.label_logs.setAlignment(Qt.AlignCenter)
        self.label_logs.setFont(QFont("Arial", 10, QFont.Bold))

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label_title)
        layout.addWidget(self.label_canvas, alignment=Qt.AlignCenter)
        layout.addWidget(self.label_or)
        layout.addWidget(self.button_upload, alignment=Qt.AlignCenter)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button_clear)
        button_layout.addWidget(self.button_predict)

        layout.addLayout(button_layout)
        layout.addWidget(self.label_result)
        layout.addWidget(self.label_logs)  # Add logs display
        self.setLayout(layout)

        # Variables for drawing
        self.drawing = False
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.label_canvas.geometry().contains(event.pos()):
            self.drawing = True
            self.last_point = self.map_to_canvas(event.pos())

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.canvas)
            pen = QPen(Qt.black, max(2, self.canvas_size // 40), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)

            new_point = self.map_to_canvas(event.pos())
            if new_point and self.last_point:
                painter.drawLine(self.last_point, new_point)
                self.last_point = new_point
                self.label_canvas.setPixmap(self.canvas)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def map_to_canvas(self, pos):
        """Map mouse position to the canvas."""
        label_rect = self.label_canvas.geometry()
        if label_rect.contains(pos):
            x = (pos.x() - label_rect.x()) * self.canvas_size / label_rect.width()
            y = (pos.y() - label_rect.y()) * self.canvas_size / label_rect.height()
            return QPoint(int(x), int(y))
        return None

    def clear_canvas(self):
        self.canvas.fill(Qt.white)
        self.label_canvas.setPixmap(self.canvas)
        self.label_result.setText("Prediction: ?")
        self.label_logs.setText("")

    def predict_digit(self):
        # Convert canvas to grayscale image
        image = self.canvas.toImage().convertToFormat(QImage.Format_Grayscale8)
        buffer = image.bits().asarray(self.canvas_size * self.canvas_size)
        img_array = np.array(buffer, dtype=np.uint8).reshape((self.canvas_size, self.canvas_size))

        # Resize image to 28x28
        img_resized = cv.resize(img_array, (28, 28), interpolation=cv.INTER_AREA)

        # Apply Otsu thresholding
        _, img_thresh = cv.threshold(img_resized, 127, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)

        # Normalize and reshape image
        img_array = img_thresh / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Measure inference time
        start_time = time.time()
        prediction = model.predict(img_array, verbose=0)  # Suppress TensorFlow logs
        end_time = time.time()

        # Extract predicted digit
        digit = np.argmax(prediction)
        inference_time = end_time - start_time  # Compute inference time

        # Update result label
        self.label_result.setText(f"Prediction: {digit}")

        # Display the model inference time logs
        self.label_logs.setText(f"1/1 ━━━━━━━━━━━━━━━━━━━━ {inference_time:.3f}s {int(inference_time * 1000)}ms/step")

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            # Load and process the image
            img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv.resize(img, (28, 28), interpolation=cv.INTER_AREA)
                _, img_thresh = cv.threshold(img_resized, 127, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)

                # Normalize and reshape for model input
                img_array = img_thresh / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)

                # Display the uploaded image in canvas
                img_qt = QImage(img_thresh.data, 28, 28, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(img_qt.scaled(self.canvas_size, self.canvas_size, Qt.KeepAspectRatio))
                self.canvas = pixmap
                self.label_canvas.setPixmap(self.canvas)

if __name__ == '__main__':
    # Load the trained model
    model = "model_mnist.keras"
    current_folder = os.path.dirname(__file__)
    model = tf.keras.models.load_model(os.path.join(current_folder, '../models/', model))
    app = QApplication(sys.argv)
    window = DigitRecognizer()
    window.show()
    sys.exit(app.exec_())
