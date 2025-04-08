import base64
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import cv2
from flask_cors import CORS
from flask_cors import cross_origin
from torchvision import transforms
import torch.nn.functional as F
import os

NUM_OF_CLASSES=62
EMNIST_BYCLASS_LABELS = {
    0: '0',  1: '1',  2: '2',  3: '3',  4: '4',  5: '5',  6: '6',  7: '7',  8: '8',  9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
}

class TAAF(nn.Module):
    def __init__(self, in_features):
        super(TAAF, self).__init__()
        # Initialize learnable parameters for scaling and shifting
        self.alpha = nn.Parameter(torch.ones(1))  # Scale parameter (learnable)
        self.beta = nn.Parameter(torch.zeros(1))  # Shift parameter (learnable)

    def forward(self, x):
        # TAAF function can be some form of scaled, shifted non-linearity
        # For example, a simple TAAF could be a scaled and shifted ReLU
        return self.alpha * F.relu(x + self.beta)

class CNNModel(nn.Module):
    def __init__(self, num_of_classes=NUM_OF_CLASSES):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=False)  # 1 input channel
        self.bn1 = nn.BatchNorm2d(16)

        self.dwconv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.dwconv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)  # 16 -> 32
        self.bn4 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.15)

        self.dwconv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False)
        self.bn5 = nn.BatchNorm2d(32)

        self.dwconv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False)
        self.bn6 = nn.BatchNorm2d(32)

        self.conv7 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)  # 32 -> 64
        self.bn7 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.2)

        self.dwconv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False)
        self.bn8 = nn.BatchNorm2d(64)

        self.dwconv9 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False)
        self.bn9 = nn.BatchNorm2d(64)

        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.2)

        self.dwconv11 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False)
        self.bn11 = nn.BatchNorm2d(64)

        self.dwconv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False)
        self.bn12 = nn.BatchNorm2d(64)

        self.conv13 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(0.2)

        self.dwconv14 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False)
        self.bn14 = nn.BatchNorm2d(64)

        self.dwconv15 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False)
        self.bn15 = nn.BatchNorm2d(64)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.drop5 = nn.Dropout(0.25)
        self.fc = nn.Linear(64, num_of_classes)

        self.taaf = TAAF(256)

    def forward(self, x):
        x = self.taaf(self.bn1(self.conv1(x)))
        x = self.taaf(self.bn2(self.dwconv2(x)))
        x = self.taaf(self.bn3(self.dwconv3(x)))

        x = self.taaf(self.bn4(self.conv4(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.taaf(self.bn5(self.dwconv5(x)))
        x = self.taaf(self.bn6(self.dwconv6(x)))

        x = self.taaf(self.bn7(self.conv7(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.taaf(self.bn8(self.dwconv8(x)))
        x = self.taaf(self.bn9(self.dwconv9(x)))

        x = self.taaf(self.bn10(self.conv10(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.taaf(self.bn11(self.dwconv11(x)))
        x = self.taaf(self.bn12(self.dwconv12(x)))

        x = self.taaf(self.bn13(self.conv13(x)))
        x = self.pool4(x)
        x = self.drop4(x)

        x = self.taaf(self.bn14(self.dwconv14(x)))
        x = self.taaf(self.bn15(self.dwconv15(x)))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.drop5(x)
        x = self.fc(x)
        return x

transform = transforms.ToTensor()

# model = ''
app = Flask(__name__, static_folder="dist", static_url_path="/")
# Explicitly configure CORS
# CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
CORS(app)

# # Load model inside function to prevent startup delays
# def load_model_on_demand():
#     return model

# Handle preflight OPTIONS request
@app.route('/predict', methods=['OPTIONS'])
def handle_options():
    response = jsonify({'message': 'CORS preflight OK'})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emnist.pth')
    print(model_path)
    model = load_model(model_path)
    try:
        data = request.get_json()
        image_data = data['image']

        # Convert base64 to image
        img_str = image_data.split(',')[1]
        img_data = base64.b64decode(img_str)
        img = Image.open(BytesIO(img_data)).convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img)

        _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        (h, w) = img_array.shape[:2]
        center = (w // 2, h // 2)  # Center of the image

        # Define rotation angle (e.g., 45 degrees)
        angle = 0

            # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 1.0 is the scale factor

        img_array = cv2.flip(img_array,1)
        # Apply rotation
        img_array = cv2.warpAffine(img_array, rotation_matrix, (w, h))

        # Normalize and reshape
        # img_array = img_array / 255.0
        # img_array = img_array.reshape(1, 28, 28, 1)
        img_array = transform(img_array)
        img_array = img_array.unsqueeze(0)

        # Load model inside function
        # model = load_model_on_demand()
        with torch.no_grad():
            prediction = model(img_array)
            #prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction))

        return jsonify({'prediction': EMNIST_BYCLASS_LABELS[predicted_class]})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

# Ensure CORS headers are always set
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response

# Serve the Vite frontend
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

def load_model(path):
    model = CNNModel()
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

if __name__ == '__main__':
    app.run(debug=True)
