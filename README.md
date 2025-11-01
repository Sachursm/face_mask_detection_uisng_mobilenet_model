# Face Mask Detector 😷

A real-time face mask detection system using **MobileNetV2** deep learning model and **OpenCV**. This project can detect whether a person is wearing a mask or not through webcam or static images.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

---

## 🎯 Features

- **Real-time Detection**: Detects face masks through webcam feed
- **Image Testing**: Test the model on static images
- **High Accuracy**: Uses MobileNetV2 architecture for efficient and accurate predictions
- **Face Detection**: Integrates Haar Cascade classifier for face detection
- **Easy to Use**: Simple scripts for training, testing, and deployment

---

## 📁 Project Structure

```
face-mask-detector/
│
├── model.py                    # MobileNetV2 model architecture and training
├── mask_webcam.py              # Real-time mask detection via webcam
├── test.py                     # Test model on static images
├── splitdata.py                # Split dataset into train/test/val sets
├── requirements.txt            # Required Python packages
├── .gitignore                  # Git ignore file
├── mask_detector_model.keras   # Trained model (download separately)
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Webcam (for real-time detection)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/face-mask-detector.git
   cd face-mask-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   
   Download the face mask dataset from Google Drive:
   
   🔗 [**Download Dataset (ZIP)**](https://drive.google.com/file/d/1sbMQknJ59usjmcA3Olt1-y3pgVce_8Tt/view?usp=sharing)
   
   - Extract the ZIP file
   - Place the extracted folders in the project directory
   - The dataset should contain `with_mask` and `without_mask` folders

4. **Download the trained model** (Optional - if not training from scratch)
   
   If you want to skip training, download the pre-trained model:
   - Place `mask_detector_model.keras` in the project root directory

---

## 📊 Dataset

The dataset contains images of people:
- **With masks** 😷
- **Without masks** 😊

After downloading, organize your data as:
```
dataset/
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── without_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

---

## 💻 Usage

### 1. Prepare the Dataset

Split your dataset into training, validation, and test sets:

```bash
python splitdata.py
```

This will create `train/`, `val/`, and `test/` directories with proper data distribution.

### 2. Train the Model

Train the MobileNetV2 model on your dataset:

```bash
python model.py
```

This will:
- Load and preprocess the dataset
- Train the MobileNetV2 model
- Save the trained model as `mask_detector_model.keras`

### 3. Test on Images

Test the model on a static image:

```bash
python test.py --image path/to/your/image.jpg
```

### 4. Real-time Webcam Detection

Run the webcam detection:

```bash
python mask_webcam.py
```

**How it works:**
1. Opens your webcam feed
2. Detects faces using Haar Cascade classifier
3. Crops each detected face
4. Predicts if the person is wearing a mask
5. Displays results with bounding boxes:
   - 🟢 **Green box**: Wearing mask
   - 🔴 **Red box**: Not wearing mask

**Controls:**
- Press `q` to quit

---

## 🧠 Model Architecture

The project uses **MobileNetV2** - a lightweight deep learning model perfect for:
- Real-time applications
- Mobile and embedded devices
- Resource-constrained environments

**Key Features:**
- Transfer learning from ImageNet weights
- Fine-tuned for binary classification (mask/no mask)
- Input size: 224×224×3
- Output: 2 classes (with_mask, without_mask)

---

## 🔧 Technical Details

### Face Detection
- **Algorithm**: Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)
- **Purpose**: Detects faces in the frame before classification
- **Advantages**: Fast, lightweight, works well for frontal faces

### Mask Classification
- **Model**: MobileNetV2
- **Framework**: TensorFlow/Keras
- **Preprocessing**: MobileNetV2 preprocessing function
- **Output**: Probability scores for each class

---

## 📦 Dependencies

```
tensorflow>=2.0.0
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📈 Results

The model achieves:
- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~98%
- **Real-time FPS**: 25-30 (depending on hardware)

---

## 🛠️ Troubleshooting

### Webcam not opening
```python
# Try changing camera index in mask_webcam.py
cap = cv2.VideoCapture(1)  # Try 0, 1, or 2
```

### Model not found error
- Ensure `mask_detector_model.keras` is in the project root
- Or retrain the model using `model.py`

### Low FPS
- Reduce webcam resolution in `mask_webcam.py`
- Use a GPU if available

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

Sachu Retna SM
- GitHub: [Sachursm](https://github.com/Sachursm)
- Email: [sachuretnasm@gmail.com](sachuretnasm@gmail.com)

---

## 🙏 Acknowledgments

- MobileNetV2 paper: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- OpenCV library for computer vision operations
- TensorFlow team for the deep learning framework
- Dataset contributors

---

## 📸 Screenshots

### Real-time Detection
![Demo](demo.gif)

### Results
| With Mask | Without Mask |
|-----------|--------------|
| ![](with_mask_example.jpg) | ![](without_mask_example.jpg) |

---

## 🔮 Future Improvements

- [ ] Add support for multiple face detection
- [ ] Implement distance detection for social distancing
- [ ] Deploy as web application
- [ ] Add mobile app support
- [ ] Improve accuracy with data augmentation
- [ ] Add mask type classification (surgical, N95, cloth)

---

## ⚠️ Disclaimer

This project is for educational purposes only. For production use in critical applications, please ensure proper testing and validation.

---

**If you found this project helpful, please give it a ⭐️!**
