from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mask_detector_model.keras")

def predict_mask(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    classes = ['with_mask', 'without_mask']
    return classes[np.argmax(prediction)]

print(predict_mask("mask_dataset/test/without_mask/without_mask_90.jpg", model))