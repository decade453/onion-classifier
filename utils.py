import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("model/onion_model.h5")
classes = ['red', 'spoiled', 'white', 'yellow']

def predict(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)]
