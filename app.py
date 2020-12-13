
#Import necessary libraries
from flask import Flask, render_template, request,url_for
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os

# Create flask instance
app = Flask(__name__)

fashion_items=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


# Function to load and prepare the image in right shape
def read_image(filename):
    # Load the image
    img = load_img(filename,color_mode="grayscale", target_size=(28, 28))
    # Convert the image to array
    img = img_to_array(img)
    
    print(img.shape)
    # Reshape the image into a sample of 1 channel
    img = img.reshape(1, 28, 28, 1)
    # Prepare it as pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('static/images', filename)
        file.save(file_path)
        img = read_image(file_path)
        # Predict the class of an image
        model1 = load_model('fashion.h5')
        class_prediction = model1.predict(img)
        fashion_object=np.argmax(class_prediction)
        print(class_prediction)
        #Map apparel category with the numerical class
        product=fashion_items[fashion_object]
        return render_template('predict.html', product = product, user_image = file_path)


if __name__ == "__main__":
    app.run()
