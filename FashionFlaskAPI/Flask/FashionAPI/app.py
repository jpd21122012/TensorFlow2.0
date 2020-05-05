#Step 1: Import all dependencies to project
import os
import requests
import numpy as np
import tensorflow as tf

#Downgrade to version 1.1.0 with pip install scipy==1.1.0 or conda install
#scipy==1.1.0
from scipy.misc import imread, imsave
from flask import Flask, request, jsonify

print(tf.__version__)

#Step 2: Load the pretrained model
with open('fashion_model_flask.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

# Load model Weights
model.load_weights("fashion_model_flask.h5")

#Step 3: Create Flask API
#Create Flask application
app = Flask(__name__)

#Define function to classify images
@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    #Define the image folder
    upload_dir = "uploads/"
    #Load one of the folder images
    image = imread(upload_dir + img_name)
    
    #Define the possible classes of prediction
    classes = ["T-shirt", "Trousers", "Sweatshirt", "Dress", "Coat",
               "Sandal", "Jersey", "Slipper", "Bag", "Boots"]

    #Make prediction using the pretrained model
    prediction = model.predict([image.reshape(1, 28 * 28)])

    #Return to user the prediction
    return jsonify({"object_identified":classes[np.argmax(prediction[0])]})

#Init Flask application
app.run(port=5000, debug=False)
