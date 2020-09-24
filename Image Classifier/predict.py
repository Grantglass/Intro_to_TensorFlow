import argparse
import json
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

dsize=(224, 224)
num_classes=102

def parse_args():
    parser = argparse.ArgumentParser(description="Udacity Flower Image Classifier")
    parser.add_argument('img_path')
    parser.add_argument('model_path')
    parser.add_argument('--top_k', default=5)
    parser.add_argument('--category_names', default='label_map.json')
    return parser.parse_args()

def process_image(image):
    global dsize
    image=tf.convert_to_tensor(image,tf.float32)
    image=tf.image.resize(image, dsize)
    image/=255
    return image

def predict(image_path=None, model=None, top_k=None):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)    
    processed_test_image=np.expand_dims(processed_test_image,0)
    probs=model.predict(processed_test_image)
    return tf.nn.top_k(probs, k=top_k)

def load_json(path):
    with open(path, 'r') as f:
        class_names = json.load(f)
    return class_names

def filtered(classes=None,class_names=None):
    return [class_names.get(str(key+1)) if key else "Placeholder" for key in classes.numpy().squeeze().tolist()]

def run():
    args = parse_args()
    class_names=load_json(args.category_names)

    classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4"
    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_url, input_shape=(224,224,3)),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dense(num_classes,activation='softmax'),
    ])    

  
    classifier.compile(optimizer='adadelta',\
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),\
                   metrics=['accuracy',
                           ],)

    classifier.load_weights(args.model_path)
    probs, classes = predict(image_path=args.img_path, model=classifier, top_k=args.top_k)
    pred_dict={filtered(classes,class_names)[i]: probs[0][i].numpy() for i in range(len(filtered(classes,class_names)))} 
    print(f"Predictions: {pred_dict}")
    return probs, classes,filtered(classes,class_names),pred_dict    
if __name__ == '__main__':
    run()