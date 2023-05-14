
import tensorflow as tf
import numpy as np
import os
import cv2
import time


def get_cropped_images(image,hub_model):
    
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 320))
        image = image[np.newaxis, ...]
        # image = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
        start_time = time.time()
        result = hub_model(image)
        end_time = time.time()
        result = {key:value.numpy()[0] for key,value in result.items()}
        
        # filter results where detection_classes == 1
        human_idx = np.where(result['detection_classes'] == 1)

        human_results = {}
        human_results['detection_boxes'] = result['detection_boxes'][human_idx]
        human_results['detection_scores'] = result['detection_scores'][human_idx]
        
        cropped_results = []
        output_image = image.copy()
        for i in range(len(human_results['detection_boxes'])):
            box = human_results['detection_boxes'][i]
            box = [int(x * 320) for x in box]
            conf_score = human_results['detection_scores'][i]
            if conf_score > 0.5:
                # write to output folder as image
                cropped_img = image[0][box[0]:box[2], box[1]:box[3]]
                if (cropped_img.shape[0] > 10) and (cropped_img.shape[1] > 10):
                    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                    cropped_results.append(cropped_img.copy())
                    
                    # draw bounding box
                    cv2.rectangle(output_image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                    
        return cropped_results, output_image
                 
                 
def predict_image_class(image, model,thresh = 0.22):
    # Load the saved model

    # Load and preprocess the image

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.image.resize(img_array, [32*2, 32])
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.

    # Make predictions
    predictions = model(img_array)
    predicted_thresh = predictions[0][0].numpy()
    print(predicted_thresh)
    if predicted_thresh < thresh:
        return 'others'
    else:
        return 'zomato'

