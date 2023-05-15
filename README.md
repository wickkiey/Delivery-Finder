# Delivery-Finder
Detects from security camera whether the person is from which delivery partner

## Approach

### Data Preparation

- The dataset is prepared by crawling the google images for zomato delivery partner images and other images and manully curated the dataset.


### Data Preprocessing

- The images are resized to 320 x 320 and normalized to 0-1 range.


### Multi Model Approach

- A pre-trained lightweight model [SSD Mobilenet]('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2') is used to detect the person in the image.

- The detected person is cropped from the image and passed to the classification model. (TF Binary Classification Model)

- The classification model is trained on the dataset of 2 classes (Zomato Delivery Partner and Other Delivery Partner) with Training accuracy of 86% and Validation accuracy of 84%.


### Results

- The model is deployed on the Fastapi server and Streamlit cloud, the results are displayed on the web page.

Streamlit Server URL : https://wickkiey-delivery-finder-streamlit-server-db4d03.streamlit.app/

Fastapi Server URL : http://34.93.156.69:8000/





