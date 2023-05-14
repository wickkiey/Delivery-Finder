from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import base64
import numpy as np
import uvicorn
import os
import json
from helper_funtions import get_cropped_images, predict_image_class
import tensorflow as tf
import tensorflow_hub as hub


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# initialize model
def load_model():
    model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'
    hub_model = hub.load(model_handle)
    
    clf_model = tf.keras.models.load_model('model/binary_22')
    return hub_model, clf_model

hub_model,clf_model = load_model()



def numpy_array_to_base64(array):
    """
    Convert a NumPy array to a base64-encoded string.
    """
    pil_image = Image.fromarray(array)
    buff = io.BytesIO()
    pil_image.save(buff, format='PNG')
    data = base64.b64encode(buff.getvalue())
    return data.decode('utf-8')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print("image updload")
    print(file.filename)
    image = await file.read()
    pil_image = Image.open(io.BytesIO(image))
    image_array = np.array(pil_image)
    
    cropped_images, output_image = get_cropped_images(image_array,hub_model)
    
    classification_result = []
    for cropped_image in cropped_images:
        classification_result.append(predict_image_class(cropped_image,clf_model,thresh=.215))
    

    base64_data = numpy_array_to_base64(output_image)
    
    text = ""
    brand = ""
    if len(cropped_image)> 0:
        text = "Human Detected !"
        brand = [f'{index}.{name}' for index, name in enumerate(classification_result)]
        brand = "\n".join(brand)
    else:
        text = "No Human Detected !"
    
    
    return {"image_path": "data:image/png;base64," + base64_data,"text":" ".join(classification_result),"text1":brand}

@app.get("/")
async def main():
    return FileResponse("template/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)