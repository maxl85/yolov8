import json
import numpy as np

import cv2
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware


########################### YOLO ################################

# Initialize the models
yolo_model = YOLO('yolov8x.pt')

###################### FastAPI Setup #############################

# title
app = FastAPI(
    title="Object Detection FastAPI YOLO v8",
    description="""YOLO v8""",
    version="2024.01.04",
)

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def save_openapi_json():
    '''This function is used to save the OpenAPI documentation 
    data of the FastAPI application to a JSON file. 
    The purpose of saving the OpenAPI documentation data is to have 
    a permanent and offline record of the API specification, 
    which can be used for documentation purposes or 
    to generate client libraries. It is not necessarily needed, 
    but can be helpful in certain scenarios.'''
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


######################### MAIN Func #################################

@app.post("/object_detection", status_code=status.HTTP_201_CREATED)
def object_detection(file: UploadFile):
    contents = file.file.read()
    nparr = np.fromstring(contents, np.uint8)
    input_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    
    res = yolo_model.predict(source=input_img, save=True, project="predicted", name="img")
    
    