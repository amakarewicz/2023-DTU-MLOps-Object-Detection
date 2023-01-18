from http import HTTPStatus
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from typing import Optional
import cv2



app = FastAPI()

# curl -X DELETE -G \
# 'http://localhost:5000/locations' \
# -d id=3 \
# -d name=Mario \
# -d surname=Bros




@app.get("/")
def read_root():
   return {"Hello": "World"}

# @app.get("/")
# def root():
#     """ Health check."""
#     response = {
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#     }
#     return response

@app.get("/items/{item_id}")
def read_item(item_id: int):
   return {"item_id": item_id}


from enum import Enum
class ItemEnum(Enum):
   alexnet = "alexnet"
   resnet = "resnet"
   lenet = "lenet"

@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
   return {"item_id": item_id}


@app.get("/query_items")
def read_item(item_id: int):
   return {"item_id": item_id}

database = {'username': [ ], 'password': [ ]}

@app.post("/login/")
def login(username: str, password: str):
   #curl -X POST "http://localhost:8000/login/?username=test1&password=pwtest1" -H "accept: application/json" -d ""

    
   username_db = database['username']
   password_db = database['password']
   if username not in username_db and password not in password_db:
      with open('database.csv', "a") as file:
            file.write(f"{username}, {password} \n")
      username_db.append(username)
      password_db.append(password)
   return "login saved"



@app.get("/db/")
def read_item():
   return database


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...)):
   with open('image.jpg', 'wb') as image:
      content = await data.read()
      image.write(content)
      image.close()

   response = {
      "input": data,
      "message": HTTPStatus.OK.phrase,
      "status-code": HTTPStatus.OK,
   }
   return response


img = cv2.imread("image.jpg")
res = cv2.resize(img, (300, 300))
# show the image, provide window name first
cv2.imshow('image window', img)
# add wait key. window waits until user presses a key
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()


cv2.imwrite('image_resize.jpg', res)
FileResponse('image_resize.jpg')