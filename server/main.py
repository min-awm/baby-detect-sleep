from typing import Union
from fastapi import FastAPI, UploadFile, File
import datetime
from fastapi.responses import JSONResponse
from vision.detection import a

app = FastAPI()


@app.get("/")
def read_root():
    a()
    return {"Hello": "Worcld"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    # folder_image = "./image-upload"
    # filename = f"{folder_image}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    # with open(filename, "wb") as f:
    #     f.write(contents)
    print(contents)

    return JSONResponse({"status": "success"})


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}