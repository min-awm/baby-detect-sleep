from typing import Union
from fastapi import FastAPI, UploadFile, File
import datetime
import tempfile
from fastapi.responses import JSONResponse
from vision.detection import detect

app = FastAPI()


@app.get("/")
def read_root():
    a()
    return {"Hello": "wws"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=True, suffix=file.filename) as temp:
        temp.write(contents)
        detect(temp)
        
    # folder_image = "./image-upload"
    # filename = f"{folder_image}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    # with open(filename, "wb") as f:
    #     f.write(contents)
    # print(contents)

    return JSONResponse({"status": "success"})


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}