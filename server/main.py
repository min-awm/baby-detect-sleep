from typing import Union
from fastapi import FastAPI, Request, File
from fastapi.staticfiles import StaticFiles
import datetime
import tempfile
from fastapi.responses import JSONResponse
from vision.detection import detect
from helper.path import get_abs_path
import os

app = FastAPI()

@app.post("/upload")
async def upload(request: Request):
    try:
        body = await request.body()  # Read raw body

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(body)
            temp.flush()  # Đảm bảo dữ liệu được ghi hết

            # Gọi hàm xử lý ảnh (nhận đường dẫn)
            detect(temp)

        return JSONResponse(content={"msg": "ok"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/sound")
async def sound(request: Request):
    try:
        body = await request.body()  # Read raw body    
        print(body)
        
        return JSONResponse(content={"msg": "ok"}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Mount static folder tại root "/"
# app.mount(
#     "/", 
#     StaticFiles(directory=get_abs_path("static", __file__), html=True), 
#     name="static"
# )