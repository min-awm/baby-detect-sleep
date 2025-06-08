import datetime
import tempfile
import os
from typing import Union
from fastapi import FastAPI, Request, File, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from vision.detection import run_detection
from helper.path import get_abs_path
from helper import ws_manager  

app = FastAPI()

@app.post("/upload")
async def upload(request: Request):
    try:
        body = await request.body()  # Read raw body

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(body)
            temp.flush()  # Đảm bảo dữ liệu được ghi hết

            # Gọi hàm xử lý ảnh (nhận đường dẫn)
            results = run_detection(temp)

        return JSONResponse(content={"msg": "ok", "results": results}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/sound")
async def sound(request: Request):
    try:
        body = await request.body()  # Read raw body    

        return JSONResponse(content={"msg": "ok"}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket): 
    await websocket.accept()
    ws_manager.websocket_connection = websocket
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Data: {data}")
    except RuntimeError:
        print("Connection closed unexpectedly")
    finally:
        ws_manager.websocket_connection = None

# Mount static folder tại root "/"
app.mount(
    "/admin", 
    StaticFiles(directory=get_abs_path("static", __file__), html=True), 
    name="static"
)