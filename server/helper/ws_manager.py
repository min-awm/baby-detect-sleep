from fastapi import WebSocket

websocket_connection: WebSocket = None

async def send_image_client(data):
    global websocket_connection
    if websocket_connection:
        try:
            await websocket_connection.send_bytes(data)
        except RuntimeError:
            # Kết nối có thể đã đóng
            websocket_connection = None
            return {"error": "WebSocket closed"}

async def send_text_client(data):
    global websocket_connection
    if websocket_connection:
        try:
            await websocket_connection.send_text(data)
        except RuntimeError:
            # Kết nối có thể đã đóng
            websocket_connection = None
            return {"error": "WebSocket closed"}