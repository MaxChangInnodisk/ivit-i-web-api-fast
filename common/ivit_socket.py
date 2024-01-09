from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from collections import defaultdict
import logging as logger

# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: dict = defaultdict()

#     async def connect(self, ws: WebSocket, uid: str):
#         await ws.accept()
#         self.active_connections[uid] = ws
#         logger.info(f'Submit WebSocket: {uid}')

#     def disconnect(self, uid: Optional[str] = None):
#         self.active_connections.pop(uid)

#     async def send(self, uid: str, message: str):
#         logger.debug(uid, message)
#         await self.active_connections[uid].send_json(message)

#     async def broadcast(self, message: str):
#         for connection in self.active_connections:
#             await connection.send_json(message)

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = defaultdict(list)

    async def connect(self, ws: WebSocket, uid: str):
        await ws.accept()
        self.active_connections[uid].append(ws)
        logger.info(f'Submit WebSocket: {uid}')

    def disconnect(self, ws: WebSocket, uid: str):
        self.active_connections[uid].remove(ws)

    async def send(self, uid: str, message: dict):
        for ws in self.active_connections[uid]:
            await ws.send_json(message)

    async def broadcast(self, message: dict):
        for uid, wss in self.active_connections.items():
            for ws in wss:
                await ws.send_json(message)

manager = ConnectionManager()
