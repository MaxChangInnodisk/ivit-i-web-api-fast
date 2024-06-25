import logging as logger

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Submit WebSocket: {websocket}")

    def disconnect(self, websocket: WebSocket):
        if websocket not in self.active_connections:
            return
        self.active_connections.remove(websocket)

    async def send(self, ws: WebSocket, message: dict):
        await ws.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: dict = defaultdict(set)

#     async def connect(self, ws: WebSocket, uid: str):
#         await ws.accept()
#         uid = uid.upper()
#         self.register(ws, uid)

#     def register(self, ws: WebSocket, uid: str):
#         uid = uid.upper()
#         if ws in self.active_connections[uid]:
#             return
#         self.active_connections[uid].add(ws)
#         logger.info(f"Submit WebSocket: {uid}")

#     def disconnect(self, ws: WebSocket, uid: str):
#         uid = uid.upper()
#         self.active_connections[uid].remove(ws)

#     async def send(self, uid: str, message: dict):
#         uid = uid.upper()
#         for ws in self.active_connections[uid]:
#             await ws.send_json(message)

#     async def broadcast(self, message: dict):
#         idx = 0
#         need_pop = defaultdict(list)
#         for uid, wss in self.active_connections.items():
#             for ws in wss:
#                 try:
#                     await ws.send_json(message)
#                     idx += 1
#                 except BaseException:
#                     need_pop[uid].append(ws)
#         for uid, wss in need_pop.items():
#             for ws in wss:
#                 try:
#                     self.active_connections[uid].remove(ws)
#                     logger.debug(f"Pop out disconnected websocket: {ws}")
#                 except BaseException:
#                     logger.debug(f"Pop out disconnected websocket: {ws} failed !")

#         logger.debug(f"WebSocket Broadcast: {idx}")


manager = ConnectionManager()
