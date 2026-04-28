import asyncio
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.event_log import EventLog

router = APIRouter(tags=["websocket"])


class LogStreamManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, run_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[run_id].add(websocket)

    def disconnect(self, run_id: str, websocket: WebSocket) -> None:
        sockets = self._connections.get(run_id)
        if not sockets:
            return
        sockets.discard(websocket)
        if not sockets:
            self._connections.pop(run_id, None)

    async def broadcast_event(self, run_id: str, event: EventLog) -> None:
        sockets = list(self._connections.get(run_id, set()))
        if not sockets:
            return
        payload = event.model_dump(mode="json")
        for socket in sockets:
            try:
                await socket.send_json(payload)
            except Exception:
                self.disconnect(run_id, socket)

    async def broadcast_message(self, run_id: str, payload: dict) -> None:
        sockets = list(self._connections.get(run_id, set()))
        if not sockets:
            return
        for socket in sockets:
            try:
                await socket.send_json(payload)
            except Exception:
                self.disconnect(run_id, socket)

    def broadcast_from_thread(self, loop: asyncio.AbstractEventLoop, run_id: str, event: EventLog) -> None:
        loop.call_soon_threadsafe(asyncio.create_task, self.broadcast_event(run_id, event))


log_stream_manager = LogStreamManager()


@router.websocket("/api/ws/runs/{run_id}")
async def run_log_stream(websocket: WebSocket, run_id: str) -> None:
    await log_stream_manager.connect(run_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        log_stream_manager.disconnect(run_id, websocket)
