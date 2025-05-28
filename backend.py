from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import logging
import json
from fastrtc import Stream, AsyncStreamHandler
import websockets 
from typing import Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment-configurable HF Space URL
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "androidguy-speaker-diarization.hf.space")
API_WS = f"wss://{HF_SPACE_URL}/ws_inference"

app = FastAPI()

# Add CORS middleware to allow cross-origin requests from HF Space
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RelayHandler(AsyncStreamHandler):
    """Handler to relay audio data between client and HF Space WebSocket"""
    
    def __init__(self):
        super().__init__()
        self.ws_connection = None
        self.clients: Set[WebSocket] = set()
        self.connection_lock = asyncio.Lock()
        self.reconnect_task = None
        
    def copy(self): 
        # Return the same instance for shared state
        return self
    
    async def start_up(self):
        """Connect to HF Space WebSocket when starting"""
        async with self.connection_lock:
            if self.ws_connection:
                return
                
            logger.info(f"Connecting to WebSocket at {API_WS}")
            try:
                self.ws_connection = await websockets.connect(API_WS)
                logger.info("WebSocket connection established")
                
                # Start background task to receive messages
                if self.reconnect_task:
                    self.reconnect_task.cancel()
                self.reconnect_task = asyncio.create_task(self.receive_from_hf())
                
            except Exception as e:
                logger.error(f"Failed to connect to HF Space WebSocket: {e}")
                self.ws_connection = None
                # Schedule reconnection
                self.reconnect_task = asyncio.create_task(self.auto_reconnect())
    
    async def shutdown(self):
        """Close WebSocket connection when shutting down"""
        async with self.connection_lock:
            if self.reconnect_task:
                self.reconnect_task.cancel()
                
            if self.ws_connection:
                await self.ws_connection.close()
                self.ws_connection = None
                logger.info("WebSocket connection closed")
    
    async def auto_reconnect(self):
        """Automatically reconnect to HF Space WebSocket"""
        while True:
            try:
                await asyncio.sleep(5)  # Wait before reconnecting
                if not self.ws_connection:
                    await self.start_up()
                    break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
    
    async def receive(self, frame):
        """Receive audio data from client and forward to HF Space"""
        try:
            data = frame.data if hasattr(frame, 'data') else frame
            
            if not self.ws_connection:
                await self.start_up()
                
            if self.ws_connection:
                await self.ws_connection.send(data)
            else:
                logger.warning("No WebSocket connection to HF Space")
                
        except Exception as e:
            logger.error(f"Error in receive: {e}")
            # Trigger reconnection
            self.ws_connection = None
            asyncio.create_task(self.start_up())
    
    async def receive_from_hf(self):
        """Background task to receive messages from HF Space and broadcast to clients"""
        while self.ws_connection:
            try:
                message = await self.ws_connection.recv()
                # Broadcast to all connected clients
                await self.broadcast_to_clients(message)
                
            except websockets.exceptions.ConnectionClosed:
                logger.info("HF Space WebSocket connection closed")
                self.ws_connection = None
                break
            except Exception as e:
                logger.error(f"Error receiving from HF Space: {e}")
                self.ws_connection = None
                break
        
        # Try to reconnect
        asyncio.create_task(self.auto_reconnect())
    
    async def broadcast_to_clients(self, message):
        """Broadcast message to all connected WebSocket clients"""
        if not self.clients:
            return
            
        # Remove disconnected clients
        disconnected = set()
        
        for client in self.clients.copy():
            try:
                await client.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
    
    def add_client(self, websocket: WebSocket):
        """Add a client WebSocket connection"""
        self.clients.add(websocket)
        logger.info(f"Client added. Total clients: {len(self.clients)}")
    
    def remove_client(self, websocket: WebSocket):
        """Remove a client WebSocket connection"""
        self.clients.discard(websocket)
        logger.info(f"Client removed. Total clients: {len(self.clients)}")
    
    async def emit(self):
        """This method is called by FastRTC - we don't need to return anything here"""
        return None

# Set up FastRTC stream and mount it on the FastAPI app
tool_handler = RelayHandler()
stream = Stream(handler=tool_handler, modality="audio", mode="send-receive")
stream.mount(app)

@app.on_event("startup")
async def startup_event():
    """Initialize WebSocket connection on startup"""
    await tool_handler.start_up()

@app.on_event("shutdown") 
async def shutdown_event():
    """Clean up on shutdown"""
    await tool_handler.shutdown()

@app.get("/")
@app.head("/")
async def root():
    return {"message": "Speaker Diarization Signaling Server"}

@app.get("/health")
@app.head("/health")
async def health(): 
    return {
        "status": "ok", 
        "connected_to_hf": tool_handler.ws_connection is not None,
        "connected_clients": len(tool_handler.clients)
    }

@app.websocket("/ws_relay")
async def websocket_relay(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket relay connection established")
    
    # Add this client to the handler
    tool_handler.add_client(websocket)
    
    try:
        # Keep the connection alive and handle any incoming messages
        while True:
            try:
                # Wait for messages from client (if any)
                message = await websocket.receive_text()
                logger.info(f"Received message from client: {message}")
            
                
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket relay: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket relay error: {e}")
    finally:
        tool_handler.remove_client(websocket)
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))