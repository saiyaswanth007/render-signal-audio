from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import logging
from fastrtc import Stream, AsyncStreamHandler

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
        self.client_queue = asyncio.Queue()
        
    def copy(self): 
        return RelayHandler()
    
    async def start_up(self):
        """Connect to HF Space WebSocket when starting"""
        logger.info(f"Connecting to WebSocket at {API_WS}")
        try:
            self.ws_connection = await WebSocket.connect(API_WS)
            logger.info("WebSocket connection established")
            asyncio.create_task(self.receive_from_hf())
        except Exception as e:
            logger.error(f"Failed to connect to HF Space WebSocket: {e}")
            self.ws_connection = None
    
    async def shutdown(self):
        """Close WebSocket connection when shutting down"""
        if self.ws_connection:
            await self.ws_connection.close()
            logger.info("WebSocket connection closed")
    
    async def receive(self, frame):
        """Receive audio data from client and forward to HF Space"""
        try:
            data = frame.data if hasattr(frame, 'data') else frame
            if self.ws_connection:
                await self.ws_connection.send_bytes(data)
            else:
                logger.warning("No WebSocket connection to HF Space")
                await self.start_up()
        except Exception as e:
            logger.error(f"Error in receive: {e}")
    
    async def receive_from_hf(self):
        """Background task to receive messages from HF Space and queue them"""
        while True:
            try:
                if not self.ws_connection:
                    await asyncio.sleep(1)
                    continue
                message = await self.ws_connection.receive_text()
                await self.client_queue.put(message)
            except Exception as e:
                logger.error(f"Error receiving from HF Space: {e}")
                await asyncio.sleep(2)
                await self.start_up()
    
    async def emit(self):
        """Send queued messages from HF Space to client"""
        try:
            if not self.client_queue.empty():
                return self.client_queue.get_nowait()
        except Exception as e:
            logger.error(f"Error in emit: {e}")
        return None

# Set up FastRTC stream and mount it on the FastAPI app
tool_handler = RelayHandler()
stream = Stream(handler=tool_handler, modality="audio", mode="send-receive")
stream.mount(app)

@app.get("/")
@app.head("/")
async def root():
    return {"message": "Speaker Diarization Signaling Server"}

@app.get("/health")
@app.head("/health")
async def health(): 
    return {"status": "ok", "connected_to_hf": tool_handler.ws_connection is not None}

@app.websocket("/ws_relay")
async def websocket_relay(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket relay connection established")
    
    # Create a local queue for this connection
    local_queue = asyncio.Queue()
    
    # Add this client's queue to a list of clients to notify
    try:
        # Forward messages from HF Space to this client
        while True:
            try:
                # Get the next message from the global handler's queue
                message = await tool_handler.client_queue.get()
                
                # Send it to the client
                await websocket.send_text(message)
                
                # Mark task as done
                tool_handler.client_queue.task_done()
                
                # Put message back for other clients
                await tool_handler.client_queue.put(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error forwarding message: {e}")
                break
    except Exception as e:
        logger.error(f"WebSocket relay error: {e}")
    finally:
        logger.info("WebSocket relay connection closed")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))