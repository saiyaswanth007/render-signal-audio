from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import logging
import json
from fastrtc import Stream, AsyncStreamHandler
import websockets 
from typing import Set
import base64

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

class AudioTranscriptionRelay(AsyncStreamHandler):
    """Handler to relay real-time audio for transcription and speaker diarization"""
    
    def __init__(self):
        super().__init__()
        self.hf_ws_connection = None
        self.transcript_clients: Set[WebSocket] = set()
        self.connection_lock = asyncio.Lock()
        self.is_connected = False
        self.reconnect_task = None
        
    def copy(self): 
        # Return the same instance to maintain shared state across audio frames
        return self
    
    async def start_up(self):
        """Connect to HF Space WebSocket for audio processing"""
        async with self.connection_lock:
            if self.hf_ws_connection and self.is_connected:
                return
                
            logger.info(f"Connecting to HF Space WebSocket at {API_WS}")
            try:
                self.hf_ws_connection = await websockets.connect(
                    API_WS,
                    ping_interval=30,
                    ping_timeout=10,
                    max_size=None  # Allow large audio frames
                )
                self.is_connected = True
                logger.info("HF Space WebSocket connection established")
                
                # Start background task to receive transcription results
                if self.reconnect_task:
                    self.reconnect_task.cancel()
                self.reconnect_task = asyncio.create_task(self.receive_transcription_results())
                
            except Exception as e:
                logger.error(f"Failed to connect to HF Space WebSocket: {e}")
                self.hf_ws_connection = None
                self.is_connected = False
                # Schedule reconnection
                asyncio.create_task(self.auto_reconnect())
    
    async def shutdown(self):
        """Close WebSocket connection when shutting down"""
        async with self.connection_lock:
            self.is_connected = False
            
            if self.reconnect_task:
                self.reconnect_task.cancel()
                
            if self.hf_ws_connection:
                await self.hf_ws_connection.close()
                self.hf_ws_connection = None
                logger.info("HF Space WebSocket connection closed")
    
    async def auto_reconnect(self):
        """Automatically reconnect to HF Space WebSocket"""
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries and not self.is_connected:
            try:
                await asyncio.sleep(min(2 ** retry_count, 30))  # Exponential backoff
                await self.start_up()
                if self.is_connected:
                    logger.info("Successfully reconnected to HF Space")
                    break
                retry_count += 1
            except Exception as e:
                logger.error(f"Reconnection attempt {retry_count + 1} failed: {e}")
                retry_count += 1
        
        if retry_count >= max_retries:
            logger.error("Max reconnection attempts reached")
    
    async def receive(self, frame):
        """Receive real-time audio data from client and forward to HF Space"""
        try:
            # Handle different audio frame formats
            if hasattr(frame, 'data'):
                audio_data = frame.data
            else:
                audio_data = frame
            
            # Ensure connection to HF Space
            if not self.is_connected:
                await self.start_up()
                
            if self.hf_ws_connection and self.is_connected:
                # Send raw audio data to HF Space for processing
                await self.hf_ws_connection.send(audio_data)
                logger.debug(f"Sent {len(audio_data)} bytes of audio to HF Space")
            else:
                logger.warning("No active connection to HF Space for audio streaming")
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("HF Space connection closed during audio send")
            self.is_connected = False
            asyncio.create_task(self.auto_reconnect())
        except Exception as e:
            logger.error(f"Error sending audio to HF Space: {e}")
            self.is_connected = False
            asyncio.create_task(self.auto_reconnect())
    
    async def receive_transcription_results(self):
        """Background task to receive transcription/diarization results from HF Space"""
        while self.hf_ws_connection and self.is_connected:
            try:
                # Receive transcription result from HF Space
                result = await self.hf_ws_connection.recv()
                
                # Parse the result (assuming JSON format)
                try:
                    if isinstance(result, str):
                        transcript_data = json.loads(result)
                    else:
                        # Handle binary data if needed
                        transcript_data = {"raw_data": base64.b64encode(result).decode()}
                    
                    logger.info(f"Received transcription: {transcript_data}")
                    
                    # Broadcast transcription results to all connected clients
                    await self.broadcast_transcript(transcript_data)
                    
                except json.JSONDecodeError:
                    # Handle non-JSON responses
                    logger.warning(f"Received non-JSON response: {result}")
                    await self.broadcast_transcript({"message": str(result)})
                
            except websockets.exceptions.ConnectionClosed:
                logger.info("HF Space WebSocket connection closed")
                self.is_connected = False
                break
            except Exception as e:
                logger.error(f"Error receiving transcription results: {e}")
                self.is_connected = False
                break
        
        # Attempt reconnection
        if not self.is_connected:
            asyncio.create_task(self.auto_reconnect())
    
    async def broadcast_transcript(self, transcript_data):
        """Broadcast transcription results to all connected clients"""
        if not self.transcript_clients:
            return
            
        # Prepare message for clients
        message = json.dumps({
            "type": "transcription",
            "timestamp": asyncio.get_event_loop().time(),
            "data": transcript_data
        })
        
        # Remove disconnected clients while broadcasting
        disconnected_clients = set()
        
        for client in self.transcript_clients.copy():
            try:
                await client.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send transcription to client: {e}")
                disconnected_clients.add(client)
        
        # Clean up disconnected clients
        self.transcript_clients -= disconnected_clients
        
        if disconnected_clients:
            logger.info(f"Removed {len(disconnected_clients)} disconnected clients")
    
    def add_transcript_client(self, websocket: WebSocket):
        """Add a client for receiving transcription results"""
        self.transcript_clients.add(websocket)
        logger.info(f"Transcript client added. Total clients: {len(self.transcript_clients)}")
    
    def remove_transcript_client(self, websocket: WebSocket):
        """Remove a transcription client"""
        self.transcript_clients.discard(websocket)
        logger.info(f"Transcript client removed. Total clients: {len(self.transcript_clients)}")
    
    async def emit(self):
        """Called by FastRTC - no need to emit anything here as we handle in broadcast"""
        return None

# Initialize the audio transcription relay
audio_relay = AudioTranscriptionRelay()

# Set up FastRTC stream for real-time audio
stream = Stream(handler=audio_relay, modality="audio", mode="send-receive")
stream.mount(app)

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    logger.info("Starting audio transcription relay server")
    await audio_relay.start_up()

@app.on_event("shutdown") 
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down audio transcription relay server")
    await audio_relay.shutdown()

@app.get("/")
@app.head("/")
async def root():
    return {
        "message": "Real-time Audio Transcription & Speaker Diarization Relay Server",
        "service": "FastRTC → HuggingFace Space Audio Processing"
    }

@app.get("/health")
@app.head("/health")
async def health(): 
    return {
        "status": "ok", 
        "hf_space_connected": audio_relay.is_connected,
        "transcript_clients": len(audio_relay.transcript_clients),
        "hf_space_url": HF_SPACE_URL
    }

@app.websocket("/ws_transcription")
async def websocket_transcription(websocket: WebSocket):
    """WebSocket endpoint for receiving real-time transcription results"""
    await websocket.accept()
    logger.info("Transcription WebSocket client connected")
    
    # Add this client to receive transcription broadcasts
    audio_relay.add_transcript_client(websocket)
    
    # Send connection confirmation
    await websocket.send_text(json.dumps({
        "type": "connection",
        "status": "connected",
        "hf_space_status": "connected" if audio_relay.is_connected else "disconnected"
    }))
    
    try:
        # Keep connection alive and handle any client messages
        while True:
            try:
                # Listen for any client messages (like configuration changes)
                client_message = await websocket.receive_text()
                logger.info(f"Received client message: {client_message}")
                
                # Handle client configuration if needed
                try:
                    client_data = json.loads(client_message)
                    if client_data.get("type") == "config":
                        # Handle configuration changes for transcription
                        logger.info(f"Configuration update: {client_data}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {client_message}")
                
            except WebSocketDisconnect:
                logger.info("Transcription client disconnected")
                break
            except Exception as e:
                logger.error(f"Error in transcription WebSocket: {e}")
                break
                
    except Exception as e:
        logger.error(f"Transcription WebSocket error: {e}")
    finally:
        audio_relay.remove_transcript_client(websocket)
        try:
            await websocket.close()
        except:
            pass

# Legacy endpoint for backward compatibility
@app.websocket("/ws_relay")
async def websocket_relay_legacy(websocket: WebSocket):
    """Legacy WebSocket endpoint - redirects to transcription"""
    await websocket_transcription(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 10000)),
        log_level="info"
    )