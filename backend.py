from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import logging
import json
from fastrtc import Stream, AsyncStreamHandler
import websockets 
from typing import Set, Dict, Any
import base64
import time

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

class ConnectionManager:
    """Manages WebSocket connections and broadcasting"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Add a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = {
            "client_id": client_id or f"client_{int(time.time())}",
            "connected_at": time.time(),
            "messages_sent": 0
        }
        
        logger.info(f"WebSocket connected: {self.connection_metadata[websocket]['client_id']}. "
                   f"Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            client_info = self.connection_metadata.get(websocket, {})
            client_id = client_info.get("client_id", "unknown")
            
            self.active_connections.discard(websocket)
            self.connection_metadata.pop(websocket, None)
            
            logger.info(f"WebSocket disconnected: {client_id}. "
                       f"Remaining connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: str):
        """Broadcast message to all active connections"""
        if not self.active_connections:
            return
        
        disconnected = set()
        
        for websocket in self.active_connections.copy():
            try:
                await websocket.send_text(message)
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["messages_sent"] += 1
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)
    
    def get_stats(self):
        """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "connection_metadata": {
                k: v for k, v in enumerate(
                    [{"client_id": m.get("client_id", "unknown"), 
                      "connected_at": m.get("connected_at", 0),
                      "messages_sent": m.get("messages_sent", 0)} 
                     for m in self.connection_metadata.values()]
                )
            }
        }

class AudioTranscriptionRelay(AsyncStreamHandler):
    """Handler to relay real-time audio for transcription and speaker diarization"""
    
    def __init__(self):
        super().__init__()
        self.hf_ws_connection = None
        self.connection_manager = ConnectionManager()
        self.connection_lock = asyncio.Lock()
        self.is_connected = False
        self.reconnect_task = None
        self.connection_stats = {
            "total_connections": 0,
            "last_audio_received": None,
            "total_audio_chunks": 0,
            "system_start_time": time.time()
        }
        
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
            
            # Update statistics
            self.connection_stats["last_audio_received"] = time.time()
            self.connection_stats["total_audio_chunks"] += 1
            
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
                        message_data = json.loads(result)
                        
                        # Log the message type
                        message_type = message_data.get("type", "unknown")
                        logger.info(f"Received message of type: {message_type} from HF Space")
                        
                        if message_type == "processing_result":
                            # This is a processing result from process_audio_chunk
                            logger.debug(f"Processing result: {message_data.get('data', {})}")
                            
                        elif message_type == "conversation_update":
                            # This is a conversation update
                            logger.info("Received conversation update")
                            
                        elif message_type == "error":
                            # This is an error message
                            logger.warning(f"Received error from HF Space: {message_data.get('message', 'Unknown error')}")
                            
                        # Forward all messages to clients regardless of type
                        await self.broadcast_transcript(message_data)
                        
                    else:
                        # Handle binary data if needed
                        message_data = {"type": "binary_data", "data": {"raw_data": base64.b64encode(result).decode()}}
                        await self.broadcast_transcript(message_data)
                    
                except json.JSONDecodeError:
                    # Handle non-JSON responses
                    logger.warning(f"Received non-JSON response: {result}")
                    await self.broadcast_transcript({"type": "raw_message", "data": {"message": str(result)}})
                
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
    
    async def broadcast_transcript(self, data):
        """Broadcast data to all connected clients"""
        if not self.connection_manager.active_connections:
            return
            
        # If data is already a dictionary with a type, use it directly
        # Otherwise wrap it in a transcription message
        if isinstance(data, dict) and "type" in data:
            message = json.dumps(data)
        else:
            # Prepare message for clients
            message = json.dumps({
                "type": "transcription",
                "timestamp": time.time(),
                "data": data
            })
        
        # Use connection manager to broadcast
        await self.connection_manager.broadcast(message)
    
    async def emit(self):
        """Called by FastRTC - no need to emit anything here as we handle in broadcast"""
        return None
    
    def get_stats(self):
        """Get comprehensive connection statistics"""
        ws_stats = self.connection_manager.get_stats()
        
        return {
            "hf_space_connected": self.is_connected,
            "connection_stats": self.connection_stats,
            "websocket_stats": ws_stats,
            "system_uptime": time.time() - self.connection_stats["system_start_time"]
        }

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
        "transcript_clients": len(audio_relay.connection_manager.active_connections),
        "hf_space_url": HF_SPACE_URL
    }

@app.get("/stats")
async def get_stats():
    """Get comprehensive connection statistics"""
    return audio_relay.get_stats()

@app.websocket("/ws_transcription")
async def websocket_transcription(websocket: WebSocket):
    """WebSocket endpoint for receiving real-time transcription results"""
    client_id = f"client_{int(time.time())}"
    
    try:
        # Add this client to receive transcription broadcasts
        await audio_relay.connection_manager.connect(websocket, client_id)
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "timestamp": time.time(),
            "hf_space_status": "connected" if audio_relay.is_connected else "disconnected"
        }))
        
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
        audio_relay.connection_manager.disconnect(websocket)

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