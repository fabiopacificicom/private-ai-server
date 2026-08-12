from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    timeout: Optional[int] = 120


class MultimodalMessage(BaseModel):
    role: str
    # Accepts plain text or HF-style content blocks: [{"type":"image","url":"..."}, ...]
    content: Union[str, List[Dict[str, Any]]]
    images: Optional[List[str]] = None  # base64 or https:// URLs
    audio: Optional[str] = None         # base64 WAV/MP3/FLAC
    video: Optional[str] = None         # base64 MP4 or https:// URL


class MultimodalChatRequest(BaseModel):
    model: str
    messages: List[MultimodalMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    timeout: Optional[int] = 180
