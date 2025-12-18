# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-18

### Added
- **Phase 1 Complete**: All production-ready features implemented
- **Persistent Job Storage (1.5)**: SQLite-based job persistence that survives server restarts
  - Jobs stored in `jobs.db` with complete schema
  - Background job operations now persist across server restarts
  - Job history queryable with filters and status-based queries
  - Added `database.py` module with `JobDatabase` class
- **Improved Error Messages (1.6)**: User-friendly actionable error messages
  - Clear installation instructions for missing dependencies
  - Specific commands for downloading models (`POST /pull {"model":"gpt2"}`)
  - Better HTTP status codes (400 for user errors, 429 for cooldown)
  - Troubleshooting guidance for CUDA/GPU issues
- **Streaming Support (1.1)**: Real-time response streaming via Server-Sent Events
  - `/chat` endpoint supports `stream=true` parameter  
  - Works with both vLLM and transformers backends
  - JSON chunks with `{"delta": {"content": "text"}, "done": false}` format
- **Progress Tracking (1.2)**: Real-time download progress for model pulls
  - Progress field (0.0-1.0) updated every 2 seconds during downloads
  - `downloaded_bytes` and `total_bytes` tracking
  - Works for large models with accurate progress calculation
- **Health Check Endpoint (1.3)**: Production monitoring and load balancer support
  - `GET /health` returns comprehensive server status
  - GPU memory statistics when CUDA available
  - Cache and download queue metrics
  - Uptime tracking and backend availability status
- **Request Timeout & Cancellation (1.4)**: Configurable request timeouts
  - Per-request timeout support (1-600 seconds)
  - Automatic GPU resource cleanup on timeout
  - HTTP 408 responses for timed-out requests
  - Non-blocking timeout implementation with `asyncio.wait_for()`

### Changed
- **Job System**: Migrated from in-memory dict to persistent SQLite storage
- **Error Handling**: Comprehensive error message improvements across all endpoints
- **Performance**: Better GPU memory management with timeout-based cleanup
- **API**: Enhanced `/jobs/{id}` endpoint with progress tracking fields
- **Documentation**: Updated README.md formatting and structure

### Fixed
- **Windows Compatibility**: Resolved symlink privilege issues with `HF_HUB_DISABLE_SYMLINKS`
- **Memory Management**: Improved CUDA memory cleanup to prevent fragmentation
- **Job Persistence**: Jobs no longer lost on server restart

### Technical Details
- Added SQLite schema with 16 fields for comprehensive job tracking
- Implemented async job polling with progress updates every 2 seconds
- Enhanced error messages with specific pip install commands
- Improved HTTP status code semantics (400/429/500)
- Added database migration support for future schema changes

### Validation
- ✅ All Phase 1 acceptance criteria met
- ✅ Backward compatibility maintained
- ✅ Comprehensive test suite passing
- ✅ Production deployment ready

---

## [0.9.0] - 2025-12-17

### Added
- **Core MVP Features**: Complete inference server with GPU optimization
- **Dynamic Model Loading**: vLLM and transformers backend fallback
- **Background Downloads**: Non-blocking `/pull` endpoint with job tracking
- **Automatic Quantization**: 4-bit quantization for models >14GB
- **LRU Model Caching**: Configurable model cache with memory management
- **CUDA Optimization**: Memory management and OOM prevention
- **Ollama Compatibility**: Compatible `/chat` API endpoint

### Technical Foundation
- FastAPI server with comprehensive error handling
- Windows GPU support with CUDA 12.6
- HuggingFace Hub integration with local model caching
- Quantization support via bitsandbytes
- Comprehensive logging and diagnostics

### Validated Models
- gpt2 (548MB), Qwen2-0.6B (1.5GB), opt-1.3b (5.3GB)
- Qwen3-Omni-30B (70GB with 4-bit quantization)

### Documentation
- Complete API documentation and usage examples
- Architecture documentation and coding guidelines  
- Detailed roadmap and implementation plans

---

## Development Notes

- **Phase 1 Goals**: Achieved production stability, monitoring, and improved UX
- **Next Phase**: Phase 2 will focus on advanced features (sessions, sampling, aliases)
- **Testing**: All features validated on RTX 4090, Windows 11, CUDA 12.6