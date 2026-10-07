# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Private AI Inference Server.
#
# Produces a one-folder build (dist/ai-server/) containing the server binary,
# the src/ package, the static web UI, and all Python dependencies.
#
# Build with:
#   pyinstaller server.spec
# or via scripts/build-server.ps1

import os

block_cipher = None

# ---------------------------------------------------------------------------
# Collect data files the server needs at runtime.
# ---------------------------------------------------------------------------
datas = [
    # Static web UI (served by app.py at the root)
    ('static', 'static'),
    # Persisted model manifest (created at runtime, but ship a default if any)
    ('models_manifest.json', '.'),
]

# ---------------------------------------------------------------------------
# Hidden imports for modules loaded dynamically or not auto-detected.
# ---------------------------------------------------------------------------
hiddenimports = [
    # FastAPI / uvicorn
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    # Pydantic / FastAPI runtime lookups
    'pydantic',
    'pydantic_core',
    # Backend libs (optional; availability is probed at runtime)
    'transformers',
    'torch',
    'huggingface_hub',
    'llama_cpp',
]

# ---------------------------------------------------------------------------
# The server entry point.
# ---------------------------------------------------------------------------
a = Analysis(
    ['app.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy ML/runtime libraries are excluded from the base build to keep
        # the installer under the NSIS 2GB limit. The setup wizard provisions
        # them on first run (CUDA torch + llama-cpp) based on the GPU.
        'vllm',
        'tensorflow',
        'keras',
        'matplotlib',
        'notebook',
        'IPython',
        'torch',
        'torchvision',
        'torchaudio',
        'transformers',
        'bitsandbytes',
        'accelerate',
        'safetensors',
        'tokenizers',
        'sentencepiece',
        'scipy',
        'numpy',
        'PIL',
        'cv2',
        'mistral_common',
        'tiktoken',
        'protobuf',
        'huggingface_hub',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ai-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ai-server',
)
