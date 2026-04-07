# 🦚 PeacockAgent — AI Desktop Assistant

An AI agent that controls your Windows PC using natural language.

## Features
- 🖥 **Computer Control** — Opens apps, clicks, types, scrolls
- 📁 **File Management** — Read, write, create, delete files
- 🌐 **Web Search** — DuckDuckGo (no API key needed)
- 📄 **Word Documents** — Create .docx files
- 📊 **PowerPoint** — Create .pptx presentations
- 📑 **PDF** — Create PDF documents
- 🖼 **AI Image Generation** — Via Pollinations AI (free, no key needed)
- ✏️ **Image Editing** — Resize, crop, rotate, filters
- 💻 **App Control** — VS Code, Android Studio, Photoshop, Chrome...
- 📎 **File Upload** — Attach files to your messages

## Setup

### Requirements
- Python 3.10+
- Windows 10/11

### Install & Run
Double-click `RUN.bat` — it installs everything and launches automatically.

Or manually:
```bash
pip install -r requirements.txt
python main.py
```

## Configuration
1. Open the app
2. Click ⚙️ **Settings**
3. Choose provider:
   - **Groq** (recommended) — Get free API key at https://console.groq.com
   - **Ollama** (local) — Install from https://ollama.com

## Keyboard Shortcuts
- `Ctrl+Enter` — Send message

## Models (Groq - Free)
- `llama-3.3-70b-versatile` ← recommended
- `llama-3.1-70b-versatile`
- `mixtral-8x7b-32768`

## Architecture
```
PeacockAgent/
├── main.py              # Entry point
├── core/
│   ├── agent.py         # Tool definitions + system prompt
│   └── llm_provider.py  # Groq + Ollama
├── tools/
│   └── executor.py      # All tool implementations
├── ui/
│   └── app_window.py    # PyQt6 UI
├── requirements.txt
└── RUN.bat
```

Built by PeacockAI
