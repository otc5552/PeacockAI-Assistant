<div align="center">

<img src="https://img.shields.io/badge/-%F0%9F%A6%9A%20PeacockAI-00D4AA?style=for-the-badge&logoColor=white" height="40"/>

# PeacockAI Assistant Platform

**The Unified AI Assistant — One Interface. Infinite Capabilities.**

[![Python](https://img.shields.io/badge/Python-3.11+-3572A5?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React Native](https://img.shields.io/badge/React_Native-0.75-61DAFB?style=flat-square&logo=react&logoColor=white)](https://reactnative.dev)
[![Firebase](https://img.shields.io/badge/Firebase-Auth-FFCA28?style=flat-square&logo=firebase&logoColor=black)](https://firebase.google.com)
[![License](https://img.shields.io/badge/License-MIT-00D4AA?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen?style=flat-square)]()

<br/>

*Built by [PeacockAI](https://peacockai.com) · Crafted for the next generation of AI-powered productivity*

</div>

---

## 📖 Overview

**PeacockAI Assistant Platform** is a unified, multi-tool AI assistant designed to consolidate the power of multiple AI capabilities into a single, seamless conversational interface.

Instead of switching between ChatGPT for answers, Midjourney for images, GitHub Copilot for code, and search engines for the web — PeacockAI brings it all into one place. The platform intelligently analyzes each user request and routes it to the most appropriate AI model or tool automatically, without requiring any manual configuration from the user.

> *"Ask anything. Create anything. Analyze anything — in one conversation."*

---

## ✨ Core Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI Chat** | Conversational assistant powered by large language models for Q&A, analysis, and reasoning |
| 💻 **Code Generation** | Write, debug, and explain code across 20+ programming languages |
| 🎨 **Image Generation** | Create stunning images from text descriptions using state-of-the-art diffusion models |
| 📄 **File Analysis** | Upload and analyze PDF, TXT, CSV, DOCX files with intelligent summarization |
| 🔎 **Web Search** | Real-time internet search integrated directly into the conversation flow |
| 🧮 **Math Solver** | Step-by-step solutions for equations, calculus, algebra, and complex problems |
| 📱 **App & Web Builder** | Generate full React Native or HTML/CSS/JS project scaffolding from a description |
| 🔌 **Extensible Tool System** | Modular architecture designed for adding new tools and AI integrations |

---

## 🏗️ Architecture Overview

PeacockAI is built around an **Intent-Driven Routing** architecture. Every message from the user passes through an intelligent pipeline before reaching any AI model:

```
┌─────────────────────────────────────────────────────────┐
│                        USER                             │
│           "Write me a Python web scraper"               │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  CHAT INTERFACE                         │
│         Unified conversational entry point              │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              INTENT ENGINE                              │
│   Analyzes message · Detects intent · Scores confidence │
│                                                         │
│   "Write" + "Python" → intent: coding  (conf: 0.94)    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  AI ROUTER                              │
│   Maps intent → optimal model or tool                   │
└──────┬──────────┬──────────┬──────────┬────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
  ┌─────────┐ ┌───────┐ ┌────────┐ ┌────────────┐
  │  Code   │ │ Image │ │  Web   │ │   File     │
  │ Service │ │  Gen  │ │ Search │ │  Analyzer  │
  │ Llama   │ │ DALL·E│ │  DDG   │ │  Gemini    │
  │  70B    │ │ /SDXL │ │  API   │ │  Vision    │
  └─────────┘ └───────┘ └────────┘ └────────────┘
```

The system is fully **stateless per request** and **horizontally scalable**. New tools can be registered in the router without modifying any existing service.

---

## 🛠️ Tech Stack

### Frontend
| Layer | Technology |
|-------|-----------|
| Mobile Framework | React Native 0.75 (Android-first) |
| Language | TypeScript |
| Navigation | React Navigation v6 |
| State | React Hooks + AsyncStorage |
| UI Components | Custom Design System |
| Authentication | Firebase SDK + Google Sign-In |

### Backend
| Layer | Technology |
|-------|-----------|
| Framework | FastAPI (Python 3.11+) |
| Database | SQLite + aiosqlite (async) |
| Auth | Firebase Admin SDK |
| HTTP Client | httpx (async) |
| Validation | Pydantic v2 |
| Server | Uvicorn (ASGI) |

### AI Integrations
| Provider | Used For | Pricing |
|----------|----------|---------|
| Groq (Llama 3.3 70B) | Chat, Code, Writing, Math | Free tier |
| Google Gemini Flash | File analysis, Vision | Free tier |
| Pollinations AI | Image generation | Free |
| DuckDuckGo API | Web search | Free |
| Cohere | Embeddings (planned) | Free tier |

### Infrastructure
| Service | Purpose |
|---------|---------|
| Firebase Authentication | Google Sign-In, Token Verification |
| Firebase Firestore | (planned) Cloud sync |
| Firebase Cloud Messaging | Push notifications (planned) |

---

## 📸 Screenshots

<div align="center">

| Splash Screen | Login | AI Chat |
|:---:|:---:|:---:|
| *Coming Soon* | *Coming Soon* | *Coming Soon* |

| Tools Menu | Code Builder | Image Generator |
|:---:|:---:|:---:|
| *Coming Soon* | *Coming Soon* | *Coming Soon* |

| File Analyzer | Dashboard | Settings |
|:---:|:---:|:---:|
| *Coming Soon* | *Coming Soon* | *Coming Soon* |

> Screenshots will be added upon first stable release (v1.0.0)

</div>

---

## 🗂️ Project Structure

```
PeacockAI-Assistant/
│
├── 📁 frontend/                        # React Native Mobile App
│   │
│   ├── 📁 screens/                     # Application Screens
│   │   ├── SplashScreen.tsx            # Animated launch screen
│   │   ├── LoginScreen.tsx             # Google OAuth login
│   │   ├── ChatScreen.tsx              # Main AI chat interface ⭐
│   │   ├── DashboardScreen.tsx         # User profile & history
│   │   ├── SettingsScreen.tsx          # App preferences
│   │   ├── ImageGenScreen.tsx          # Image creation studio
│   │   ├── CodeBuilderScreen.tsx       # Code editor & viewer
│   │   └── FileAnalyzerScreen.tsx      # Document upload & analysis
│   │
│   ├── 📁 components/
│   │   ├── chat/
│   │   │   └── MessageBubble.tsx       # Markdown-rendered message bubbles
│   │   └── tools/
│   │       └── ToolsMenu.tsx           # Horizontal tool picker
│   │
│   ├── 📁 navigation/
│   │   └── AppNavigator.tsx            # Stack + Tab navigation
│   │
│   └── 📁 utils/
│       ├── constants.ts                # Colors, tools config, labels
│       └── api.ts                      # Typed API service layer
│
└── 📁 backend/                         # FastAPI Python Server
    │
    ├── main.py                         # App entry point + lifespan
    ├── database.py                     # Async SQLite ORM layer
    ├── requirements.txt
    ├── .env.example
    │
    ├── 📁 api/                         # HTTP Route Handlers
    │   ├── chat.py                     # /api/chat — main chat endpoint
    │   ├── auth.py                     # /api/auth — token verification
    │   ├── users.py                    # /api/users — profile & plans
    │   └── tools.py                    # /api/tools — direct tool access
    │
    ├── 📁 intent_engine/               # NLU Layer
    │   └── engine.py                   # Keyword + pattern intent classifier
    │
    ├── 📁 router/                      # Orchestration Layer
    │   └── ai_router.py                # Intent → Service dispatcher
    │
    ├── 📁 services/                    # AI Service Adapters
    │   ├── base_service.py             # Abstract base class
    │   ├── all_services.py             # All service implementations
    │   ├── coding_service.py
    │   ├── image_gen_service.py
    │   ├── search_service.py
    │   ├── file_service.py
    │   ├── writing_service.py
    │   ├── math_service.py
    │   └── general_service.py
    │
    ├── 📁 models/
    │   └── request_models.py           # Pydantic request/response schemas
    │
    └── 📁 middleware/
        └── auth_middleware.py          # Firebase JWT verification
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- Java JDK 17
- Android Studio (Hedgehog or later)
- A Firebase project

### Backend Setup

```bash
# 1. Navigate to backend
cd backend

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Add Firebase credentials
# Download firebase-credentials.json from Firebase Console
# → Project Settings → Service Accounts → Generate New Private Key
# Place it in the /backend directory

# 6. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API Docs available at: `http://localhost:8000/docs`

### Frontend Setup

```bash
# 1. Navigate to frontend
cd frontend

# 2. Install packages
npm install

# 3. Add google-services.json
# Firebase Console → Project Settings → Android
# Register package: com.peacockai.assistant
# Download google-services.json → place in android/app/

# 4. Configure Google Sign-In
# Edit screens/LoginScreen.tsx:
# Replace YOUR_WEB_CLIENT_ID with your Firebase Web Client ID

# 5. Run on Android
npx react-native run-android
```

### Environment Variables

```env
# backend/.env

GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx       # console.groq.com (Free)
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxx       # aistudio.google.com (Free)
COHERE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx     # cohere.com (Free tier)
FIREBASE_CREDENTIALS_PATH=firebase-credentials.json
DEV_MODE=false
SECRET_KEY=change-this-in-production
```

---

## 🔌 API Reference

All endpoints require: `Authorization: Bearer <firebase_id_token>`

### `POST /api/chat/message`
Main conversational endpoint. Automatically routes to the appropriate AI service.

```json
// Request
{
  "message": "Create a REST API in Python using FastAPI",
  "session_id": "sess_abc123",
  "user_id": "uid_xyz789",
  "tool_hint": null,
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}

// Response
{
  "message": "```python\nfrom fastapi import FastAPI\n...",
  "intent": "coding",
  "confidence": 0.94,
  "data": { "mode": "code" },
  "session_id": "sess_abc123"
}
```

### `POST /api/chat/message/with-file`
Multipart upload for document analysis.

```
Content-Type: multipart/form-data
Fields: message, session_id, user_id, tool_hint, file
```

### `GET /api/chat/sessions/{user_id}`
Retrieve paginated conversation history.

### `POST /api/tools/image/generate`
Direct image generation endpoint.

```json
// Request
{ "prompt": "A futuristic city at sunset", "style": "cinematic", "width": 768, "height": 768 }

// Response
{ "image_url": "https://image.pollinations.ai/prompt/...", "prompt": "..." }
```

---

## 🧩 Adding a New Tool

PeacockAI is designed for extensibility. To add a new capability:

**1. Create a service** in `backend/services/all_services.py`:
```python
class TranslationService(BaseAIService):
    async def process(self, message, intent, session_id, **kwargs) -> ServiceResponse:
        # Call your translation API here
        translated = await call_groq([
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": message}
        ])
        return ServiceResponse(text=translated, data={"type": "translation"})
```

**2. Register the intent** in `backend/intent_engine/engine.py`:
```python
Intent.TRANSLATE: [
    r"\b(ترجم|translate|translation|بالإنجليزية|بالعربية)\b"
]
```

**3. Register the service** in `backend/router/ai_router.py`:
```python
Intent.TRANSLATE: TranslationService(),
```

**4. Add the tool card** in `frontend/utils/constants.ts`:
```typescript
{ id: "translate", label: "ترجمة", icon: "globe", color: "#A29BFE" }
```

That's it. No other files need to change.

---

## 🗺️ Roadmap

### v1.1 — Enhanced Intelligence
- [ ] Streaming responses (Server-Sent Events)
- [ ] Conversation memory and context compression
- [ ] Multi-model routing with fallback chains
- [ ] Confidence threshold tuning

### v1.2 — Voice & Vision
- [ ] Voice input (Speech-to-Text)
- [ ] Voice output (Text-to-Speech)
- [ ] Camera integration for real-time image analysis
- [ ] OCR for document scanning

### v1.3 — Platform Expansion
- [ ] iOS support
- [ ] Web application (React)
- [ ] Desktop application (Electron)
- [ ] Browser extension

### v2.0 — Local AI
- [ ] On-device model inference (Llama.cpp / Ollama)
- [ ] Offline mode with quantized models
- [ ] Plugin marketplace
- [ ] Enterprise self-hosted deployment

---

## 🤝 Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please ensure your code follows the existing style conventions and includes appropriate documentation.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this software for personal and commercial purposes.

---

## 🏢 About

<div align="center">

**🦚 PeacockAI**

*Building the next generation of accessible, multi-modal AI tools*

| | |
|---|---|
| **Company** | PeacockAI |
| **Founder & Lead Developer** | Mostafa Ahmed Farghaly |
| **Platform** | Android · Web · API |
| **Contact** | support@peacockai.com |

<br/>

*© 2025 PeacockAI — All Rights Reserved*

</div>
