import requests
import json
from typing import Optional, Generator
from core.agent import TOOLS, SYSTEM_PROMPT


class LLMProvider:
    def __init__(self):
        self.provider = "groq"
        self.groq_api_key = ""
        self.groq_model = "llama-3.3-70b-versatile"
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "llama3"

    def set_provider(self, provider: str, **kwargs):
        self.provider = provider
        for k, v in kwargs.items():
            setattr(self, k, v)

    def chat(self, messages: list, on_token=None) -> dict:
        if self.provider == "groq":
            return self._groq_chat(messages, on_token)
        elif self.provider == "ollama":
            return self._ollama_chat(messages, on_token)
        else:
            return {"error": "Unknown provider"}

    def _groq_chat(self, messages: list, on_token=None) -> dict:
        if not self.groq_api_key:
            return {"error": "Groq API key not set. Go to Settings to add your key."}

        tools_schema = []
        for tool in TOOLS:
            tools_schema.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.groq_model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "tools": tools_schema,
            "tool_choice": "auto",
            "max_tokens": 4096,
            "stream": False
        }

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]["message"]
            return {
                "content": choice.get("content", ""),
                "tool_calls": choice.get("tool_calls", [])
            }
        except requests.exceptions.HTTPError as e:
            return {"error": f"Groq API error: {e.response.text}"}
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}

    def _ollama_chat(self, messages: list, on_token=None) -> dict:
        tools_schema = []
        for tool in TOOLS:
            tools_schema.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })

        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "tools": tools_schema,
            "stream": False
        }

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})
            return {
                "content": msg.get("content", ""),
                "tool_calls": msg.get("tool_calls", [])
            }
        except Exception as e:
            return {"error": f"Ollama error: {str(e)}"}
