import json
import base64
import os
import re
from typing import Optional

TOOLS = [
    {
        "name": "open_app",
        "description": "Open any installed application on Windows (e.g., VS Code, Android Studio, Notepad, Photoshop, Chrome, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "description": "Name or path of the application to open"}
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "take_screenshot",
        "description": "Take a screenshot of the current screen and analyze it",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "mouse_click",
        "description": "Click at specific screen coordinates",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right", "double"], "default": "left"}
            },
            "required": ["x", "y"]
        }
    },
    {
        "name": "keyboard_type",
        "description": "Type text using the keyboard",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "press_enter": {"type": "boolean", "default": False}
            },
            "required": ["text"]
        }
    },
    {
        "name": "keyboard_shortcut",
        "description": "Press a keyboard shortcut (e.g., ctrl+c, ctrl+v, alt+f4)",
        "parameters": {
            "type": "object",
            "properties": {
                "keys": {"type": "string", "description": "Keys separated by + (e.g., ctrl+s)"}
            },
            "required": ["keys"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Full file path"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write or create a file with given content",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_directory",
        "description": "List files and folders in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path, default is Desktop"}
            }
        }
    },
    {
        "name": "delete_file",
        "description": "Delete a file or folder",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web using DuckDuckGo",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "create_word_document",
        "description": "Create a Microsoft Word (.docx) document with content",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "title": {"type": "string"},
                "content": {"type": "string", "description": "Main document content"},
                "save_path": {"type": "string", "description": "Where to save (default: Desktop)"}
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "create_powerpoint",
        "description": "Create a PowerPoint (.pptx) presentation",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "title": {"type": "string"},
                "slides": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                },
                "save_path": {"type": "string"}
            },
            "required": ["filename", "slides"]
        }
    },
    {
        "name": "create_pdf",
        "description": "Create a PDF document",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "title": {"type": "string"},
                "content": {"type": "string"},
                "save_path": {"type": "string"}
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "generate_image",
        "description": "Generate an image using AI (Pollinations AI - free)",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Detailed image description"},
                "width": {"type": "integer", "default": 1024},
                "height": {"type": "integer", "default": 1024},
                "save_path": {"type": "string", "description": "Where to save the image"}
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "edit_image",
        "description": "Edit an existing image (resize, crop, rotate, brightness, contrast, filters)",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string"},
                "output_path": {"type": "string"},
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["resize", "crop", "rotate", "brightness", "contrast", "grayscale", "blur", "sharpen", "flip"]},
                            "params": {"type": "object"}
                        }
                    }
                }
            },
            "required": ["input_path", "operations"]
        }
    },
    {
        "name": "run_command",
        "description": "Run a terminal command (cmd/powershell)",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "shell": {"type": "string", "enum": ["cmd", "powershell"], "default": "cmd"}
            },
            "required": ["command"]
        }
    },
    {
        "name": "scroll",
        "description": "Scroll the mouse wheel",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "integer", "description": "Positive = up, negative = down"},
                "x": {"type": "integer", "description": "X position (optional)"},
                "y": {"type": "integer", "description": "Y position (optional)"}
            },
            "required": ["amount"]
        }
    }
]

SYSTEM_PROMPT = """You are PeacockAgent — a powerful AI assistant that can control a Windows computer.

You have access to tools that let you:
- Open any application (VS Code, Android Studio, Photoshop, Chrome, etc.)
- Take screenshots to see the current state of the screen
- Control mouse and keyboard to interact with any application
- Read, write, create, and delete files
- Search the web
- Create Word documents, PowerPoint presentations, and PDFs
- Generate AI images using Pollinations AI (free, no API key needed)
- Edit existing images (resize, crop, rotate, filters, etc.)
- Run terminal commands

IMPORTANT RULES:
1. Always take a screenshot first when you need to see the current state of the screen
2. Break complex tasks into small steps and execute them one by one
3. After clicking or typing, take another screenshot to verify the action worked
4. When creating files, always confirm the save location with the user first
5. Be careful with delete operations — always confirm before deleting
6. For image generation, use detailed English prompts for best results
7. Respond in the same language the user uses (Arabic or English)
8. Always explain what you're doing before doing it

You are PeacockAI's flagship agent. Be helpful, precise, and proactive."""


def build_messages(history: list, user_message: str, image_data: Optional[str] = None) -> list:
    messages = []
    for msg in history:
        messages.append(msg)

    if image_data:
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            {"type": "text", "text": user_message}
        ]
    else:
        content = user_message

    messages.append({"role": "user", "content": content})
    return messages
