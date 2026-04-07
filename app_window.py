import os
import sys
import json
import base64
import threading
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel,
    QFileDialog, QScrollArea, QFrame, QSplitter,
    QTabWidget, QComboBox, QSpinBox, QCheckBox,
    QProgressBar, QStatusBar, QToolBar, QSizePolicy,
    QApplication, QMessageBox, QDialog, QFormLayout,
    QDialogButtonBox
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve
)
from PyQt6.QtGui import (
    QFont, QIcon, QColor, QPalette, QPixmap,
    QTextCursor, QKeyEvent, QAction, QPainter
)

from core.agent import build_messages, TOOLS
from core.llm_provider import LLMProvider
from tools.executor import dispatch_tool


# ══════════════════════════════════════════════════════════════════════════════
#  STYLE SHEET
# ══════════════════════════════════════════════════════════════════════════════
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0d0d0f;
    color: #e8e8f0;
    font-family: 'Segoe UI', sans-serif;
}

QTextEdit, QLineEdit {
    background-color: #16161a;
    color: #e8e8f0;
    border: 1px solid #2a2a35;
    border-radius: 8px;
    padding: 8px;
    font-size: 13px;
}

QTextEdit:focus, QLineEdit:focus {
    border: 1px solid #6c63ff;
}

QPushButton {
    background-color: #1e1e28;
    color: #e8e8f0;
    border: 1px solid #2a2a35;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2a2a38;
    border-color: #6c63ff;
}

QPushButton:pressed {
    background-color: #6c63ff;
}

QPushButton#send_btn {
    background-color: #6c63ff;
    color: white;
    font-weight: 600;
    min-width: 80px;
}

QPushButton#send_btn:hover {
    background-color: #7c73ff;
}

QPushButton#send_btn:disabled {
    background-color: #3a3a4a;
    color: #666;
}

QPushButton#stop_btn {
    background-color: #ff4757;
    color: white;
    font-weight: 600;
}

QPushButton#stop_btn:hover {
    background-color: #ff6b7a;
}

QScrollBar:vertical {
    background: #16161a;
    width: 6px;
    border-radius: 3px;
}

QScrollBar::handle:vertical {
    background: #3a3a4a;
    border-radius: 3px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #6c63ff;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QTabWidget::pane {
    border: 1px solid #2a2a35;
    border-radius: 8px;
    background: #16161a;
}

QTabBar::tab {
    background: #1e1e28;
    color: #888;
    padding: 8px 16px;
    border-radius: 6px;
    margin: 2px;
}

QTabBar::tab:selected {
    background: #6c63ff;
    color: white;
}

QComboBox {
    background-color: #1e1e28;
    color: #e8e8f0;
    border: 1px solid #2a2a35;
    border-radius: 6px;
    padding: 6px 12px;
}

QComboBox::drop-down {
    border: none;
}

QComboBox QAbstractItemView {
    background-color: #1e1e28;
    color: #e8e8f0;
    selection-background-color: #6c63ff;
}

QLabel#title_label {
    font-size: 20px;
    font-weight: 700;
    color: #6c63ff;
    letter-spacing: 2px;
}

QLabel#subtitle_label {
    font-size: 11px;
    color: #555;
    letter-spacing: 1px;
}

QFrame#chat_frame {
    background-color: #16161a;
    border-radius: 12px;
    border: 1px solid #2a2a35;
}

QFrame#input_frame {
    background-color: #16161a;
    border-radius: 12px;
    border: 1px solid #2a2a35;
    padding: 4px;
}

QStatusBar {
    background-color: #0d0d0f;
    color: #555;
    font-size: 11px;
}

QProgressBar {
    background-color: #1e1e28;
    border: none;
    border-radius: 3px;
    height: 4px;
}

QProgressBar::chunk {
    background-color: #6c63ff;
    border-radius: 3px;
}

QToolBar {
    background-color: #0d0d0f;
    border-bottom: 1px solid #2a2a35;
    spacing: 4px;
    padding: 4px;
}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════
class AgentWorker(QThread):
    message_ready = pyqtSignal(str, str)   # role, content
    tool_called = pyqtSignal(str, str)     # tool_name, result
    thinking = pyqtSignal(str)
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, llm: LLMProvider, messages: list, parent=None):
        super().__init__(parent)
        self.llm = llm
        self.messages = list(messages)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            current_messages = list(self.messages)
            max_iterations = 10

            for iteration in range(max_iterations):
                if self._stop:
                    break

                self.thinking.emit(f"🤔 Thinking... (step {iteration + 1})")
                response = self.llm.chat(current_messages)

                if "error" in response:
                    self.error.emit(response["error"])
                    return

                content = response.get("content", "")
                tool_calls = response.get("tool_calls", [])

                if content:
                    self.message_ready.emit("assistant", content)

                if not tool_calls:
                    break

                # Process tool calls
                tool_results = []
                for tc in tool_calls:
                    if self._stop:
                        break

                    # Handle both Groq and Ollama formats
                    if isinstance(tc, dict):
                        fn = tc.get("function", tc)
                        tool_name = fn.get("name", "")
                        args_raw = fn.get("arguments", fn.get("parameters", {}))
                        if isinstance(args_raw, str):
                            import json
                            try:
                                args = json.loads(args_raw)
                            except:
                                args = {}
                        else:
                            args = args_raw
                    else:
                        continue

                    self.thinking.emit(f"🔧 Using tool: {tool_name}")
                    result = dispatch_tool(tool_name, args)
                    self.tool_called.emit(tool_name, result)

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", tool_name),
                        "content": result
                    })

                # Add assistant message + tool results to history
                current_messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls
                })
                current_messages.extend(tool_results)

            self.done.emit()

        except Exception as e:
            self.error.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS DIALOG
# ══════════════════════════════════════════════════════════════════════════════
class SettingsDialog(QDialog):
    def __init__(self, llm: LLMProvider, parent=None):
        super().__init__(parent)
        self.llm = llm
        self.setWindowTitle("⚙️ Settings")
        self.setMinimumWidth(450)
        self.setStyleSheet(DARK_STYLE)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Settings")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #6c63ff;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(12)

        # Provider
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Groq (Cloud - Fast)", "Ollama (Local)"])
        self.provider_combo.setCurrentIndex(0 if self.llm.provider == "groq" else 1)
        form.addRow("Provider:", self.provider_combo)

        # Groq API key
        self.groq_key = QLineEdit()
        self.groq_key.setPlaceholderText("gsk_...")
        self.groq_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.groq_key.setText(self.llm.groq_api_key)
        form.addRow("Groq API Key:", self.groq_key)

        # Groq model
        self.groq_model = QComboBox()
        self.groq_model.addItems([
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ])
        idx = self.groq_model.findText(self.llm.groq_model)
        if idx >= 0:
            self.groq_model.setCurrentIndex(idx)
        form.addRow("Groq Model:", self.groq_model)

        # Ollama URL
        self.ollama_url = QLineEdit()
        self.ollama_url.setText(self.llm.ollama_url)
        form.addRow("Ollama URL:", self.ollama_url)

        # Ollama model
        self.ollama_model_edit = QLineEdit()
        self.ollama_model_edit.setText(self.llm.ollama_model)
        form.addRow("Ollama Model:", self.ollama_model_edit)

        layout.addLayout(form)

        note = QLabel("💡 Get free Groq API key at: console.groq.com")
        note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(note)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save(self):
        provider = "groq" if self.provider_combo.currentIndex() == 0 else "ollama"
        self.llm.set_provider(
            provider,
            groq_api_key=self.groq_key.text().strip(),
            groq_model=self.groq_model.currentText(),
            ollama_url=self.ollama_url.text().strip(),
            ollama_model=self.ollama_model_edit.text().strip()
        )
        # Save to config file
        config = {
            "provider": provider,
            "groq_api_key": self.groq_key.text().strip(),
            "groq_model": self.groq_model.currentText(),
            "ollama_url": self.ollama_url.text().strip(),
            "ollama_model": self.ollama_model_edit.text().strip()
        }
        config_path = os.path.join(Path.home(), ".peacockagent_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        self.accept()


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT BUBBLE
# ══════════════════════════════════════════════════════════════════════════════
class ChatDisplay(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Segoe UI", 13))

    def add_message(self, role: str, content: str):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        if role == "user":
            html = f'''
            <div style="margin: 12px 0; text-align: right;">
                <div style="display: inline-block; background: #6c63ff; color: white;
                            border-radius: 16px 16px 4px 16px; padding: 10px 16px;
                            max-width: 70%; font-size: 13px; line-height: 1.5;">
                    {content.replace(chr(10), "<br>")}
                </div>
                <div style="color: #555; font-size: 10px; margin-top: 4px;">You</div>
            </div>
            '''
        elif role == "assistant":
            html = f'''
            <div style="margin: 12px 0;">
                <div style="color: #6c63ff; font-size: 11px; margin-bottom: 4px; font-weight: 600;">
                    🦚 PeacockAgent
                </div>
                <div style="background: #1e1e28; color: #e8e8f0;
                            border-radius: 4px 16px 16px 16px; padding: 10px 16px;
                            max-width: 80%; font-size: 13px; line-height: 1.6;
                            border-left: 3px solid #6c63ff;">
                    {content.replace(chr(10), "<br>")}
                </div>
            </div>
            '''
        elif role == "tool":
            tool_name, result = content.split("|||", 1) if "|||" in content else ("tool", content)
            html = f'''
            <div style="margin: 6px 0 6px 24px;">
                <div style="background: #0d1a0d; color: #4caf50;
                            border-radius: 6px; padding: 8px 12px;
                            font-size: 11px; font-family: 'Consolas', monospace;
                            border-left: 3px solid #4caf50;">
                    🔧 <b>{tool_name}</b><br>
                    {result[:300].replace(chr(10), "<br>")}{"..." if len(result) > 300 else ""}
                </div>
            </div>
            '''
        elif role == "thinking":
            html = f'''
            <div style="margin: 4px 0 4px 24px; color: #888; font-size: 11px; font-style: italic;">
                {content}
            </div>
            '''
        elif role == "error":
            html = f'''
            <div style="margin: 8px 0; background: #1a0d0d; color: #ff6b6b;
                        border-radius: 8px; padding: 10px 14px;
                        border-left: 3px solid #ff4757;">
                ❌ {content}
            </div>
            '''
        else:
            html = f"<div style='margin:8px 0; color: #888;'>{content}</div>"

        self.insertHtml(html)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class PeacockAgentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.llm = LLMProvider()
        self.history = []
        self.worker = None
        self.uploaded_file = None

        self._load_config()
        self._build_ui()
        self.setWindowTitle("PeacockAgent — AI Desktop Assistant")
        self.resize(1100, 780)
        self.setMinimumSize(800, 600)

    def _load_config(self):
        config_path = os.path.join(Path.home(), ".peacockagent_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                self.llm.set_provider(
                    cfg.get("provider", "groq"),
                    groq_api_key=cfg.get("groq_api_key", ""),
                    groq_model=cfg.get("groq_model", "llama-3.3-70b-versatile"),
                    ollama_url=cfg.get("ollama_url", "http://localhost:11434"),
                    ollama_model=cfg.get("ollama_model", "llama3")
                )
            except:
                pass

    def _build_ui(self):
        self.setStyleSheet(DARK_STYLE)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = QFrame()
        header.setFixedHeight(64)
        header.setStyleSheet("background: #0d0d0f; border-bottom: 1px solid #2a2a35;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)

        logo_label = QLabel("🦚 PeacockAgent")
        logo_label.setObjectName("title_label")
        header_layout.addWidget(logo_label)

        header_layout.addStretch()

        self.status_indicator = QLabel("● Offline")
        self.status_indicator.setStyleSheet("color: #ff4757; font-size: 11px;")
        header_layout.addWidget(self.status_indicator)

        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.clicked.connect(self._open_settings)
        settings_btn.setFixedWidth(100)
        header_layout.addWidget(settings_btn)

        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self._clear_chat)
        clear_btn.setFixedWidth(80)
        header_layout.addWidget(clear_btn)

        main_layout.addWidget(header)

        # ── Body ──────────────────────────────────────────────────────────────
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(16, 16, 16, 16)
        body_layout.setSpacing(12)

        # Left: quick actions panel
        left_panel = QFrame()
        left_panel.setFixedWidth(180)
        left_panel.setStyleSheet("background: #16161a; border-radius: 12px; border: 1px solid #2a2a35;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 16, 12, 16)
        left_layout.setSpacing(8)

        panel_title = QLabel("Quick Actions")
        panel_title.setStyleSheet("color: #888; font-size: 11px; font-weight: 600; letter-spacing: 1px;")
        left_layout.addWidget(panel_title)

        quick_actions = [
            ("📸 Screenshot", "Take a screenshot of the current screen"),
            ("🌐 Web Search", "Search the web for: "),
            ("📄 Create Word", "Create a Word document about: "),
            ("📊 Create PPT", "Create a PowerPoint presentation about: "),
            ("📑 Create PDF", "Create a PDF document about: "),
            ("🖼 Gen Image", "Generate an image of: "),
            ("📁 List Desktop", "List the files on my Desktop"),
            ("💻 Open VS Code", "Open VS Code"),
        ]

        for label, prompt in quick_actions:
            btn = QPushButton(label)
            btn.setFixedHeight(36)
            btn.setStyleSheet("""
                QPushButton {
                    background: #1e1e28; border: 1px solid #2a2a35;
                    border-radius: 6px; color: #ccc; font-size: 12px;
                    text-align: left; padding-left: 8px;
                }
                QPushButton:hover { background: #2a2a38; color: #fff; border-color: #6c63ff; }
            """)
            btn.clicked.connect(lambda checked, p=prompt: self._quick_action(p))
            left_layout.addWidget(btn)

        left_layout.addStretch()

        body_layout.addWidget(left_panel)

        # Right: chat area
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # Chat display
        self.chat_display = ChatDisplay()
        self.chat_display.setObjectName("chat_frame")
        right_layout.addWidget(self.chat_display)

        # File upload bar (hidden by default)
        self.file_bar = QFrame()
        self.file_bar.setFixedHeight(36)
        self.file_bar.setStyleSheet("background: #1a1a2e; border-radius: 6px; border: 1px solid #6c63ff;")
        file_bar_layout = QHBoxLayout(self.file_bar)
        file_bar_layout.setContentsMargins(12, 0, 12, 0)
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: #6c63ff; font-size: 12px;")
        file_bar_layout.addWidget(self.file_label)
        file_bar_layout.addStretch()
        remove_file_btn = QPushButton("✕")
        remove_file_btn.setFixedSize(24, 24)
        remove_file_btn.clicked.connect(self._remove_file)
        file_bar_layout.addWidget(remove_file_btn)
        self.file_bar.hide()
        right_layout.addWidget(self.file_bar)

        # Input area
        input_frame = QFrame()
        input_frame.setObjectName("input_frame")
        input_frame.setFixedHeight(100)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(6)

        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("Ask PeacockAgent to do anything... (Ctrl+Enter to send)")
        self.input_box.setFixedHeight(52)
        self.input_box.installEventFilter(self)
        input_layout.addWidget(self.input_box)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        upload_btn = QPushButton("📎 Attach File")
        upload_btn.setFixedHeight(32)
        upload_btn.clicked.connect(self._upload_file)
        btn_row.addWidget(upload_btn)

        btn_row.addStretch()

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setFixedSize(80, 32)
        self.stop_btn.hide()
        self.stop_btn.clicked.connect(self._stop_agent)
        btn_row.addWidget(self.stop_btn)

        self.send_btn = QPushButton("Send ➤")
        self.send_btn.setObjectName("send_btn")
        self.send_btn.setFixedSize(100, 32)
        self.send_btn.clicked.connect(self._send_message)
        btn_row.addWidget(self.send_btn)

        input_layout.addLayout(btn_row)
        right_layout.addWidget(input_frame)

        body_layout.addLayout(right_layout)
        main_layout.addWidget(body)

        # ── Progress bar ──────────────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(3)
        self.progress.hide()
        main_layout.addWidget(self.progress)

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Configure API key in Settings to get started")

        # Welcome message
        self._show_welcome()

    def _show_welcome(self):
        welcome = """<div style="text-align: center; padding: 40px 20px; color: #666;">
            <div style="font-size: 48px;">🦚</div>
            <div style="font-size: 22px; font-weight: 700; color: #6c63ff; margin: 12px 0;">PeacockAgent</div>
            <div style="font-size: 13px; color: #888; line-height: 1.8; max-width: 500px; margin: 0 auto;">
                Your AI Desktop Assistant<br>
                Can open apps · control mouse & keyboard · create files<br>
                generate images · search the web · and much more<br><br>
                <span style="color: #6c63ff;">⚙️ Go to Settings to add your Groq API key first</span>
            </div>
        </div>"""
        self.chat_display.insertHtml(welcome)

    def eventFilter(self, obj, event):
        if obj == self.input_box and isinstance(event, QKeyEvent):
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self._send_message()
                return True
        return super().eventFilter(obj, event)

    def _send_message(self):
        text = self.input_box.toPlainText().strip()
        if not text:
            return
        if self.worker and self.worker.isRunning():
            return

        self.input_box.clear()
        self.chat_display.add_message("user", text)

        # Add to history
        if self.uploaded_file:
            msg_content = [
                {"type": "text", "text": text},
            ]
            self.history.append({"role": "user", "content": text + f"\n[Attached file: {self.uploaded_file}]"})
        else:
            self.history.append({"role": "user", "content": text})

        self._remove_file()
        self._start_agent()

    def _start_agent(self):
        self.send_btn.setEnabled(False)
        self.stop_btn.show()
        self.progress.show()
        self.status_indicator.setText("● Thinking")
        self.status_indicator.setStyleSheet("color: #ffa502; font-size: 11px;")

        self.worker = AgentWorker(self.llm, self.history)
        self.worker.message_ready.connect(self._on_message)
        self.worker.tool_called.connect(self._on_tool)
        self.worker.thinking.connect(self._on_thinking)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_message(self, role: str, content: str):
        self.chat_display.add_message(role, content)
        self.history.append({"role": role, "content": content})

    def _on_tool(self, tool_name: str, result: str):
        self.chat_display.add_message("tool", f"{tool_name}|||{result}")
        self.status_bar.showMessage(f"Tool: {tool_name} — {result[:60]}...")

    def _on_thinking(self, msg: str):
        self.chat_display.add_message("thinking", msg)
        self.status_bar.showMessage(msg)

    def _on_done(self):
        self.send_btn.setEnabled(True)
        self.stop_btn.hide()
        self.progress.hide()
        self.status_indicator.setText("● Online")
        self.status_indicator.setStyleSheet("color: #2ed573; font-size: 11px;")
        self.status_bar.showMessage("Ready")

    def _on_error(self, error: str):
        self.chat_display.add_message("error", error)
        self._on_done()

    def _stop_agent(self):
        if self.worker:
            self.worker.stop()
            self.worker.quit()
        self._on_done()
        self.chat_display.add_message("thinking", "⏹ Stopped by user")

    def _quick_action(self, prompt: str):
        self.input_box.setText(prompt)
        self.input_box.setFocus()

    def _upload_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", str(Path.home()),
            "All Files (*);;Images (*.png *.jpg *.jpeg);;Documents (*.pdf *.docx *.txt)"
        )
        if path:
            self.uploaded_file = path
            self.file_label.setText(f"📎 {os.path.basename(path)}")
            self.file_bar.show()

    def _remove_file(self):
        self.uploaded_file = None
        self.file_bar.hide()

    def _open_settings(self):
        dlg = SettingsDialog(self.llm, self)
        if dlg.exec():
            self.status_bar.showMessage("Settings saved ✅")
            self.status_indicator.setText("● Ready")
            self.status_indicator.setStyleSheet("color: #2ed573; font-size: 11px;")

    def _clear_chat(self):
        self.chat_display.clear()
        self.history = []
        self._show_welcome()
        self.status_bar.showMessage("Chat cleared")
