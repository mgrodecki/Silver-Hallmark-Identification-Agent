"""
Silver Hallmark Identification AI Agent
========================================
Powered by OpenAI API or Anthropic API.
Supports text descriptions and image uploads (base64).

Usage:
    python silver_hallmark_agent.py
    python silver_hallmark_agent.py --provider anthropic --model claude-3-7-sonnet-latest
    python silver_hallmark_agent.py --gui

Requirements:
    pip install openai anthropic pillow
"""

import os
import sys
import base64
import json
import argparse
import threading
from pathlib import Path
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  System prompt â€” hallmark expert knowledge
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SYSTEM_PROMPT = """You are an expert silver hallmark identification agent with deep knowledge
equivalent to Miller's Encyclopedia of World Silver Marks and Jackson's Silver & Gold Marks.

You specialise in:
- British hallmarks: date letters by assay office (London, Birmingham, Sheffield, Edinburgh,
  Dublin, Chester, Exeter, Newcastle, York), lion passant, Britannia, sovereign's head, maker's marks
- European hallmarks: French guarantee and warranty marks, German city marks, Dutch assay marks,
  Scandinavian marks, Italian marks, Austrian and Austro-Hungarian marks
- Russian hallmarks: zolotniks system (84, 88, 91, 96), city kokoshniks, Soviet marks
- American marks: coin silver, sterling, maker's marks, no official assay system notes
- Asian and other international marks

When identifying a hallmark, always provide:
1. Most likely country / region of origin
2. Approximate date or date range
3. Silver purity / standard (e.g. sterling 92.5%, 800/1000, etc.)
4. Assay office if determinable
5. Mark type (date letter, maker's mark, duty mark, standard mark, import mark, etc.)
6. Confidence level: HIGH / MEDIUM / LOW
7. Any important notes or forgery warnings

Be precise and scholarly but approachable. If uncertain, explain what additional
information would help narrow the identification."""


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Hallmark Agent class
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class SilverHallmarkAgent:
    """Conversational AI agent for identifying silver hallmarks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "auto",
    ):
        provider = self._resolve_provider(provider, api_key)
        self.provider = provider

        if provider == "openai":
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY or pass api_key=."
                )
            if OpenAI is None:
                raise ImportError("openai package not found. Install with: pip install openai")
            self.client = OpenAI(api_key=key)
            self.model = model or "gpt-4.1"
        else:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key=."
                )
            if anthropic is None:
                raise ImportError("anthropic package not found. Install with: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=key)
            self.model = model or "claude-3-7-sonnet-latest"

        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _resolve_provider(provider: str, api_key: Optional[str]) -> str:
        normalized = provider.strip().lower()
        if normalized in {"openai", "anthropic"}:
            return normalized
        if normalized != "auto":
            raise ValueError("provider must be one of: auto, openai, anthropic")

        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"

        if api_key:
            if api_key.startswith("sk-ant-"):
                return "anthropic"
            return "openai"

        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, "
            "or pass api_key= with provider='openai'/'anthropic'."
        )

    # â”€â”€ Core API call â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _call(self, user_content: list[dict[str, Any]] | str) -> str:
        """Send a message (with optional image) and return the assistant's reply."""
        self.history.append({"role": "user", "content": user_content})

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.history,
            )
            reply = response.choices[0].message.content or ""
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=self.history,
            )
            reply = response.content[0].text

        self.history.append({"role": "assistant", "content": reply})
        return reply

    # â”€â”€ Public methods â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def identify_text(self, description: str) -> str:
        """Identify a hallmark from a text description."""
        return self._call(description)

    def identify_image(self, image_path: str, extra_context: str = "") -> str:
        """
        Identify a hallmark from an image file.

        Args:
            image_path: Path to a JPEG, PNG, GIF, or WebP image.
            extra_context: Optional text to include alongside the image.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        suffix = path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")

        with open(path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        prompt_text = (
            extra_context
            or "Please identify the silver hallmark(s) in this image. "
               "Provide country of origin, date range, silver purity, assay office, "
               "mark type, and confidence level."
        )

        if self.provider == "openai":
            content = [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}",
                    },
                },
            ]
        else:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt_text},
            ]
        return self._call(content)

    def identify_wizard(
        self,
        shape: Optional[str] = None,
        country: Optional[str] = None,
        standard: Optional[str] = None,
        extra_notes: str = "",
    ) -> str:
        """
        Identify a hallmark using structured wizard inputs.

        Args:
            shape:       Visual shape (e.g. 'shield', 'lion', 'crown', 'anchor', 'letter')
            country:     Country or region hint (e.g. 'British Isles', 'Russia', 'France')
            standard:    Silver standard if visible (e.g. '925', '800', '84')
            extra_notes: Any additional description
        """
        parts = []
        if shape:
            parts.append(f"Mark shape: {shape}")
        if country:
            parts.append(f"Country/region: {country}")
        if standard:
            parts.append(f"Silver standard visible: {standard}")
        if extra_notes:
            parts.append(f"Additional notes: {extra_notes}")

        if not parts:
            raise ValueError("Provide at least one of: shape, country, standard, extra_notes.")

        prompt = (
            "Identify a silver hallmark with these characteristics:\n"
            + "\n".join(f"â€¢ {p}" for p in parts)
            + "\n\nProvide all identification details including country, date range, "
              "purity, assay office, mark type, and confidence level."
        )
        return self._call(prompt)

    def chat(self, message: str) -> str:
        """Send a free-form follow-up message in the ongoing conversation."""
        return self._call(message)

    def reset(self):
        """Clear conversation history to start a fresh session."""
        self.history = []

    def export_session(self, filepath: str):
        """Save the conversation history to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"Session saved to {filepath}")

    def load_session(self, filepath: str):
        """Load a previously saved conversation history."""
        with open(filepath, encoding="utf-8") as f:
            self.history = json.load(f)
        print(f"Session loaded from {filepath} ({len(self.history)} messages)")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Interactive CLI
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

BANNER = """
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘     Silver Hallmark Identification Agent         â•‘
â•‘     Powered by OpenAI / Anthropic                â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

Commands:
  /image <path>          Analyse a hallmark image file
  /wizard                Interactive guided identification
  /reset                 Start a new conversation
  /save <file.json>      Export session to JSON
  /load <file.json>      Load a saved session
  /help                  Show this help
  /quit                  Exit

Otherwise just type a description or question.
"""

HELP_TEXT = """
Examples of text queries:
  â€¢ "Lion passant with a letter H in a shield â€” what assay office?"
  â€¢ "Small oval mark with 84 and a woman's head â€” Russian silver?"
  â€¢ "What does a crowned leopard's head mean on British silver?"
  â€¢ "My spoon has 'COIN' stamped on it â€” is it silver?"

Wizard fields:
  Shape    : shield | oval | rectangle | lion | crown | anchor | letter | figure
  Country  : British Isles | France | Germany | Russia | Netherlands | Scandinavia |
             USA | Italy | Austria | Unknown
  Standard : 925 | 800 | 830 | 950 | 999 | 84 | 88 | 96 | none
"""


def run_wizard(agent: SilverHallmarkAgent):
    """Prompt the user through the visual wizard interactively."""
    print("\nâ”€â”€ Visual Wizard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")

    shape = input("Mark shape (shield/oval/lion/crown/anchor/letter/figure, or Enter to skip): ").strip() or None
    country = input("Country/region (or Enter to skip): ").strip() or None
    standard = input("Silver standard visible (925/800/84/none, or Enter to skip): ").strip() or None
    notes = input("Any extra description (or Enter to skip): ").strip()

    print("\nðŸ” Identifyingâ€¦\n")
    result = agent.identify_wizard(
        shape=shape,
        country=country,
        standard=standard or None,
        extra_notes=notes,
    )
    print(result)
    print()


def interactive_cli(
    provider: str = "auto",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
):
    provider = provider or os.environ.get("HALLMARK_PROVIDER", "auto")
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI or Anthropic API key: ").strip()

    agent = SilverHallmarkAgent(api_key=api_key, provider=provider, model=model)
    print(BANNER)
    print(f"Using provider: {agent.provider} (model: {agent.model})\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # â”€â”€ Commands â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye.")
            break

        elif user_input.lower() in ("/help", "/?"):
            print(HELP_TEXT)

        elif user_input.lower() == "/reset":
            agent.reset()
            print("Conversation reset.\n")

        elif user_input.lower() == "/wizard":
            run_wizard(agent)

        elif user_input.lower().startswith("/image "):
            path = user_input[7:].strip().strip('"').strip("'")
            extra = input("Additional context (or Enter to skip): ").strip()
            print("\nðŸ” Analysing imageâ€¦\n")
            try:
                result = agent.identify_image(path, extra_context=extra or "")
                print(f"Agent: {result}\n")
            except FileNotFoundError as e:
                print(f"Error: {e}\n")

        elif user_input.lower().startswith("/save "):
            path = user_input[6:].strip()
            agent.export_session(path)

        elif user_input.lower().startswith("/load "):
            path = user_input[6:].strip()
            try:
                agent.load_session(path)
            except FileNotFoundError:
                print(f"File not found: {path}\n")

        # â”€â”€ Normal chat â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        else:
            print("\nðŸ” Thinkingâ€¦\n")
            reply = agent.chat(user_input)
            print(f"Agent: {reply}\n")


def launch_gui(
    provider: str = "auto",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    run_demo: bool = False,
):
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
    except ImportError:
        print("Tkinter is not available in this Python environment.")
        return

    root = tk.Tk()
    root.title("Silver Hallmark Identification Agent")
    root.geometry("980x720")

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    top = ttk.Frame(frame)
    top.pack(fill="x")

    provider_var = tk.StringVar(value=provider or "auto")
    model_var = tk.StringVar(value=model or "")
    default_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
    api_key_var = tk.StringVar(value=default_key)
    status_var = tk.StringVar(value="Not connected")
    input_var = tk.StringVar()
    busy_var = tk.BooleanVar(value=False)

    state: dict[str, Any] = {"agent": None}

    ttk.Label(top, text="Provider").grid(row=0, column=0, sticky="w", padx=(0, 6))
    provider_box = ttk.Combobox(
        top, textvariable=provider_var, values=["auto", "openai", "anthropic"], width=12, state="readonly"
    )
    provider_box.grid(row=0, column=1, sticky="w")

    ttk.Label(top, text="Model").grid(row=0, column=2, sticky="w", padx=(12, 6))
    ttk.Entry(top, textvariable=model_var, width=28).grid(row=0, column=3, sticky="we")

    ttk.Label(top, text="API Key").grid(row=0, column=4, sticky="w", padx=(12, 6))
    ttk.Entry(top, textvariable=api_key_var, show="*", width=34).grid(row=0, column=5, sticky="we")

    top.columnconfigure(3, weight=1)
    top.columnconfigure(5, weight=1)

    button_row = ttk.Frame(frame)
    button_row.pack(fill="x", pady=(10, 8))

    output = scrolledtext.ScrolledText(frame, wrap="word", height=28, state="disabled")
    output.pack(fill="both", expand=True, pady=(0, 10))

    input_row = ttk.Frame(frame)
    input_row.pack(fill="x")
    user_input = ttk.Entry(input_row, textvariable=input_var)
    user_input.pack(side="left", fill="x", expand=True)

    def append_output(prefix: str, text: str):
        output.configure(state="normal")
        output.insert("end", f"{prefix}: {text}\n\n")
        output.see("end")
        output.configure(state="disabled")

    def set_busy(is_busy: bool):
        busy_var.set(is_busy)
        send_btn.configure(state="disabled" if is_busy else "normal")
        img_btn.configure(state="disabled" if is_busy else "normal")
        demo_btn.configure(state="disabled" if is_busy else "normal")

    def ensure_agent() -> Optional[SilverHallmarkAgent]:
        if state["agent"] is not None:
            return state["agent"]
        try:
            resolved_model = model_var.get().strip() or None
            key = api_key_var.get().strip() or None
            agent = SilverHallmarkAgent(
                provider=provider_var.get().strip() or "auto",
                model=resolved_model,
                api_key=key,
            )
            state["agent"] = agent
            status_var.set(f"Connected: {agent.provider} / {agent.model}")
            append_output("System", f"Connected using {agent.provider} ({agent.model})")
            return agent
        except Exception as exc:
            messagebox.showerror("Connection Error", str(exc))
            status_var.set("Connection failed")
            return None

    def run_worker(task):
        def wrapped():
            try:
                task()
            finally:
                root.after(0, lambda: set_busy(False))
        set_busy(True)
        threading.Thread(target=wrapped, daemon=True).start()

    def on_connect():
        ensure_agent()

    def on_reset():
        agent = ensure_agent()
        if not agent:
            return
        agent.reset()
        append_output("System", "Conversation reset.")

    def on_send(event=None):
        message = input_var.get().strip()
        if not message:
            return
        input_var.set("")
        append_output("You", message)

        agent = ensure_agent()
        if not agent:
            return

        def task():
            try:
                reply = agent.chat(message)
                root.after(0, lambda: append_output("Agent", reply))
            except Exception as exc:
                root.after(0, lambda: messagebox.showerror("Request Error", str(exc)))

        run_worker(task)

    def on_image():
        path = filedialog.askopenfilename(
            title="Select hallmark image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        extra = simpledialog.askstring("Image Context", "Additional context (optional):", parent=root) or ""
        append_output("You", f"/image {path}")

        agent = ensure_agent()
        if not agent:
            return

        def task():
            try:
                reply = agent.identify_image(path, extra_context=extra)
                root.after(0, lambda: append_output("Agent", reply))
            except Exception as exc:
                root.after(0, lambda: messagebox.showerror("Image Error", str(exc)))

        run_worker(task)

    def on_demo():
        agent = ensure_agent()
        if not agent:
            return
        append_output("System", "Running demo flow...")

        def task():
            try:
                text_result = agent.identify_text(
                    "A small shield-shaped mark containing a lion walking to the left, "
                    "and next to it a rectangle with the letter 'h' in a gothic typeface."
                )
                wizard_result = agent.identify_wizard(
                    shape="oval",
                    country="Russia",
                    standard="84",
                    extra_notes="Woman's profile facing right inside the oval, with city initials SP",
                )
                follow_up = agent.chat("What period of Russian history does this mark come from?")
                root.after(0, lambda: append_output("Demo 1", text_result))
                root.after(0, lambda: append_output("Demo 2", wizard_result))
                root.after(0, lambda: append_output("Demo 3", follow_up))
            except Exception as exc:
                root.after(0, lambda: messagebox.showerror("Demo Error", str(exc)))

        run_worker(task)

    connect_btn = ttk.Button(button_row, text="Connect", command=on_connect)
    connect_btn.pack(side="left")
    reset_btn = ttk.Button(button_row, text="Reset Chat", command=on_reset)
    reset_btn.pack(side="left", padx=(8, 0))
    demo_btn = ttk.Button(button_row, text="Run Demo", command=on_demo)
    demo_btn.pack(side="left", padx=(8, 0))
    img_btn = ttk.Button(button_row, text="Analyze Image", command=on_image)
    img_btn.pack(side="left", padx=(8, 0))
    ttk.Label(button_row, textvariable=status_var).pack(side="right")

    send_btn = ttk.Button(input_row, text="Send", command=on_send)
    send_btn.pack(side="left", padx=(8, 0))
    user_input.bind("<Return>", on_send)

    append_output("System", "Set provider/model/API key, then click Connect.")
    if run_demo:
        root.after(150, on_demo)

    root.mainloop()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Programmatic usage examples
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def demo(provider: str = "auto", model: Optional[str] = None, api_key: Optional[str] = None):
    """
    Example of using SilverHallmarkAgent programmatically.
    Set OPENAI_API_KEY or ANTHROPIC_API_KEY before running.
    """
    agent = SilverHallmarkAgent(provider=provider, model=model, api_key=api_key)

    # 1. Text description
    print("=== Text identification ===")
    result = agent.identify_text(
        "A small shield-shaped mark containing a lion walking to the left, "
        "and next to it a rectangle with the letter 'h' in a gothic typeface."
    )
    print(result)

    # 2. Structured wizard
    print("\n=== Wizard identification ===")
    result = agent.identify_wizard(
        shape="oval",
        country="Russia",
        standard="84",
        extra_notes="Woman's profile facing right inside the oval, with city initials Ð¡ÐŸ",
    )
    print(result)

    # 3. Follow-up question in the same conversation
    print("\n=== Follow-up ===")
    result = agent.chat("What period of Russian history does this mark come from?")
    print(result)

    # 4. Image (uncomment and provide a real path)
    # print("\n=== Image identification ===")
    # result = agent.identify_image("hallmark.jpg")
    # print(result)

    # 5. Save session
    agent.export_session("hallmark_session.json")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Silver Hallmark Identification Agent")
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "anthropic"],
        default="auto",
        help="AI backend provider to use.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name for the selected provider.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override (otherwise uses OPENAI_API_KEY / ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run programmatic demo flow (or auto-run demo inside GUI with --gui).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the desktop GUI.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.gui:
        launch_gui(provider=args.provider, model=args.model, api_key=args.api_key, run_demo=args.demo)
    elif args.demo:
        demo(provider=args.provider, model=args.model, api_key=args.api_key)
    else:
        interactive_cli(provider=args.provider, model=args.model, api_key=args.api_key)

