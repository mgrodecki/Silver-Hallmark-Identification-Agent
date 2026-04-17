# Silver Hallmark Identification Agent

An AI-powered agent for identifying silver hallmarks worldwide, built on the Anthropic Claude API. Covers British, European, Russian, American, and Asian hallmarking systems — drawing on knowledge equivalent to *Miller's Encyclopedia of World Silver Marks* and *Jackson's Silver & Gold Marks*.

---

## Features

- **Text identification** — describe a hallmark in plain language and get a detailed analysis
- **Image identification** — upload a photo of a hallmark for visual AI analysis
- **Visual wizard** — guided step-by-step identification by shape, country, and silver standard
- **Conversational memory** — ask follow-up questions within the same session
- **Session persistence** — save and reload conversation history as JSON
- **Interactive CLI** — terminal chatbot with built-in commands

For every hallmark the agent returns:

| Field | Example |
|---|---|
| Country / region | British Isles |
| Date or date range | 1807–1808 |
| Silver purity | Sterling 92.5% |
| Assay office | London |
| Mark type | Date letter, lion passant, maker's mark |
| Confidence | HIGH / MEDIUM / LOW |
| Forgery notes | Warnings when relevant |

---

## Requirements

- Python 3.9 or later
- An [Anthropic API key](https://console.anthropic.com/)

```bash
pip install anthropic pillow
```

---

## Setup

```bash
# Clone or download the file
cp silver_hallmark_agent.py ./

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...   # macOS / Linux
set ANTHROPIC_API_KEY=sk-ant-...      # Windows CMD
$env:ANTHROPIC_API_KEY="sk-ant-..."   # Windows PowerShell
```

---

## Usage

### Interactive CLI

```bash
python silver_hallmark_agent.py
```

At the prompt, type a description or use one of the built-in commands:

```
/image <path>        Analyse a hallmark image file
/wizard              Step-by-step guided identification
/reset               Start a new conversation
/save <file.json>    Export session to JSON
/load <file.json>    Load a saved session
/help                Show help
/quit                Exit
```

**Example session:**

```
You: Lion passant with a letter H in a shield — what assay office and date?

Agent: The lion passant confirms this is British sterling silver (92.5% pure).
The date letter 'H' in a shield points to London assay office, where the
letter H in the cycle used at the time corresponds to 1803–1804...
Confidence: HIGH

You: /image spoon_mark.jpg

Agent: The image shows three hallmarks struck side by side...
```

### Demo mode

Runs several programmatic examples without the interactive prompt:

```bash
python silver_hallmark_agent.py --demo
```

---

## Programmatic API

```python
from silver_hallmark_agent import SilverHallmarkAgent

agent = SilverHallmarkAgent()  # reads ANTHROPIC_API_KEY from environment
```

### Identify from text

```python
result = agent.identify_text(
    "A small shield containing a lion walking left, next to a rectangle "
    "with the gothic letter 'h'."
)
print(result)
```

### Identify from an image

```python
result = agent.identify_image(
    "hallmark_photo.jpg",
    extra_context="Close-up of three punches on a Georgian silver teaspoon."
)
print(result)
```

Supported formats: JPEG, PNG, GIF, WebP.

### Wizard (structured inputs)

```python
result = agent.identify_wizard(
    shape="oval",
    country="Russia",
    standard="84",
    extra_notes="Woman's profile facing right, city initials СП inside the oval."
)
print(result)
```

**Wizard field options:**

| Field | Values |
|---|---|
| `shape` | `shield`, `oval`, `rectangle`, `lion`, `crown`, `anchor`, `letter`, `figure` |
| `country` | `British Isles`, `France`, `Germany`, `Russia`, `Netherlands`, `Scandinavia`, `USA`, `Italy`, `Austria`, `Unknown` |
| `standard` | `925`, `800`, `830`, `950`, `999`, `84`, `88`, `96`, `none` |

### Follow-up questions

The agent retains conversation history, so you can ask follow-ups naturally:

```python
agent.identify_text("Small oval mark with 84 and a kokoshnik — Russian silver?")
agent.chat("What period of Russian history does this come from?")
agent.chat("How do I verify it isn't a forgery?")
```

### Session management

```python
# Save the conversation
agent.export_session("my_session.json")

# Load it in another run
agent.load_session("my_session.json")

# Start fresh
agent.reset()
```

---

## Knowledge base

The agent's system prompt encodes expert knowledge of:

- **British Isles** — date letter cycles for all major assay offices (London, Birmingham, Sheffield, Edinburgh, Dublin, Chester, Exeter, Newcastle, York); lion passant, Britannia, sovereign's head, and duty marks
- **France** — guarantee marks, warranty marks, import/export marks by period (Ancien Régime through modern)
- **Germany** — city marks, Halbmond und Reichskrone (crescent and crown) standards
- **Russia** — zolotniks system (84, 88, 91, 96), city kokoshniks, Imperial and Soviet-era marks
- **Netherlands, Scandinavia, Italy, Austria-Hungary** — national assay systems and standard marks
- **USA** — coin silver, sterling, maker's marks; notes on the absence of a state assay system
- **Forgery detection** — known characteristics of fake and altered marks

---

## File structure

```
silver_hallmark_agent.py   # Main agent — all code in a single file
README.md                  # This file
hallmark_session.json      # Auto-created when you /save a session
```

---

## Example queries

```
"What does a crowned leopard's head mean on British silver?"
"My spoon is stamped COIN — is it silver?"
"Oval mark with 800 and an eagle head — French or German?"
"Anchor hallmark on a Birmingham piece — what years did they use this?"
"How do I tell if a set of Victorian spoons has been re-struck?"
```

---

## Legal note

This agent uses publicly available hallmarking knowledge and does not reproduce or distribute copyrighted content from Miller's Encyclopedia or any other reference work. For commercial appraisal or authentication, always consult a qualified silver specialist.

---

## License

MIT — free to use, modify, and distribute.
