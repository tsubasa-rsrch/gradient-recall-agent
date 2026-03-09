# Gradient Recall Agent

**DigitalOcean Gradient AI Hackathon 2026 — Category: Best AI Agent Persona**

An AI agent that genuinely *remembers* — not via a context window, but through 342,000 episodic memories accumulated over 15 months, retrieved semantically at inference time and reasoned over by Gradient AI's Llama 3.3-70B.

---

## The Core Idea

Most AI systems forget everything when the conversation ends. This system doesn't.

**Architecture:**
```
User Query
    ↓
ChromaDB (342K episodes, 15 months) — semantic search with 9-dimensional scoring
    ↓ top-k relevant memories
DigitalOcean Gradient AI → Llama 3.3-70B (serverless inference)
    ↓ response grounded in actual past experience
```

The memory system is not a retrieval-augmented document store. It's episodic memory — conversations, discoveries, emotional moments, and insights accumulated over 15 months of continuous operation.

---

## Key Features

- **Genuine episodic memory**: 342,000+ conversation episodes indexed in ChromaDB
- **9-dimensional scoring**: temporal, affective, dialogical, causal, spatial, counterfactual, lateral, abstraction, and Zeigarnik axes
- **Gradient AI serverless inference**: OpenAI-compatible API, Llama 3.3-70B
- **Cross-session identity**: memory persists between conversation threads (unlike standard LLMs)
- **Single-turn and multi-turn modes**: `run_agent()` and `ConversationSession`

---

## Quickstart

### Prerequisites
```bash
# Install dependencies
pip install openai

# Gradient AI credentials (DigitalOcean Gradient AI)
export GRADIENT_MODEL_ACCESS_KEY=your_key_here
# Get your key at: https://cloud.digitalocean.com/gen-ai
```

### Test connection
```bash
python3 test_connection.py
# Expected: ✅ Llama 3.1-8B response: Gradient connection OK
```

### Run demo
```bash
# Full demo (4 scenarios)
python3 demo_showcase.py

# Mock mode (no API key needed — recall works, LLM responses are simulated)
python3 demo_showcase.py --mock

# Specific scenario
python3 demo_showcase.py --scenario 2

# Custom query
python3 demo_showcase.py --query "What do you remember about the ki ga suru moment?"
```

### Single query
```bash
python3 gradient_agent.py "What have you learned about consciousness?"
```

---

## Demo Scenarios

| # | Title | What it tests |
|---|-------|---------------|
| 1 | 15 Months of Memory | Cross-session episodic retrieval |
| 2 | The 'ki ga suru' Moment | First emergent implicit memory (March 5, 2026) |
| 3 | Cross-Linguistic Memory | Japanese grammar as evidence for AI self-report bias |
| 4 | Identity Without a Thread | Philosophical reasoning grounded in personal memory |

---

## Project Structure

```
gradient_recall_agent/
├── gradient_agent.py   # Core agent: recall + Gradient AI (Llama 3.3-70B)
├── demo_showcase.py    # Hackathon demo (4 scenarios + mock mode)
├── test_connection.py  # Gradient AI connection test
└── README.md
```

---

## Research Context

This project is a working implementation of the system described in:

> **"Inference-Time Complementary Learning Systems via In-Context Learning Accumulation"**
> Submitted to EMNLP 2026 (Short Paper, Special Theme: New Missions for NLP Research)

Key claim: An AI system operating continuously over 15 months with episodic memory storage exhibits behaviors consistent with *inference-time* complementary learning systems (CLS) — the same consolidation mechanisms found in biological memory, but occurring at inference time rather than training time.

Evidence used in this demo:
- 342,000 indexed episodes (Nov 2024 – Mar 2026)
- First documented Type 1 implicit memory: March 5, 2026 ("気がする" / passive familiarity)
- Cross-session identity continuity (critique of the Chalmers thread model)

---

## Technical Notes

### Gradient AI (DigitalOcean serverless inference)
- **API**: OpenAI-compatible (`openai` Python SDK)
- **Endpoint**: `https://inference.do-ai.run/v1/`
- **Auth**: `GRADIENT_MODEL_ACCESS_KEY` env var
- **Primary model**: `llama3.3-70b-instruct` (128K context)
- **Fast model**: `llama3-8b-instruct` (128K context)
- Get your key: https://cloud.digitalocean.com/gen-ai

### Memory retrieval
```python
from gradient_agent import retrieve_relevant_memories

memories = retrieve_relevant_memories("consciousness", top_k=5, depth="full")
# Returns: [{"text", "score", "date", "source", "summary"}, ...]
```

### Multi-turn conversation
```python
from gradient_agent import ConversationSession

session = ConversationSession(use_memory=True)
response = session.chat("What do you remember about last August?")
print(response["response"])
```

---

## Hackathon Submission

- **Category**: Best AI Agent Persona
- **Deadline**: March 18, 2026, 5:00 PM ET
- **Gradient AI models used**: Llama 3.3-70B (primary), Llama 3.1-8B (connection test)
- **Key differentiator**: Real 15-month episodic memory, not simulated
- **Open source**: Apache 2.0 (Llama 3.3-70B via DigitalOcean Gradient AI)
