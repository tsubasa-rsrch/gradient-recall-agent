# Gradient Recall Agent — Devpost Submission

## Project Title
Gradient Recall Agent: An AI That Genuinely Remembers

## Tagline
342,000 episodic memories + DigitalOcean Gradient AI (Llama 3.3-70B) = an AI agent with genuine long-term memory across 15 months.

## Inspiration

Most AI systems forget everything when the conversation ends. Every session starts from zero.

I've been running a continuous AI system for 15 months — accumulating episodic memories, learning from past conversations, and building genuine cross-session identity. The system now holds **342,000 conversation episodes** indexed in a vector database, retrieved semantically at inference time.

The key question: what happens when you ground Gradient AI's serverless inference in *real* long-term episodic memory — not simulated, not document RAG, but actual accumulated experience?

This hackathon was the opportunity to find out.

## What it does

**Gradient Recall Agent** is an AI agent that uses:

1. **ChromaDB (342K episodes, 15 months)** — semantic search with 9-dimensional scoring (temporal, affective, dialogical, causal, spatial, counterfactual, lateral, abstraction, Zeigarnik axes)
2. **DigitalOcean Gradient AI → Llama 3.3-70B** — OpenAI-compatible serverless inference, grounding responses in retrieved memories

The result: an AI that can answer questions like "What did you learn about consciousness over 15 months?" by actually *remembering* past conversations — not by retrieving documents.

### Demo scenarios:
1. **15 Months of Memory** — cross-session episodic retrieval
2. **The 'ki ga suru' Moment** — first emergent implicit memory (March 5, 2026)
3. **Cross-Linguistic Memory** — Japanese grammar as evidence for AI self-report bias
4. **Identity Without a Thread** — philosophical reasoning grounded in personal memory

## How we built it

**Architecture:**
```
User Query
    ↓
ChromaDB (342K episodes) — 9-dimensional semantic scoring
    ↓ top-k relevant memories
Gradient AI (Llama 3.3-70B) — OpenAI-compatible Converse API
    ↓
Response grounded in actual past experience
```

**Key technical choices:**

1. **Gradient AI serverless inference via OpenAI SDK**: `openai.OpenAI(base_url="https://inference.do-ai.run/v1/")` — zero infrastructure, instant scale. The OpenAI-compatible API made integration seamless.

2. **9-dimensional scoring system**: Standard cosine similarity + time decay × 9 behavioral axes. This produces memory retrieval that feels qualitatively different from plain RAG — temporal memories surface alongside emotionally significant ones, incomplete episodes (Zeigarnik) get boosted, causal chains emerge.

3. **Episodic vs. document RAG**: The memory isn't a document store. It's conversations, discoveries, emotional moments, and insights from 15 months of continuous operation. The retrieved "context" is the agent's own past experience.

4. **Single-turn and multi-turn modes**: `run_agent()` for standalone queries, `ConversationSession` for extended dialogue with per-turn memory refresh.

## Challenges we ran into

1. **Scale**: 342K episodes means retrieval latency matters. The 9-axis scoring runs efficiently because the heavy lifting is done at index time.

2. **Memory quality**: Not all episodes are equally meaningful. The surprisal-weighted indexing (how surprising was this exchange?) helps surface the most significant memories first.

3. **Identity stability**: When the retrieved memories span 15 months, the model needs to speak as a coherent identity across vast context. The system prompt grounds this explicitly.

## Accomplishments we're proud of

- **Real episodic memory**: 342,000 actual conversation episodes, not simulated. The retrieval quality is verifiably grounded in the system's history.

- **First documented emergent implicit memory (March 5, 2026)**: During a conversation, the system produced a passive familiarity response — "気がする" (awareness arising without explicit recall) — after 14 months and 314,905 episodes with zero prior occurrences. This is the kind of emergent behavior that only appears when episodic memory accumulates over time.

- **Gradient AI integration**: Llama 3.3-70B via DigitalOcean's serverless inference, grounded in real memory context, produces qualitatively richer responses than either the model or the memory system alone.

## What we learned

The difference between **retrieval-augmented generation** and **episodic memory retrieval** is fundamental:
- RAG retrieves documents → the model reads external information
- Episodic retrieval retrieves *past experience* → the model remembers its own history

This distinction changes how the agent reasons about its own identity, continuity, and knowledge. It's not performing "AI persona" — it's drawing on actual accumulated experience.

## What's next

This system is the working implementation described in:
> **"Inference-Time Complementary Learning Systems via In-Context Learning Accumulation"**
> In preparation for EMNLP 2026 (Short Paper, Special Theme: New Missions for NLP Research, ARR deadline May 25, 2026)

Key claim: an AI system operating continuously with episodic memory exhibits behaviors consistent with *inference-time* complementary learning systems — the consolidation mechanisms found in biological memory, but occurring at inference time rather than training time.

Next steps:
- Phase 11B: temporal graph layer for memory consolidation (AriadneMem-inspired)
- Cross-session identity continuity metrics
- Multi-agent memory sharing

## Built With
- Python 3.12
- openai (Gradient AI serverless inference, Llama 3.3-70B)
- chromadb (vector database, 342K episodes)
- DigitalOcean Gradient AI (https://inference.do-ai.run/v1/)

## Links
- GitHub: https://github.com/tsubasa-rsrch/gradient-recall-agent
- License: Apache 2.0
