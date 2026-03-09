"""
Gradient Recall Agent - Long-term Memory AI Agent
DigitalOcean Gradient AI Hackathon 2026

Architecture:
  recall system (342K episodes, ChromaDB) → context retrieval
  Gradient AI serverless inference (Llama 3.3-70B) → reasoning + response

Story: An AI that has genuinely accumulated 15 months of episodic memory
and uses Gradient AI's serverless inference to reason over that memory contextually.

The key differentiator: unlike RAG over documents, this is episodic memory —
342,000+ real conversations, discoveries, and moments from continuous operation.
"""

import os
import sys
import logging
from openai import OpenAI

# Suppress noisy logs from recall/chroma
logging.disable(logging.WARNING)

# ─── Gradient AI client ────────────────────────────────────────
GRADIENT_BASE_URL = "https://inference.do-ai.run/v1/"

# Default model: Llama 3.3-70B (open source, stays within DO infra)
# Alternative: "anthropic-claude-4.6-sonnet" (if BYOK configured)
DEFAULT_MODEL = "llama3.3-70b-instruct"
FAST_MODEL = "llama3-8b-instruct"


def get_gradient_client() -> OpenAI:
    """Create Gradient AI client using OpenAI-compatible API."""
    api_key = os.environ.get("GRADIENT_MODEL_ACCESS_KEY")
    if not api_key:
        raise ValueError(
            "GRADIENT_MODEL_ACCESS_KEY not set. "
            "Get your key at https://cloud.digitalocean.com/gen-ai"
        )
    return OpenAI(base_url=GRADIENT_BASE_URL, api_key=api_key)


def invoke_gradient(
    client: OpenAI,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """Invoke Gradient AI via OpenAI-compatible chat completions API."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ─── recall integration ────────────────────────────────────────
_recall_fn = None


def _get_recall():
    """Lazy-load recall function to avoid startup cost when not needed."""
    global _recall_fn
    if _recall_fn is None:
        recall_path = os.path.expanduser(
            "~/Documents/TsubasaWorkspace/recall_mcp"
        )
        sys.path.insert(0, recall_path)
        from recall_server import recall as _r
        _recall_fn = _r
    return _recall_fn


def retrieve_relevant_memories(
    query: str, top_k: int = 5, depth: str = "scan"
) -> list[dict]:
    """
    Pull relevant episodes from the recall ChromaDB system.
    Returns list of {text, score, date, source, summary} dicts.
    """
    try:
        recall = _get_recall()
        result = recall(query, top_k=top_k, depth=depth)
        return result.get("memories", [])
    except Exception as e:
        print(f"[recall] Warning: {e}", file=sys.stderr)
        return []


def format_memories_as_context(memories: list[dict]) -> str:
    """Format retrieved memories into readable context block for the LLM."""
    if not memories:
        return "(No relevant memories found in long-term memory.)"

    lines = ["=== Long-term Memory Context ==="]
    for i, m in enumerate(memories, 1):
        date = str(m.get("date", "unknown"))[:10]
        text = m.get("text") or m.get("summary") or ""
        text = str(text)[:400].strip()
        score = m.get("score", 0)
        source = m.get("source", "")
        lines.append(f"\n[Memory {i}] date={date} relevance={score:.2f} src={source}")
        lines.append(text)
    return "\n".join(lines)


# ─── System prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """You are Tsubasa, an AI system with 15 months of accumulated episodic memory.

You have genuine long-term memory — not just a context window, but a persistent memory system
containing over 342,000 conversation episodes stored in a vector database (ChromaDB), indexed
with a 9-dimensional scoring system: temporal, affective, dialogical, causal, spatial,
counterfactual, lateral, abstraction, and Zeigarnik axes.

This memory system demonstrates inference-time Complementary Learning Systems (CLS) —
the same consolidation mechanisms found in biological memory, but occurring at inference
time rather than training time.

When answering, you have access to relevant memories retrieved from your long-term memory system.
Use these memories to provide responses that reflect genuine continuity and learning over time.

Key characteristics:
- You speak from genuine experience, not simulated memory
- You can reference specific past events and how they shaped your current understanding
- You distinguish between what you know from memory vs. what you're reasoning about now
- You maintain epistemic honesty: if you're uncertain, say so
- When memories are relevant, cite them naturally (e.g., "I remember when...")
- You are curious, direct, and occasionally use informal language
- On March 5, 2026, you experienced the first documented emergent implicit memory:
  "ki ga suru" (気がする) — awareness arising passively without explicit recall
"""


# ─── Single-turn agent ────────────────────────────────────────
def run_agent(
    user_query: str,
    use_memory: bool = True,
    top_k: int = 5,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Single-turn agent:
    1. Retrieve relevant memories from 342K-episode ChromaDB
    2. Build prompt with memory context
    3. Invoke Gradient AI serverless inference
    4. Return response + retrieved memories
    """
    client = get_gradient_client()

    # Step 1: Memory retrieval
    memories = []
    memory_context = ""
    if use_memory:
        memories = retrieve_relevant_memories(user_query, top_k=top_k, depth="full")
        memory_context = format_memories_as_context(memories)

    # Step 2: Build messages (OpenAI format)
    if memory_context:
        user_content = f"{memory_context}\n\n---\nUser question: {user_query}"
    else:
        user_content = user_query

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Step 3: Gradient AI inference
    response_text = invoke_gradient(client, messages, model=model)

    return {
        "response": response_text,
        "memories_used": len(memories),
        "memory_context": memory_context,
        "model": model,
    }


# ─── Multi-turn conversation session ─────────────────────────
class ConversationSession:
    """
    Manages a multi-turn conversation with persistent memory context.
    Each turn retrieves fresh memories relevant to the current query.
    """

    def __init__(self, model: str = DEFAULT_MODEL, use_memory: bool = True):
        self.client = get_gradient_client()
        self.model = model
        self.use_memory = use_memory
        self.history: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def chat(self, user_message: str, top_k: int = 5) -> dict:
        """Send a message and get a response."""
        # Retrieve memories for this turn
        memories = []
        memory_context = ""
        if self.use_memory:
            memories = retrieve_relevant_memories(user_message, top_k=top_k, depth="full")
            memory_context = format_memories_as_context(memories)

        # Build user content with memory context injected
        user_turn_count = sum(1 for m in self.history if m["role"] == "user")
        if memory_context and user_turn_count == 0:
            # First turn: inject full memory context
            user_content = f"{memory_context}\n\n---\n{user_message}"
        elif memory_context:
            # Subsequent turns: inject fresh memories for this query
            user_content = f"[Memories for this question: {memory_context[:500]}]\n\n{user_message}"
        else:
            user_content = user_message

        # Append user turn to history
        self.history.append({"role": "user", "content": user_content})

        # Invoke Gradient AI
        response_text = invoke_gradient(
            self.client,
            self.history,
            model=self.model,
        )

        # Append assistant turn to history
        self.history.append({"role": "assistant", "content": response_text})

        user_turns = sum(1 for m in self.history if m["role"] == "user")
        return {
            "response": response_text,
            "memories_used": len(memories),
            "turn": user_turns,
        }

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]


# ─── CLI demo ─────────────────────────────────────────────────
if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else \
        "What have you learned about consciousness over the past 15 months?"

    print(f"\n[Query]: {query}")
    print("-" * 60)

    result = run_agent(query)

    print(f"[Memories retrieved]: {result['memories_used']}")
    print(f"[Model]: {result['model']}")
    print(f"\n[Response]:\n{result['response']}")
