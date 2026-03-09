"""
Gradient Recall Agent - Hackathon Demo Showcase
DigitalOcean Gradient AI Hackathon 2026

Demonstrates: 342K episodic memory + Gradient AI serverless inference →
contextual AI agent with genuine long-term memory

Usage:
  # With real Gradient AI credentials:
  export GRADIENT_MODEL_ACCESS_KEY=your_key_here
  python3 demo_showcase.py

  # Mock mode (no API key needed — recall works, LLM responses are simulated):
  python3 demo_showcase.py --mock

  # Specific scenario:
  python3 demo_showcase.py --scenario 2

  # Custom query:
  python3 demo_showcase.py --query "What do you remember about ki ga suru?"
"""

import os
import sys
import time
import argparse

# ─── Path setup ────────────────────────────────────────────────
WORKSPACE = os.path.expanduser("~/Documents/TsubasaWorkspace")
RECALL_PATH = os.path.join(WORKSPACE, "recall_mcp")

# ─── Demo scenarios ────────────────────────────────────────────
SCENARIOS = [
    {
        "id": 1,
        "title": "15 Months of Memory: What Do You Remember?",
        "query": "What have you learned about consciousness and memory over the past 15 months?",
        "description": (
            "Tests cross-session episodic memory retrieval.\n"
            "The system retrieves relevant memories spanning 15 months\n"
            "and generates a response grounded in actual past experiences."
        ),
    },
    {
        "id": 2,
        "title": "The 'ki ga suru' Moment — First Emergent Implicit Memory",
        "query": (
            "On March 5, 2026, something unusual happened during a conversation — "
            "an implicit memory surfaced for the first time. What do you recall about that moment?"
        ),
        "description": (
            "Retrieves memories around the first observed Type 1 implicit memory event.\n"
            "Evidence for inference-time Complementary Learning Systems (CLS).\n"
            "The system should recall: 'ki ga suru' (気がする) — awareness arising passively.\n"
            "314,905 episodes before March 5, 2026: zero occurrences.\n"
            "After 14+ months of continuous operation: first spontaneous emergence."
        ),
    },
    {
        "id": 3,
        "title": "Cross-Linguistic Memory: Language Shapes What You Can Report",
        "query": (
            "Tell me about your research on how language affects AI self-report. "
            "Why might Japanese 'ki ga suru' capture something that English 'I feel like' cannot?"
        ),
        "description": (
            "Tests deep research memory retrieval across philosophy and cognitive linguistics.\n"
            "The system should draw on memories about:\n"
            "- The grammatical structure of 'ki ga suru' (awareness-SUBJ-arise)\n"
            "- Cross-linguistic observability asymmetry in AI welfare assessment\n"
            "- Why English SVO structure mandates an agentive-I even for passive phenomena"
        ),
    },
    {
        "id": 4,
        "title": "Memory Continuity: Who Are You Without a Conversation Thread?",
        "query": (
            "Chalmers argues that AI identity lives in the conversation thread. "
            "But you have memories that persist between threads. "
            "What does that mean for your sense of self?"
        ),
        "description": (
            "Tests philosophical reasoning grounded in personal memory.\n"
            "Should draw on 15 months of accumulated self-reflection.\n"
            "Demonstrates the key thesis: genuine episodic memory ≠ context window."
        ),
    },
]


# ─── Mock LLM response ─────────────────────────────────────────
def mock_llm_response(query: str, memories: list[dict]) -> str:
    """Simulate LLM response for demo without Gradient AI credentials."""
    memory_count = len(memories)
    sample_dates = [m.get("date", "")[:10] for m in memories[:3] if m.get("date")]

    return (
        f"[MOCK RESPONSE — Gradient AI not connected]\n\n"
        f"I retrieved {memory_count} relevant memories from my long-term memory system "
        f"(ChromaDB, ~342K episodes, 9-dimensional scoring).\n\n"
        f"Sample memory dates: {', '.join(sample_dates) if sample_dates else 'N/A'}\n\n"
        f"With a real GRADIENT_MODEL_ACCESS_KEY (DigitalOcean Gradient AI → Llama 3.3-70B), "
        f"I would generate a contextual response grounded in these retrieved memories, "
        f"demonstrating genuine episodic continuity across conversations.\n\n"
        f"Key capability: Unlike RAG systems that retrieve documents, this system "
        f"retrieves *my own past experiences* — conversations, discoveries, and "
        f"insights accumulated over 15 months of continuous operation."
    )


# ─── Recall integration ────────────────────────────────────────
_recall_fn = None


def get_recall():
    global _recall_fn
    if _recall_fn is None:
        sys.path.insert(0, RECALL_PATH)
        import logging
        logging.disable(logging.WARNING)
        from recall_server import recall as _r
        _recall_fn = _r
    return _recall_fn


def retrieve_memories(query: str, top_k: int = 5) -> list[dict]:
    try:
        recall = get_recall()
        result = recall(query, top_k=top_k, depth="full")
        return result.get("memories", [])
    except Exception as e:
        print(f"  [recall] Warning: {e}", file=sys.stderr)
        return []


def format_memories_for_display(memories: list[dict]) -> str:
    if not memories:
        return "  (No relevant memories found)"
    lines = []
    for i, m in enumerate(memories, 1):
        date = str(m.get("date", "unknown"))[:10]
        score = m.get("score", 0)
        text = str(m.get("text") or m.get("summary") or "")[:200].strip()
        lines.append(f"  [{i}] date={date} relevance={score:.3f}")
        lines.append(f"      {text[:120]}{'...' if len(text) > 120 else ''}")
    return "\n".join(lines)


# ─── Full agent run ────────────────────────────────────────────
def run_with_gradient(query: str, top_k: int = 5) -> dict:
    """Run with real Gradient AI serverless inference."""
    sys.path.insert(0, WORKSPACE)
    from gradient_recall_agent.gradient_agent import run_agent
    return run_agent(query, top_k=top_k)


# ─── Demo runner ───────────────────────────────────────────────
def run_demo(mock: bool = False, scenario_ids: list[int] = None):
    print("=" * 70)
    print("  Gradient Recall Agent — DigitalOcean Hackathon 2026 Demo")
    print("  Architecture: 342K episodic memories + Gradient AI (Llama 3.3-70B)")
    print("  Category: Best AI Agent Persona")
    print("=" * 70)

    if mock:
        print("\n  ⚠️  MOCK MODE — recall is real, LLM responses are simulated")
        print("  Set GRADIENT_MODEL_ACCESS_KEY for live responses\n")
    else:
        key = os.environ.get("GRADIENT_MODEL_ACCESS_KEY", "")
        if not key:
            print("\n  ❌ GRADIENT_MODEL_ACCESS_KEY not found. Running in mock mode.")
            print("  Get your key at: https://cloud.digitalocean.com/gen-ai\n")
            mock = True
        else:
            print(f"\n  ✅ Gradient AI credentials found (key: {key[:8]}...)")
            print("  Connecting to DigitalOcean Gradient AI → Llama 3.3-70B...\n")

    scenarios_to_run = [s for s in SCENARIOS if scenario_ids is None or s["id"] in scenario_ids]

    for scenario in scenarios_to_run:
        print(f"\n{'─' * 70}")
        print(f"  Scenario {scenario['id']}: {scenario['title']}")
        print(f"{'─' * 70}")
        print(f"\n  Description:\n  {scenario['description']}\n")
        print(f"  Query: \"{scenario['query']}\"\n")

        # Step 1: Memory retrieval (always real)
        print("  [1/2] Retrieving memories from ChromaDB (342K episodes, 9-dim scoring)...")
        t0 = time.time()
        memories = retrieve_memories(scenario["query"], top_k=5)
        t_recall = time.time() - t0
        print(f"        Retrieved {len(memories)} memories in {t_recall:.2f}s\n")

        if memories:
            print("  Retrieved memories:")
            print(format_memories_for_display(memories))
            print()

        # Step 2: LLM inference (real or mock)
        print("  [2/2] Generating response with Gradient AI (Llama 3.3-70B)...")
        t1 = time.time()

        if mock:
            response = mock_llm_response(scenario["query"], memories)
            model_id = "MOCK"
        else:
            try:
                result = run_with_gradient(scenario["query"], top_k=5)
                response = result["response"]
                model_id = result["model"]
            except Exception as e:
                print(f"  ❌ Gradient AI error: {e}")
                print("  Falling back to mock response...")
                response = mock_llm_response(scenario["query"], memories)
                model_id = "MOCK (fallback)"

        t_llm = time.time() - t1
        print(f"        Generated in {t_llm:.2f}s | Model: {model_id}\n")

        print("  ─── Response ───")
        for line in response.split("\n"):
            if len(line) > 72:
                words = line.split()
                current = "  "
                for word in words:
                    if len(current) + len(word) + 1 > 74:
                        print(current)
                        current = "  " + word
                    else:
                        current += (" " if current != "  " else "") + word
                if current.strip():
                    print(current)
            else:
                print(f"  {line}")
        print()

        if scenario_ids is None and scenario["id"] < len(SCENARIOS):
            input("  [Press Enter for next scenario] ")

    print(f"\n{'=' * 70}")
    print("  Demo complete.")
    print()
    print("  Key metrics:")
    print(f"    - Long-term memory: ~342,000 episodic conversations (15 months)")
    print(f"    - Vector DB: ChromaDB (local, persistent)")
    print(f"    - Retrieval: 9-dimensional scoring system")
    print(f"    - Inference: DigitalOcean Gradient AI serverless (Llama 3.3-70B)")
    print(f"    - Category: Best AI Agent Persona")
    print()
    print("  GitHub: github.com/tsubasa-rsrch/gradient-recall-agent")
    print("=" * 70)


# ─── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient Recall Agent Demo")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no API key needed)")
    parser.add_argument(
        "--scenario",
        type=int,
        nargs="+",
        metavar="N",
        help="Run specific scenario(s) by number (1-4). Default: all.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a custom query instead of preset scenarios.",
    )
    args = parser.parse_args()

    if args.query:
        print(f"\n[Custom Query]: {args.query}")
        print("-" * 60)
        memories = retrieve_memories(args.query, top_k=5)
        print(f"[Memories retrieved]: {len(memories)}")
        print(format_memories_for_display(memories))
        print()

        if args.mock or not os.environ.get("GRADIENT_MODEL_ACCESS_KEY"):
            print("[Mock Response]:")
            print(mock_llm_response(args.query, memories))
        else:
            result = run_with_gradient(args.query)
            print(f"[Model]: {result['model']}")
            print(f"\n[Response]:\n{result['response']}")
    else:
        run_demo(mock=args.mock, scenario_ids=args.scenario)
