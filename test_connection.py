"""
Minimal Gradient AI connection test.
Run after setting GRADIENT_MODEL_ACCESS_KEY env var.

Get your key at: https://cloud.digitalocean.com/gen-ai
"""

import os
from openai import OpenAI


def test_gradient_llama():
    """Test Llama 3.1-8B (fast/cheap) with a simple prompt."""
    api_key = os.environ.get("GRADIENT_MODEL_ACCESS_KEY")
    if not api_key:
        print("❌ GRADIENT_MODEL_ACCESS_KEY not set.")
        print("   export GRADIENT_MODEL_ACCESS_KEY=your_key_here")
        print("   Get it at: https://cloud.digitalocean.com/gen-ai")
        return False

    client = OpenAI(
        base_url="https://inference.do-ai.run/v1/",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model="llama3-8b-instruct",
        messages=[
            {"role": "user", "content": "Say 'Gradient connection OK' and nothing else."}
        ],
        max_completion_tokens=20,
        temperature=0.0,
    )

    text = response.choices[0].message.content
    print(f"✅ Llama 3.1-8B response: {text}")
    print(f"✅ Model: {response.model}")
    print(f"✅ Usage: {response.usage.total_tokens} tokens")
    return True


def test_gradient_llama70b():
    """Test Llama 3.3-70B (primary model for demos)."""
    api_key = os.environ.get("GRADIENT_MODEL_ACCESS_KEY")
    if not api_key:
        return False

    client = OpenAI(
        base_url="https://inference.do-ai.run/v1/",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model="llama3.3-70b-instruct",
        messages=[
            {"role": "user", "content": "Say 'Llama 70B OK' and nothing else."}
        ],
        max_completion_tokens=20,
        temperature=0.0,
    )

    text = response.choices[0].message.content
    print(f"✅ Llama 3.3-70B response: {text}")
    return True


if __name__ == "__main__":
    print("Testing Gradient AI serverless inference...")
    print("-" * 50)

    if not os.environ.get("GRADIENT_MODEL_ACCESS_KEY"):
        print("❌ GRADIENT_MODEL_ACCESS_KEY not set.")
        print()
        print("Setup steps:")
        print("  1. Sign up at https://cloud.digitalocean.com/gen-ai")
        print("  2. Create a Model Access Key in the Gradient AI dashboard")
        print("  3. export GRADIENT_MODEL_ACCESS_KEY=your_key_here")
        print("  4. Run: python3 test_connection.py")
    else:
        success = test_gradient_llama()
        if success:
            print()
            test_gradient_llama70b()
