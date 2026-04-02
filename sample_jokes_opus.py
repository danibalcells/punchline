import json
import sys
import time

import anthropic
from dotenv import load_dotenv

from api.index import (
    ANTHROPIC_API_KEY,
    STEP1_SYSTEM_PROMPT,
    STEP1_TOOL,
    STEP2_SYSTEM_PROMPT,
    STEP2_TOOL,
)

load_dotenv()

MODEL = "claude-opus-4-6"

PUNCHLINES = [
    "So now I have two mortgages.",
    "And that's why I'm no longer allowed at the aquarium.",
    "Turns out it was just a regular dog.",
    "He said, 'That's not what I meant by cashback.'",
    "I didn't even know you could return a tattoo.",
    "Apparently that's not what 'bring your own bag' means.",
    "The worst part is, the horse was right.",
    "And the landlord said, 'That's not a pet deposit.'",
    "So technically, I'm employee of the month.",
    "They said it was the most polite robbery they'd ever seen.",
    "Now every time I hear a doorbell, I flinch.",
    "Which is weird because I don't even own a boat.",
    "He looked me dead in the eye and said, 'Sir, this is a Wendy's.'",
    "And I still don't know whose cat that was.",
    "That's when I realized the interview had started twenty minutes ago.",
]

OUTPUT_FILE = "sample_jokes_opus_output.json"


def generate_joke(client: anthropic.Anthropic, punchline: str) -> dict:
    step1 = client.messages.create(
        temperature=1.0,
        model=MODEL,
        max_tokens=600,
        system=STEP1_SYSTEM_PROMPT,
        tools=[STEP1_TOOL],
        tool_choice={"type": "tool", "name": "submit_jokes"},
        messages=[{"role": "user", "content": f"Punchline: {punchline}"}],
    )
    tool_block = next(b for b in step1.content if b.type == "tool_use")
    candidates: list[str] = tool_block.input["jokes"]

    jokes_text = "\n\n".join(f"[{i}] {j}" for i, j in enumerate(candidates))

    step2 = client.messages.create(
        temperature=0.0,
        model=MODEL,
        max_tokens=64,
        system=STEP2_SYSTEM_PROMPT,
        tools=[STEP2_TOOL],
        tool_choice={"type": "tool", "name": "select_joke"},
        messages=[{"role": "user", "content": jokes_text}],
    )
    select_block = next(b for b in step2.content if b.type == "tool_use")
    joke = candidates[select_block.input["index"]]

    return {
        "punchline": punchline,
        "candidates": candidates,
        "selected_joke": joke,
    }


def main():
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    results = []

    print(f"Using model: {MODEL}\n")

    for i, punchline in enumerate(PUNCHLINES, 1):
        print(f"[{i}/{len(PUNCHLINES)}] Generating joke for: {punchline}")
        try:
            result = generate_joke(client, punchline)
            results.append(result)
            print(f"  -> Done")
        except Exception as e:
            print(f"  -> ERROR: {e}", file=sys.stderr)
            results.append({"punchline": punchline, "error": str(e)})
        time.sleep(0.5)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print(f"GENERATED JOKES ({MODEL})")
    print("=" * 70)
    for i, r in enumerate(results, 1):
        print(f"\n--- Joke {i} (punchline: {r['punchline']}) ---")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(r["selected_joke"])


if __name__ == "__main__":
    main()
