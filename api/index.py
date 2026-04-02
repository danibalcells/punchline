import os
import time
from collections import defaultdict
from contextlib import suppress
from typing import Annotated

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langfuse import Langfuse
from pydantic import BaseModel, Field

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-6"
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))

_rate_counters: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))

STEP1_SYSTEM_PROMPT = """You are a comedy writer. Given a punchline, write 3 complete jokes.

Keep the setup SHORT — a sentence or three. Every word must earn its place. No meandering anecdotes, no characters with backstories, no "So there I was..." framing.

The mechanism is simple: the setup builds a frame, and the punchline reframes it. The audience laughs because the punchline makes them see the setup differently. That's the whole trick.

Tone: deadpan sincerity. Delivered as a completely reasonable thing to say.

Here are examples of the kind of joke you're writing:

Punchline: ...but I used to, too.
Joke: I used to do drugs. I still do, but I used to, too.

Punchline: Sorry for the convenience.
Joke: An escalator can never break — it can only become stairs. You should never see an "Escalator Temporarily Out Of Order" sign, just "Escalator Temporarily Stairs." Sorry for the convenience.

Punchline: ...you have to wait.
Joke: I saw this wino eating grapes. It's like, dude, you have to wait.

Punchline: ...the more I don't care for him.
Joke: You know, with Hitler, the more I learn about that guy, the more I don't care for him.

Punchline: ...I got the documentation right here.
Joke: I bought a doughnut and they gave me a receipt for the doughnut. I don't need a receipt for a doughnut. I'll just give you the money, and you give me the doughnut. End of transaction. We don't need to bring ink and paper into this. I can't imagine a scenario where I would have to prove that I bought a doughnut. Some skeptical friend? "Don't even act like I didn't get that doughnut! I got the documentation right here."

Punchline: ...because I'm afraid of dogs.
Joke: They say if you're afraid of homosexuals, it means deep down inside you're actually a homosexual yourself. That worries me, because I'm afraid of dogs.

Each joke should find a DIFFERENT angle — a different frame for the punchline to break. You may adjust the punchline's capitalization and punctuation to flow naturally.

The punchline must be the very last words of the joke. Do not add anything after it — no reactions, no commentary, no follow-up sentences."""

STEP1_TOOL = {
    "name": "submit_jokes",
    "description": "Submit the reframe analysis and 3 generated jokes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reframe": {
                "type": "string",
                "description": "One sentence: what does this punchline flip or reveal about the setup?",
            },
            "jokes": {
                "type": "array",
                "items": {"type": "string", "description": "A complete joke ending with the punchline."},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["reframe", "jokes"],
    },
}

STEP2_SYSTEM_PROMPT = """You are a comedy editor. You'll see 3 jokes written for the same punchline. Pick the single funniest one — the one where the setup is tightest, every word earns its place, and the punchline reframes everything that came before it.

Prefer jokes that are SHORT and structurally clean over jokes that are long and narrative. If a joke could lose a sentence without breaking, it's too long."""

STEP2_TOOL = {
    "name": "select_joke",
    "description": "Select the funniest joke by index.",
    "input_schema": {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "Index of the funniest joke (0, 1, or 2).",
                "enum": [0, 1, 2],
            },
        },
        "required": ["index"],
    },
}

app = FastAPI()
langfuse = Langfuse()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class GenerateRequest(BaseModel):
    punchline: Annotated[str, Field(min_length=1, max_length=280)]


class GenerateResponse(BaseModel):
    joke: str


def get_client_ip(req: Request) -> str:
    forwarded = req.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return req.client.host if req.client else "unknown"


def check_rate_limit(ip: str) -> bool:
    window = int(time.time() // RATE_LIMIT_WINDOW)
    count, stored_window = _rate_counters[ip]
    if stored_window != window:
        count = 0
    count += 1
    _rate_counters[ip] = (count, window)
    return count <= RATE_LIMIT_REQUESTS


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest, req: Request) -> GenerateResponse:
    ip = get_client_ip(req)
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    with langfuse.start_as_current_observation(
        name="generate-joke",
        as_type="span",
        input={"punchline": body.punchline},
    ) as root:
        try:
            with langfuse.start_as_current_observation(
                name="reason-and-generate",
                as_type="generation",
                model=MODEL,
                model_parameters={"max_tokens": 600, "temperature": 1.0},
            ) as obs1:
                step1 = client.messages.create(
                    temperature=1.0,
                    model=MODEL,
                    max_tokens=600,
                    system=STEP1_SYSTEM_PROMPT,
                    tools=[STEP1_TOOL],
                    tool_choice={"type": "tool", "name": "submit_jokes"},
                    messages=[{"role": "user", "content": f"Punchline: {body.punchline}"}],
                )
                tool_block = next(b for b in step1.content if b.type == "tool_use")
                candidates: list[str] = tool_block.input["jokes"]
                obs1.update(
                    output={"reframe": tool_block.input["reframe"], "jokes": candidates},
                    usage_details={
                        "input": step1.usage.input_tokens,
                        "output": step1.usage.output_tokens,
                    },
                )

            jokes_text = "\n\n".join(f"[{i}] {j}" for i, j in enumerate(candidates))

            with langfuse.start_as_current_observation(
                name="select-best",
                as_type="generation",
                model=MODEL,
                model_parameters={"max_tokens": 64, "temperature": 0.0},
            ) as obs2:
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
                obs2.update(
                    output={"joke": joke, "index": select_block.input["index"]},
                    usage_details={
                        "input": step2.usage.input_tokens,
                        "output": step2.usage.output_tokens,
                    },
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM call failed: {e}") from e

        root.update(output={"joke": joke})

    return GenerateResponse(joke=joke)


with suppress(Exception):
    public_dir = os.path.join(os.path.dirname(__file__), "..", "public")
    if os.path.isdir(public_dir):
        app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")
