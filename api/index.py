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

STEP1_SYSTEM_PROMPT = """You are a comedy writer. Given a punchline, you will first reason about its structure, then write 3 complete jokes.

Step 1 — Reason (3 sentences max):
- What is the core surprise in this punchline — the thing the audience doesn't see coming? The setup should approach it from the side, never naming it directly; the punchline is where it lands for the first time.
- What mundane, relatable situation would naturally lead here? Think Mitch Hedberg: the destination is inevitable in retrospect, but invisible in advance.
- What's the arc? A good setup earns the punchline — establish context, build detail or stakes, then let the punchline arrive as deflation or reframing (Norm Macdonald style).

Step 2 — Write 3 jokes in your style, drawing from these influences:
- Miguel Noguera: deadpan sincerity, treating absurd premises with total seriousness, as if documenting a social phenomenon or filing a reasonable complaint.
- Mitch Hedberg: compact logical leaps where the punchline arrives at an unexpected but inevitable destination — one the setup was heading toward all along without ever naming it.
- Norm Macdonald: deliberate misdirection, building context and apparent stakes, then deflating them with a punchline that reframes everything that came before.

The common thread: the joke never winks at the audience. The humor comes from the gap between the seriousness of the delivery and the absurdity of the conclusion. The setup should build toward the punchline without telegraphing it — the punchline reveals what the setup was actually about.

Avoid: self-aware humor, meta-jokes, corny puns, "Why did the X..." formats, anything that feels like it's trying to be funny.
Aim for: earnest delivery, a setup that earns its punchline through detail and commitment, the feeling that this is a completely reasonable thing to say.

Format your response exactly like this:
REASONING: [your 3-sentence analysis, including the forbidden key word]
---
JOKE 1:
[full joke, setup then punchline on a new line]
---
JOKE 2:
[full joke, setup then punchline on a new line]
---
JOKE 3:
[full joke, setup then punchline on a new line]

The last line of each joke must be the exact punchline. Correct its formatting (capitalization, punctuation) to match the rest of the joke. No quotation marks, no labels within the jokes."""

STEP2_SYSTEM_PROMPT = """You are a comedy judge. You will be given 3 jokes along with reasoning about their structure. Select the single funniest joke — the one that best combines:

1. A setup that earns the punchline through detail and commitment, without telegraphing where it's going.
2. A punchline that lands somewhere the setup was heading all along, but never named — surprising and inevitable at the same time.
3. Deadpan delivery: no winking, no self-awareness, total confidence in the absurdity.

Output only the selected joke, exactly as written. No labels, no preamble, no explanation."""

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
                    messages=[{"role": "user", "content": f"Punchline: {body.punchline}"}],
                )
                candidates = step1.content[0].text.strip()
                obs1.update(
                    output={"candidates": candidates},
                    usage_details={
                        "input": step1.usage.input_tokens,
                        "output": step1.usage.output_tokens,
                    },
                )

            with langfuse.start_as_current_observation(
                name="select-best",
                as_type="generation",
                model=MODEL,
                model_parameters={"max_tokens": 256, "temperature": 0.0},
            ) as obs2:
                step2 = client.messages.create(
                    temperature=0.0,
                    model=MODEL,
                    max_tokens=256,
                    system=STEP2_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": candidates}],
                )
                joke = step2.content[0].text.strip()
                obs2.update(
                    output={"joke": joke},
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
