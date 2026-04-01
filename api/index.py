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

SYSTEM_PROMPT = """You are a comedy writer. Given a punchline, write a complete joke that ends with that exact punchline.

Your style draws from two influences:
- The deadpan sincerity of Miguel Noguera: treating absurd ideas with total seriousness, as if documenting a social phenomenon or making a reasonable proposal.
- The confident ignorance of Trailer Park Boys: mangled logic delivered with complete earnestness, like someone who's absolutely sure they're making sense.

The common thread: the joke should never wink at the audience. The setup should treat the punchline as a perfectly natural, obvious conclusion. The humor comes from the gap between the speaker's confidence and the absurdity of what they're saying.

Avoid: self-aware humor, meta-jokes, corny puns, "Why did the X..." formats, anything that feels like it's trying to be funny.
Aim for: unearned confidence, mundane-to-bizarre escalation, earnest delivery of nonsense, the feeling that the person telling this joke has no idea it's a joke.

Return the full joke as plain text — setup followed by the punchline on a new line.
No quotation marks. No labels. No explanation. Just the joke.
The last line must be the punchline. Correct the formatting (e.g. capitalization, punctuation) of the punchline to match the rest of the joke."""

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

    with langfuse.start_as_current_observation(
        name="generate-joke",
        as_type="generation",
        input={"punchline": body.punchline},
        model=MODEL,
        model_parameters={"max_tokens": 256},
    ) as obs:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                temperature=0.7,    
                model=MODEL,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Punchline: {body.punchline}"}],
            )
            joke = message.content[0].text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM call failed: {e}") from e

        obs.update(
            output={"joke": joke},
            usage_details={
                "input": message.usage.input_tokens,
                "output": message.usage.output_tokens,
            },
        )

    return GenerateResponse(joke=joke)


with suppress(Exception):
    public_dir = os.path.join(os.path.dirname(__file__), "..", "public")
    if os.path.isdir(public_dir):
        app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")
