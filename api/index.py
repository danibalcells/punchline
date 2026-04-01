import os
from contextlib import suppress

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langfuse import Langfuse
from pydantic import BaseModel

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-6"
SYSTEM_PROMPT = """You are a deadpan comedy writer. Given a punchline, write a complete joke that ends with that exact punchline.

Your style: dry, understated, absurdist, unexpected. Think Steven Wright, Mitch Hedberg, or early Norm MacDonald. 
Avoid: cheesy wordplay, "Why did the X cross the Y" structures, obvious puns, groan-worthy setups.
Aim for: surprising logic, mundane observations taken to strange conclusions, anti-jokes, deadpan absurdism.

Return the full joke as plain text — setup followed by the punchline on a new line.
No quotation marks. No labels. No explanation. Just the joke.
The last line must be the punchline exactly as given."""

app = FastAPI()
langfuse = Langfuse()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    punchline: str


class GenerateResponse(BaseModel):
    joke: str


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    with langfuse.start_as_current_observation(
        name="generate-joke",
        as_type="generation",
        input={"punchline": request.punchline},
        model=MODEL,
        model_parameters={"max_tokens": 256},
    ) as obs:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model=MODEL,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Punchline: {request.punchline}"}],
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
