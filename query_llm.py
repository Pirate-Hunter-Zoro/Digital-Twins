import json
from openai import APIConnectionError, OpenAI

# Lazy singleton pattern so we only load the model once
_llm_client = None

def get_llm_client():
    global _llm_client
    if _llm_client is None:
            _llm_client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="not-needed-for-localhost",
            )
    return _llm_client

import re

def clean_response(raw_response: str) -> str:
    # Remove <think>...</think> blocks (including multiline)
    no_thought = re.sub(r"<think>[\s\S]*?<\/think>", "", raw_response)

    # Remove bracketed annotations like [probably], [maybe], etc.
    # Only remove if they're not part of any actual diagnosis/med/tx string
    clean = re.sub(r"\s*\[.*?\]\s*", " ", no_thought)

    # Collapse excess whitespace to avoid awkward gaps
    return re.sub(r"\s+", " ", clean).strip()

def query_llm(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    llm_client = get_llm_client()
    try:
        output = llm_client.chat.completions.create(
            model="medgemma",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Only return structured responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response = output.choices[0].message.content
    except APIConnectionError as e:
        response = f"ERROR: API connection failed - {str(e)}"

    with open("debug_prompts_and_responses.jsonl", "a") as f:
        f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
    
    return clean_response(response).strip()