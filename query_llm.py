import json
from openai import APIConnectionError, OpenAI
from config import get_global_config

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
    # First, attempt to extract summary from potential JSON output
    summary_content = None
    
    # Try to find a ```json ... ``` block
    json_block_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_response, re.DOTALL)
    if json_block_match:
        try:
            json_data = json.loads(json_block_match.group(1))
            if "summary" in json_data:
                summary_content = json_data["summary"]
        except json.JSONDecodeError:
            pass # Not valid JSON within the block

    # If no JSON block summary, try to parse the whole response as direct JSON
    if summary_content is None:
        try:
            direct_json_data = json.loads(raw_response)
            if "summary" in direct_json_data:
                summary_content = direct_json_data["summary"]
        except json.JSONDecodeError:
            pass # Not direct JSON

    # If a summary was successfully extracted, return it clean
    if summary_content is not None:
        # Collapse excess whitespace in the extracted summary
        return re.sub(r"\s+", " ", summary_content).strip()

    # If no valid JSON summary was found, fall back to original cleaning logic
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

    with open(f"{'synthetic_data' if get_global_config().use_synthetic_data else 'real_data'}/debug_prompts_and_responses.jsonl", "a") as f:
        f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
    
    return clean_response(response).strip()