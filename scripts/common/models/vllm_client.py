import os
from dotenv import load_dotenv
import requests
from typing import Dict, List

load_dotenv()

class VllmClient:
    def __init__(self) -> None:
        # Save for parity with callers that may introspect these attrs
        self.base_url = os.environ['VLLM_URL']
        
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float=0.0,
    ) -> str:
        """Generate a chat completion from the llm server
        """
        url = f"{self.base_url}/v1/chat/completions"
        model = os.environ['VLLM_MODEL_PATH']
        max_tokens = int(os.environ['MAX_TOKENS'])
        payload_dictionary = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(url, json=payload_dictionary)
        
        # raise an exception in the case of the response failing
        response.raise_for_status()
        
        json_response = response.json()
        llm_response_text = json_response["choices"][0]["message"]["content"]
        return llm_response_text