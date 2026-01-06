import os
from dotenv import load_dotenv
import requests
from typing import Dict, List
import re

load_dotenv()

class VllmClient:
    def __init__(self) -> None:
        # Save for parity with callers that may introspect these attrs
        self.base_url = os.environ['VLLM_URL']
        
    def _clean_response(self, response: str) -> str:
        """Cleans response text from the llm server by removing thought sections
        """
        return re.sub(
            r"<(think|thought).*?>.*?</\1.*?>", 
            "", 
            response, 
            flags=re.DOTALL | re.IGNORECASE
        ).strip()
        
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
        return self._clean_response(llm_response_text)