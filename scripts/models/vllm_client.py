import os
from dotenv import load_dotenv
from typing import Dict, List, Optional
import re
import httpx

load_dotenv()

class VllmClient:
    def __init__(self) -> None:
        # Save for parity with callers that may introspect these attrs
        self.base_url = os.environ['VLLM_URL']
        self.async_client = httpx.AsyncClient(timeout=300, limits=httpx.Limits(max_connections=int(os.environ['LLM_MAX_CONCURRENCY']), max_keepalive_connections=int(os.environ['LLM_MAX_CONCURRENCY'])))
        
    def _clean_response(self, response: str) -> str:
        """Cleans response text from the llm server by removing thought sections
        """
        return re.sub(
            r"<(think|thought).*?>.*?</\1.*?>", 
            "", 
            response, 
            flags=re.DOTALL | re.IGNORECASE
        ).strip()
    
    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float=0.0,
        guided_json: Optional[dict]=None,
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
        if guided_json is not None:
            payload_dictionary['guided_json'] = guided_json
        
        response = await self.async_client.post(url, json=payload_dictionary)
        response.raise_for_status()
        json_response = response.json()
        llm_response_text = json_response["choices"][0]["message"]["content"]
        return self._clean_response(llm_response_text)