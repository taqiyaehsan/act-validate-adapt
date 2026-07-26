"""
LLM API Client — OpenAI GPT Wrapper for Hypothesis Generation
================================================================

Thin wrapper around the OpenAI Chat Completions API.  Used by
``LLMGenerator`` in ``generators.py`` to query GPT-4 for plausible
causal edge hypotheses given a domain description.

DOMAIN-AGNOSTIC KNOBS:
  - ``api_key``:       OpenAI API key (loaded from .env via OPENAI_API_KEY).
  - ``base_url``:      API endpoint.  Change for Azure OpenAI or other providers.
  - ``max_retries``:   Retry count on transient failures (default 3).
  - ``model``:         Set in the calling code (LLMGenerator); default gpt-4.

The prompt content is constructed in ``LLMGenerator.generate()`` — update
that prompt when adapting to a new domain.
"""

import json
from typing import Dict, List, Any
import requests
import time
import logging

logger = logging.getLogger(__name__)


class GPTClient:
    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: float = 1.0):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def _clean_json_response(self, content: str) -> str:
        """Clean the response string to ensure valid JSON"""
        # Remove any "json" prefix
        if content.strip().startswith('`json'):
            content = content.split('`json', 1)[1]
        # Remove any remaining backticks
        content = content.strip('`')
        return content.strip()

    def generate_response(self, system_prompt: str, user_message: str, temperature=1, top_p=0.9) -> str:
        self.call_count += 1
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": 1000
                    },
                    timeout=60
                )

                if response.status_code == 429:
                    wait_time = float(response.headers.get('Retry-After', self.retry_delay))
                    logger.warning(f"Rate limit hit, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                    
                if response.status_code == 500:
                    logger.error("OpenAI server error")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                    
                if response.status_code != 200:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    if attempt == self.max_retries - 1:
                        raise Exception(error_msg)
                    time.sleep(self.retry_delay)
                    continue

                result = response.json()
                usage = result.get('usage', {})
                self.total_prompt_tokens += usage.get('prompt_tokens', 0)
                self.total_completion_tokens += usage.get('completion_tokens', 0)
                self.total_tokens += usage.get('total_tokens', 0)
                content = result['choices'][0]['message']['content']
            
                # Clean and validate JSON before returning
                cleaned_content = self._clean_json_response(content)
                try:
                    # Test if it's valid JSON
                    # json.loads(cleaned_content)
                    return cleaned_content
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON after cleaning: {cleaned_content}")
                    logger.error(f"JSON error: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise
                    continue

            except requests.Timeout:
                logger.error(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise
                    
            except requests.RequestException as e:
                logger.error(f"Network error: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)

            except requests.ConnectionError:
                # Add exponential backoff
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Connection error, retrying in {wait_time}s")
                time.sleep(wait_time)
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
                
        raise Exception("Max retries exceeded")