#!/usr/bin/env python3
"""
LLM API utilities for creating API handles for different LLM providers.
"""

import os
import requests
import asyncio
from typing import List, Dict, Callable


def create_openai_api_handle(
    base_url: str,
    api_key: str,
    model: str,
    clear_proxy: bool = True
) -> Callable:
    """
    Create a function handle for OpenAI-compatible API using requests.

    Args:
        base_url: Base URL of the API endpoint (e.g., "http://api.openai.com/v1")
        api_key: API key for authentication
        model: Model name to use (e.g., "gpt-4", "deepseek-v3-1-terminus")
        clear_proxy: Whether to clear proxy environment variables (default: True)

    Returns:
        A callable function that takes messages and kwargs, returns LLM response

    Example:
        >>> llm_handle = create_openai_api_handle(
        ...     base_url="http://localhost:8000/v1",
        ...     api_key="your-api-key",
        ...     model="gpt-4"
        ... )
        >>> response = llm_handle([{"role": "user", "content": "Hello"}])
    """

    # Clear proxy environment variables to avoid SOCKS proxy issues
    if clear_proxy:
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        os.environ.pop('ALL_PROXY', None)
        os.environ.pop('all_proxy', None)

    def openai_api_handle(messages: List[Dict], **kwargs) -> Dict:
        """
        Handle OpenAI-compatible API requests.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Response dictionary with 'role', 'content', and optionally 'usage'

        Raises:
            RuntimeError: If API request fails or response is invalid
        """
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4000),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=kwargs.get('timeout', 120),
                proxies={'http': None, 'https': None}  # Disable proxy
            )
            response.raise_for_status()
            data = response.json()

            choice = data['choices'][0]
            message = choice['message']

            result = {
                "role": message.get('role', 'assistant'),
                "content": message.get('content', ''),
                "model": data.get('model', model)
            }

            if 'usage' in data:
                result['usage'] = data['usage']

            return result

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Invalid API response: {str(e)}")

    return openai_api_handle



def create_openai_api_handle_async(
    base_url: str,
    api_key: str,
    model: str,
    clear_proxy: bool = True
) -> Callable:
    """
    Create an async function handle for OpenAI-compatible API using aiohttp.

    Args:
        base_url: Base URL of the API endpoint (e.g., "http://api.openai.com/v1")
        api_key: API key for authentication
        model: Model name to use (e.g., "gpt-4", "deepseek-v3-1-terminus")
        clear_proxy: Whether to clear proxy environment variables (default: True)

    Returns:
        An async callable function that takes messages and kwargs, returns LLM response

    Example:
        >>> llm_handle = create_openai_api_handle_async(
        ...     base_url="http://localhost:8000/v1",
        ...     api_key="your-api-key",
        ...     model="gpt-4"
        ... )
        >>> response = await llm_handle([{"role": "user", "content": "Hello"}])
    """

    if clear_proxy:
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        os.environ.pop('ALL_PROXY', None)
        os.environ.pop('all_proxy', None)

    # Shared session for connection pooling (lazy initialization)
    session = None

    async def openai_api_handle_async(messages: List[Dict], **kwargs) -> Dict:
        """
        Handle OpenAI-compatible API requests asynchronously.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Response dictionary with 'role', 'content', and optionally 'usage'

        Raises:
            RuntimeError: If API request fails or response is invalid
        """
        import aiohttp

        nonlocal session

        # Create shared session if not exists
        if session is None or session.closed:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            session = aiohttp.ClientSession(connector=connector)

        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4000),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False
        }

        try:
            timeout = aiohttp.ClientTimeout(total=kwargs.get('timeout', 120))
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            ) as response:
                response.raise_for_status()
                data = await response.json()

            choice = data['choices'][0]
            message = choice['message']

            result = {
                "role": message.get('role', 'assistant'),
                "content": message.get('content', ''),
                "model": data.get('model', model)
            }

            if 'usage' in data:
                result['usage'] = data['usage']

            return result

        except aiohttp.ClientError as e:
            raise RuntimeError(f"API request failed: {str(e)}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Invalid API response: {str(e)}")

    return openai_api_handle_async


__all__ = [
    'create_openai_api_handle',
    'create_openai_api_handle_async'
]
