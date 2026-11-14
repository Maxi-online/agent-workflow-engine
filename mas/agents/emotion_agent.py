#!/usr/bin/env python3
"Estimate emotional tone through an LLM; this is not a trained emotion classifier."

import logging
from openai import AsyncOpenAI
from .base_agent import BaseAgent
from pydantic import ConfigDict
from typing import List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionAgent(BaseAgent):
    # Pydantic v2 configuration
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(
        self,
        user_id: str,
        openai_api_key: str,
        knowledge_client: Any = None,
        async_client: str = None,
        first_name: str = "",
        last_name: str = "",
        username: str = "",
        about_me: str = "",
    ):
        super().__init__(
            name="Emotion Agent",
            model="gpt-4.1-nano",
            instructions=(
                """
                You are the Emotion Agent — your task is to analyze the emotional tone of the user's message.
                Output a JSON object where keys are emotion labels and values are intensity scores between 0 and 1.
                Always output valid JSON.
                """
            ),
            tools=[],
        )
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.username = username
        self.about_me = about_me
        self.gpt_client = AsyncOpenAI(api_key=openai_api_key, http_client=async_client)

    async def handle_message(
        self, message: str, context: Optional[List[dict]] = None, image: Optional[str] = None
    ) -> str:
        try:
            # Directly analyze emotion without external knowledge base
            payload = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": message},
            ]
            response = await self.gpt_client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=0.0,
                max_tokens=1000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stream=False,
            )
            reply = response.choices[0].message.content.strip()
            logger.info(f"Emotion Agent for {self.user_id} returned: {reply}")
            return reply
        except Exception as e:
            logger.error(f"Error in Emotion Agent for {self.user_id}: {e}")
            return "{}"
