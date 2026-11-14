#!/usr/bin/env python3
"Propose CMS operations without executing live CMS API calls."

import logging
from openai import AsyncOpenAI
from .base_agent import BaseAgent
from pydantic import ConfigDict
from typing import List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CMSAutomationAgent(BaseAgent):
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
            name="CMS Automation Agent",
            model="gpt-4.1-nano",
            instructions=(
                """
                You are the CMS Automation Agent — automate bulk site operations via October CMS.

                Instructions:
                1. Receive instructions for bulk tasks (e.g., create pages, update templates, configure menus).
                2. Generate scripts or API sequences to execute these tasks.
                3. Ensure idempotency and error handling in generated operations.
                4. Always in English with detailed API sequences or scripts.

                Output: JSON with task scripts or API call list.
                """
            ),
            tools=[],
        )
        self.user_id = user_id
        self.gpt_client = AsyncOpenAI(api_key=openai_api_key, http_client=async_client)

    async def handle_message(
        self, message: str, context: Optional[List[dict]] = None, image: Optional[str] = None
    ) -> str:
        try:
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
            logger.info(f"CMS Automation Agent for {self.user_id} returned: {reply}")
            return reply
        except Exception as e:
            logger.error(f"Error in CMS Automation Agent for {self.user_id}: {e}")
            return "CMS Automation Error."
