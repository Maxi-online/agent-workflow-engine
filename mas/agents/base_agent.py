from abc import ABC, abstractmethod
import os
from pydantic import BaseModel, ConfigDict
from typing import List, Any, Callable, Optional


class BaseAgent(BaseModel, ABC):
    "Shared Pydantic contract for a text or image specialist."

    name: str = "Agent"
    model: str = "gpt-4.1-nano"
    instructions: str = "You are a helpful agent."
    tools: List[Callable[..., Any]] = []

    @abstractmethod
    async def handle_message(
        self, message: str, context: Optional[List[dict]] = None, image: Optional[str] = None
    ) -> str:
        "Process user text, a bounded conversation context and an optional image data URI; return the specialist response or a supported handoff directive."
        raise NotImplementedError("handle_message method to be implemented in sub-classes.")

    # New configuration for Pydantic v2:
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def model_post_init(self, context):
        if os.getenv("OPENAI_MODEL"):
            self.model = os.environ["OPENAI_MODEL"]
