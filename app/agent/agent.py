from abc import ABC, abstractmethod
import os
from typing import Any

from langchain_together import ChatTogether
from langchain_core.tools import StructuredTool

from app.agent.tools import get_langchain_tools_by_user_and_availability
from app.agent.config import AgentConfig, AgentContext


class Agent(ABC):
    def __init__(self, chat: ChatTogether, config: AgentConfig, context: AgentContext):
        self.context = context
        self.chat = chat
        self.config = config
        self.max_iterations = config.max_iterations
        self.model = config.model
        self.system_prompt = config.prompt
        self.temperature = config.temperature
        self.timeout_s = config.timeout_s

        self.tools: list[StructuredTool] | None = (
            self._get_tools_by_agent_context(context) if context.should_use_tools else None
        )

        self.message_history: list[dict[str, Any]] = []

    @abstractmethod
    def llm_call(self) -> str:
        pass

    @abstractmethod
    def parse_response(self, response: str) -> Any:
        pass

    @abstractmethod
    async def run(self, input: str):
        pass

    @staticmethod
    def _get_tools_by_agent_context(context: AgentContext) -> list[StructuredTool]:
        return get_langchain_tools_by_user_and_availability(
            user=context.user,
            available_tools=context.available_tools,
        )

    def get_chat(model: str) -> ChatTogether:
        agent_chat = ChatTogether(
            api_key=os.getenv("LLM_API_KEY"),
            model=model, #"meta-llama/Llama-4-Scout-17B-16E-Instruct",
        )
        return agent_chat
