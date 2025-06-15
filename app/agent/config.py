import uuid
from pydantic import BaseModel
from typing import NamedTuple

from app.agent.system_prompt import system_prompt
from app.database.models import User
from app.tools.registry import Tool


class AgentContext(NamedTuple):
    user: User
    should_use_tools: bool
    available_tools: list[Tool]


class AgentConfig(BaseModel):
    model: str
    max_iterations: int
    timeout_s: int
    temperature: float
    track_id: str = uuid.uuid4()
    prompt: str = system_prompt
    should_print: bool = False


available_tools: list[Tool] = [
    Tool.GENERATE_EXERCISE,
    Tool.SEARCH_KNOWLEDGE,
]

reasoning_agent_config = AgentConfig(
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    max_iterations=5,
    timeout_s=20,
    temperature=0.1,
    prompt=system_prompt,
    should_print=True,
)