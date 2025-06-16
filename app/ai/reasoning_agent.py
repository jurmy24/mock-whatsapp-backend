import json
import time
import re
from typing import Any

from langchain_together import ChatTogether

from app.ai.agent import Agent
from app.ai.utils import print_boxed
from app.ai.config import AgentConfig, AgentContext
from app.ai.tools import execute_tool_call
import app.database.db as db
from app.database.models import Message, User, MessageRole


class ReasoningAgent(Agent):
    def __init__(self, chat: ChatTogether, config: AgentConfig, context: AgentContext):
        super().__init__(chat, config, context)

    def llm_call(self) -> str:
        return self.chat.invoke(self.message_history)

    async def run(self, message: list[dict]) -> Message:
        api_messages = self._format_messages(
            new_messages=[message],
            database_messages=None,  # history,
            user=self.context.user,
        )
        self.message_history.extend(api_messages)

        result = None
        current_iteration = 0
        time_start = time.time()
        timeout = time_start + 60 * self.timeout_s

        task = self.message_history[-1]["content"]
        print("🚀 Starting ReAct Agent")
        print(f"📝 Task: {task}")
        print("=" * 80)

        while current_iteration < self.max_iterations:
            if time.time() > timeout:
                print("⚠️ Timeout exceeded. Returning last result:")
                return "Task incomplete - Timeout exceeded"

            try:
                llm_response = self.llm_call()

                result, action = self.parse_response(llm_response.content)

                if action is None:
                    print_boxed(result, "Final Answer", "🎯")

                    final_message = self._generate_final_response()
                    return final_message

                if action.get("error") is not None:
                    raise Exception

                thought = result

                print_boxed(
                    thought, f"Thought (Iteration {current_iteration + 1})", "🤔"
                )
                print_boxed(json.dumps(action), "Action", "🛠️")

                action_result = await execute_tool_call(action["name"], action["args"])

                self._save_action(action=action, result=action_result)

                print_boxed(action_result, f"Result from action {action}")

                add_to_history = f"Thought: {thought}\nAction: {action}"
                self.message_history.append(
                    {"role": "assistant", "content": add_to_history}
                )
                self.message_history.append(
                    {"role": "user", "content": f"Observation: {action_result}"}
                )

                current_iteration += 1
                print("-" * 80)

            except Exception as e:
                print(f"❌ Error in iteration {current_iteration + 1}: {str(e)}")
                self.message_history.append(
                    {
                        "role": "user",
                        "content": f"Error occurred: {str(e)}. Please try a different approach.",
                    }
                )
                current_iteration += 1

        print(
            f"⚠️ Maximum iterations ({self.max_iterations}) reached without completion"
        )
        return "Task incomplete - maximum iterations reached"

    def parse_response(self, response: str) -> tuple[str, dict[str, Any] | None]:
        """Parse the LLM response and extract thought and action input"""
        print(response)
        if "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[1].strip()
            return final_answer, None

        if "Thought:" in response and "Action:" in response:
            thought_match = re.search(
                r"Thought:\s*(.*?)\n\nAction:", response, re.DOTALL
            )
            thought = thought_match.group(1).strip() if thought_match else None

            action_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            if not action_match:
                action_match = re.search(r"Action:\s*(\{.*\})", response, re.DOTALL)

            action = json.loads(action_match.group(1)) if action_match else None

            action = {
                "id": action["id"],
                "name": action["name"],
                "args": action["args"],
            }

        else:
            thought = "The assistant didn't follow the ReAct format properly."
            action = {"error": "The assistant didn't follow the ReAct format properly."}

        return thought, action

    def _save_action(self, action: dict[str, Any], result: str) -> None:
        tool_message = Message(
            user_id=self.context.user.id,
            role=MessageRole.tool,
            content=result,
            tool_call_id=action["id"],
            tool_name=action["name"],
        )
        db.create_new_message(tool_message)
        return

    def _generate_final_response(self) -> Message:
        final_response = self.chat.invoke(self.message_history)
        response_content = str(final_response.content) if final_response.content else ""

        final_message = Message(
            user_id=self.context.user.id,
            role=MessageRole.assistant,
            content=response_content,
        )
        db.create_new_message(final_message)

        return final_message

    def _format_messages(
        self,
        new_messages: list[Message],
        database_messages: list[Message] | None,
        user: User,
    ) -> list[dict]:
        """
        Format messages for the API, removing duplicates between new messages and database history.
        """
        # Initialize with system prompt
        formatted_messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(
                    user_name=user.name,
                    class_info=user.formatted_class_info,
                ),
            }
        ]

        # Add history messages
        if database_messages:
            # Exclude potential duplicates
            message_count = len(new_messages)
            db_message_count = len(database_messages)

            # Safety check: ensure we don't slice more messages than we have
            if db_message_count < message_count:
                raise Exception(
                    f"Unusual message count scenario detected: There are {message_count} new messages but only {db_message_count} messages in the database."
                )

            old_messages = (
                database_messages[:-message_count]
                if message_count > 0
                else database_messages
            )
            formatted_messages.extend(msg.to_api_format() for msg in old_messages)

        # Add new messages
        formatted_messages.extend(msg.to_api_format() for msg in new_messages)

        return formatted_messages
