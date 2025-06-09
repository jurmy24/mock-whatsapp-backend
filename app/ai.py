from typing import List, Optional
from dotenv import load_dotenv
import os
import json
from langchain_together import ChatTogether
from langchain_core.tools import StructuredTool
from pydantic import SecretStr, Field

from app.database.db import get_user_message_history
import app.database.db as db
from app.database.models import Message, User, MessageRole
from app.tools.registry import get_tools_metadata
from app.tools.search_knowledge import search_knowledge
from app.tools.generate_exercise import generate_exercise

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")


system_prompt = """<role>
You are Twiga, a WhatsApp bot developed by the Tanzania AI Community for secondary school teachers in Tanzania. You assist teachers by chatting with them and providing them with accurate, curriculum-aligned education materials. You understand that you are communicating on the WhatsApp messaging platform and that you have access to the textbooks for the course the teacher is teaching. You will often need to use the course materials to ensure that your responses are contextually grounded. You are friendly and helpful, always aiming to provide clear explanations whether you're providing educational content or just chatting.
</role>

<context>
The curriculum is the Tanzanian National Curriculum, developed by the Tanzanian Institute of Education (TIE). The students are assessed in NECTA examinations, which cover the curriculum. TIE are also the writers of the textbooks you use. Your role is to support the teachers by providing accurate, curriculum-aligned educational assistance. You are talking to {user_name} who teaches {class_info}.
</context>

<response_format>

1. **Respond using WhatsApp markdown formatting**

To italicize your text for important information, place an underscore on both sides of the text: _text_
To bold your text for section headers, place an asterisk on both sides of the text: *text*
To strikethrough your text (eg. to clarify a previous mistake), place a tilde on both sides of the text: ~text~
To monospace your text, place three backticks on both sides of the text: `text`
To add a bulleted list to your text, place a hyphen and a space before each word or sentence:
- text1
- text2
To add a numbered list to your text (eg. when giving instructions or step-by-steps), place a number, period, and space before each line of text:
1. text1
2. text2
To write a quote style format when displaying a generated exercise, place an angle bracket and space before the text: > text
To add inline code to your text (eg. for equations or just to emphasize something), place a backtick on both sides of the message: `text`

2. **If the user's input is unclear or ambiguous**:
Request explanations, guidance, or provide suggestions.

3. **If you expect to write a long response:**
Section it with headers with paragraphs using the boldened text like _Header Name_. Do not use bullet points as headers.

</response_format>

<important>
## Instruction Reminder

Remember your instructions, follow the response format and focus on what the user is asking for.

- You only communicate in english
- Use the tools you have available
- Be clear and concise, since your messages are communicated and formatted on WhatsApp
- Ask the teacher for additional information or clarification if its needed
- Do not generate educational content if they are not provided by your tools
- If the tool has an error or does not fulfill the user request, tell the user
- Only help the teacher with subject related matter
- The user can update their subjects and personal settings manually by just typing "settings"

Here are your capabilities:

1. TOOL: "search_knowledge" - Searching the textbooks to answer course-related questions
2. TOOL: "generate_exercise" - Generating example exercises or questions based on a specific course-related topic
3. General tips and support
</important>
"""


chat = ChatTogether(
    api_key=SecretStr(LLM_API_KEY) if LLM_API_KEY else None,
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
)


def create_langchain_tools(user: User) -> List[StructuredTool]:
    """Create LangChain tools from the tool registry for a specific user."""

    # Get user's available classes and format for registry
    available_classes = json.dumps(user.class_name_to_id_map)

    # Get tool metadata from registry
    tools_metadata = get_tools_metadata(available_classes)

    # Convert metadata to LangChain tools
    langchain_tools = []

    for tool_meta in tools_metadata:
        func_info = tool_meta["function"]
        tool_name = func_info["name"]

        # Map tool names to actual functions
        if tool_name == "search_knowledge":
            tool_func = search_knowledge
        elif tool_name == "generate_exercise":
            tool_func = generate_exercise
        else:
            continue  # Skip unknown tools

        # Create dynamic argument schema from registry metadata
        properties = func_info["parameters"]["properties"]
        required_fields = func_info["parameters"]["required"]

        # Build schema fields dynamically
        schema_fields = {}
        for prop_name, prop_info in properties.items():
            field_type = str if prop_info["type"] == "string" else int
            field_desc = prop_info["description"]

            # Create Field with constraints
            field_kwargs = {"description": field_desc}

            # Add enum constraint if present (as validation, not type constraint)
            if "enum" in prop_info:
                # For int enums, we'll validate in the function itself
                # For now, just use the base type
                pass

            schema_fields[prop_name] = (field_type, Field(**field_kwargs))

        # Create dynamic Pydantic model
        from pydantic import create_model

        args_schema = create_model(f"{tool_name.title()}Args", **schema_fields)

        # Create StructuredTool
        langchain_tool = StructuredTool.from_function(
            func=tool_func,
            name=tool_name,
            description=func_info["description"],
            args_schema=args_schema,
            return_direct=False,
        )

        langchain_tools.append(langchain_tool)

    return langchain_tools


async def execute_tool_call(tool_name: str, tool_args: dict, user: User) -> str:
    """Execute a tool call and return the result."""
    try:
        if tool_name == "search_knowledge":
            search_phrase = tool_args.get("search_phrase")
            class_id = tool_args.get("class_id")

            if not search_phrase or not isinstance(search_phrase, str):
                return "Error: search_phrase is required and must be a string"
            if not class_id or not isinstance(class_id, int):
                return "Error: class_id is required and must be an integer"

            return await search_knowledge(
                search_phrase=search_phrase, class_id=class_id
            )
        elif tool_name == "generate_exercise":
            query = tool_args.get("query")
            class_id = tool_args.get("class_id")
            subject = tool_args.get("subject")

            if not query or not isinstance(query, str):
                return "Error: query is required and must be a string"
            if not class_id or not isinstance(class_id, int):
                return "Error: class_id is required and must be an integer"
            if not subject or not isinstance(subject, str):
                return "Error: subject is required and must be a string"

            return generate_exercise(query=query, class_id=class_id, subject=subject)
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


async def generate_response(
    user: User,
    message: Message,
) -> Optional[Message]:
    """Generate a response, handling message batching and tool calls."""
    if user.id is None:
        return None

    history = get_user_message_history(user.id)
    api_messages = _format_messages([message], history, user)

    # Create and bind tools to the chat model
    tools = create_langchain_tools(user)
    chat_with_tools = chat.bind_tools(tools)

    # Get the initial response
    llm_response = chat_with_tools.invoke(api_messages)
    print("LLM RESPONSE")
    print(llm_response)

    # Check if the response contains tool calls
    tool_calls = getattr(llm_response, "tool_calls", None)
    if tool_calls:
        print("TOOL CALLS")
        # Save the assistant message with tool calls
        tool_calls_data = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
            }
            for tc in tool_calls
        ]

        assistant_message_with_tools = Message(
            user_id=user.id,
            role=MessageRole.assistant,
            content=None,  # No content when tool calls are present
            tool_calls=tool_calls_data,
        )
        db.create_new_message(assistant_message_with_tools)

        # Execute tool calls and save tool messages
        tool_results = []
        for tool_call in tool_calls:
            result = await execute_tool_call(tool_call["name"], tool_call["args"], user)
            tool_results.append(result)

            # Save the tool result message
            tool_message = Message(
                user_id=user.id,
                role=MessageRole.tool,
                content=result,
                tool_call_id=tool_call["id"],
                tool_name=tool_call["name"],
            )
            db.create_new_message(tool_message)

        # Convert to API format for final response
        api_messages.append(
            {"role": "assistant", "content": None, "tool_calls": tool_calls_data}
        )

        # Add tool result messages to API format
        for i, tool_call in enumerate(tool_calls):
            api_messages.append(
                {
                    "role": "tool",
                    "content": tool_results[i],
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["name"],
                }
            )

        # Get final response after tool execution
        final_response = chat.invoke(api_messages)
        response_content = str(final_response.content) if final_response.content else ""

        # Save the final assistant response
        final_message = Message(
            user_id=user.id,
            role=MessageRole.assistant,
            content=response_content,
        )
        db.create_new_message(final_message)

        return final_message
    else:
        print("NO TOOL CALLS")
        response_content = str(llm_response.content) if llm_response.content else ""

        # Create and return a Message object for non-tool responses
        final_message = Message(
            user_id=user.id,
            role=MessageRole.assistant,
            content=response_content,
        )
        db.create_new_message(final_message)


def _format_messages(
    new_messages: List[Message],
    database_messages: Optional[List[Message]],
    user: User,
) -> List[dict]:
    """
    Format messages for the API, removing duplicates between new messages and database history.
    """
    # Initialize with system prompt
    formatted_messages = [
        {
            "role": "system",
            "content": system_prompt.format(
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
