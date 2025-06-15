import json
from langchain_core.tools import StructuredTool
from pydantic import Field

from app.database.models import User
from app.tools.registry import get_tools_metadata, Tool
from app.tools.search_knowledge import search_knowledge
from app.tools.generate_exercise import generate_exercise


def get_langchain_tools_by_user_and_availability(
    user: User, available_tools: list[Tool]
) -> list[StructuredTool]:
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

        if tool_name not in available_tools:
            continue

        # Map tool names to actual functions
        if tool_name == Tool.SEARCH_KNOWLEDGE:
            tool_func = search_knowledge
        elif tool_name == Tool.GENERATE_EXERCISE:
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


async def execute_tool_call(tool_name: str, tool_args: dict) -> str:
    """Execute a tool call and return the result."""
    try:
        if tool_name == Tool.SEARCH_KNOWLEDGE:
            search_phrase = tool_args.get("search_phrase")
            class_id = tool_args.get("class_id")

            if not search_phrase or not isinstance(search_phrase, str):
                return "Error: search_phrase is required and must be a string"
            if not class_id or not isinstance(class_id, int):
                return "Error: class_id is required and must be an integer"

            return await search_knowledge(
                search_phrase=search_phrase, class_id=class_id
            )
        elif tool_name == Tool.GENERATE_EXERCISE:
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
