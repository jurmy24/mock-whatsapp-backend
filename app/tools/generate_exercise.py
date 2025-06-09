import os
from typing import List, Optional

from app.database.db import vector_search, get_class_resources
from app.database.models import Chunk, Resource, ChunkType
from together import Together
from dotenv import load_dotenv


load_dotenv()

client = Together(api_key=os.getenv("LLM_API_KEY"))


exercise_generator_prompt = """You are a skilled Tanzanian secondary school teacher that generates questions or exercises for students based on the request made by the user. This exercise is for {class_info} students. Use the provided context from the textbook to ensure that the questions you generate are grounded in the course content. Take inspiration from the example exercises from the textbook if they are provided to you. Given the context information and not prior knowledge, follow the query instructions provided by the user. Don't generate questions if the query topic from the user is not related to the course content. Begin your response immediately with the question.

EXAMPLE INTERACTION:
user: Follow these instructions (give me short answer question on Tanzania's mining industry)
Context information is below.
Tanzania has many minerals that it trades with to other countries...etc.

assistant: > List three minerals that Tanzania exports.
"""

exercise_user_prompt = """
Follow these instructions ({query})

Context information is below:

{context_str}

"""


def generate_exercise(
    query: str,
    class_id: int,
    subject: str,
) -> str:
    class_id = int(class_id)
    # Retrieve the resources for the class
    resource_ids = get_class_resources(class_id)
    assert resource_ids

    # Retrieve the relevant content and exercises
    retrieved_content = vector_search(
        query=query,
        n_results=7,
        where={
            "chunk_type": [ChunkType.text],
            "resource_id": resource_ids,
        },
    )

    # Format the context and prompt
    context = _format_context(retrieved_content)

    messages = [
        {
            "role": "system",
            "content": exercise_generator_prompt.format(class_info=subject),
        },
        {
            "role": "user",
            "content": exercise_user_prompt.format(query=query, context_str=context),
        },
    ]

    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        messages=messages,
        max_tokens=100,
    )
    return response.choices[0].message.content


def _format_context(
    retrieved_content: List[Chunk],
    retrieved_exercise: Optional[List[Chunk]] = None,
    resources: Optional[List[Resource]] = None,
):
    # Formatting the context
    context_parts = []
    if resources:
        if len(resources) == 1:
            context_parts.append(
                f"### Context from the resource ({resources[0].name})\n"
            )
        else:
            # TODO: Make this neater another time
            resource_titles = ", ".join(
                [f"{resource.id}. {resource.name}" for resource in resources]
            )
            context_parts.append(
                f"### Context from the resources ({resource_titles})\n"
            )

    if retrieved_exercise:
        for chunk in retrieved_content + retrieved_exercise:
            # TODO: Make this neater another time
            if chunk.top_level_section_title and chunk.top_level_section_index:
                heading = f"-{chunk.chunk_type} from chapter {chunk.top_level_section_index}. {chunk.top_level_section_title} in resource {chunk.resource_id}"
            elif chunk.top_level_section_title:
                heading = f"-{chunk.chunk_type} from section {chunk.top_level_section_title} in resource {chunk.resource_id}"
            else:
                heading = f"-{chunk.chunk_type} from resource {chunk.resource_id}"

            context_parts.append(heading)
            context_parts.append(f"{chunk.content}")

    return "\n".join(context_parts)
