from together import Together
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")


def get_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text using the Together API."""
    client = Together(api_key=LLM_API_KEY)
    response = client.embeddings.create(
        model="BAAI/bge-large-en-v1.5",
        input=text,
    )
    embedding = response.data[0].embedding

    if embedding is None:
        raise ValueError("Failed to generate embedding")

    return embedding
