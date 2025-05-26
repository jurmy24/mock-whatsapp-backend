from typing import List, Optional
from app.embeddings import get_embedding
from sqlmodel import Session
from app.database.engine import db_engine
from app.database.models import Chunk
from sqlalchemy import text
from sqlmodel import select, desc
from sqlalchemy.orm import selectinload

from app.database.models import (
    User,
    Message,
    TeacherClass,
    Class,
    Chunk,
)


def vector_search(query: str, n_results: int, where: dict) -> List[Chunk]:
    try:
        query_vector = get_embedding(query)
    except Exception as e:
        raise Exception(f"Failed to get embedding for query: {str(e)}")

    # Decode the where dict
    filters = []
    for key, value in where.items():
        if isinstance(value, list) and len(value) > 1:
            filters.append(getattr(Chunk, key).in_(value))
        elif isinstance(value, list) and len(value) == 1:
            filters.append(getattr(Chunk, key) == value[0])
        else:
            filters.append(getattr(Chunk, key) == value)

    with Session(db_engine) as session:
        try:
            result = session.execute(
                select(Chunk)
                .where(*filters)
                .order_by(Chunk.embedding.cosine_distance(query_vector))
                .limit(n_results)
            )
            return list(result.scalars().all())
        except Exception as e:
            raise Exception(f"Failed to search for knowledge: {str(e)}")


def get_or_create_user(wa_id: str, name: Optional[str] = None) -> User:
    """
    Get existing user or create new one if they don't exist.
    Handles all database operations and error logging.
    This uses a lot of eager loading to get the class information from the user.
    """
    with Session(db_engine) as session:
        try:
            # First try to get existing user
            statement = select(User).where(User.wa_id == wa_id).with_for_update()
            # First try to get existing user
            statement = (
                select(User)
                .where(User.wa_id == wa_id)
                .options(
                    selectinload(User.taught_classes)  # type: ignore
                    .selectinload(TeacherClass.class_)  # type: ignore
                    .selectinload(Class.subject_)  # type: ignore
                )
                .with_for_update()
            )
            result = session.exec(statement)
            user = result.one_or_none()
            # user = result.scalar_one_or_none()
            if user:
                session.refresh(user)
                return user
            # Create new user if they don't exist
            new_user = User(
                name=name,
                wa_id=wa_id,
                state="new",
                role="teacher",
            )
            session.add(new_user)
            session.flush()  # Get the ID without committing
            session.commit()
            session.refresh(new_user)
            return new_user
        except Exception as e:
            raise Exception(f"Failed to get or create user: {str(e)}")


def get_user_message_history(user_id: int, limit: int = 10) -> Optional[List[Message]]:
    with Session(db_engine) as session:
        try:
            statement = (
                select(Message)
                .where(Message.user_id == user_id)
                .order_by(desc(Message.created_at))
                .limit(limit)
            )

            result = session.exec(statement)
            messages = result.all()
            # messages = result.scalars().all()

            # If no messages found, return empty list
            if not messages:
                return None

            # Convert to list and reverse to get chronological order (oldest first)
            return list(reversed(messages))

        except Exception as e:
            raise Exception(f"Failed to retrieve message history: {str(e)}")


def create_new_messages(messages: List[Message]) -> List[Message]:
    """Optimized bulk message creation"""
    with Session(db_engine) as session:
        try:
            # Add all messages to the session
            session.add_all(messages)
            session.flush()  # Get IDs without committing
            return messages
        except Exception as e:
            raise Exception(f"Failed to create messages: {str(e)}")


def create_new_message(message: Message) -> Message:
    """
    Create a single message in the database.
    """
    with Session(db_engine) as session:
        try:
            # Add the message to the session
            session.add(message)

            # Commit the transaction
            session.commit()

            # Refresh the message to get its ID and other DB-populated fields
            session.refresh(message)

            return message

        except Exception as e:
            raise Exception(f"Failed to create message: {str(e)}")


def get_class_resources(class_id: int) -> Optional[List[int]]:
    """
    Get all resource IDs accessible to a class.
    Uses a single optimized SQL query with proper indexing.
    """
    with Session(db_engine) as session:
        try:
            # Use text() for a more efficient raw SQL query
            query = text(
                """
                SELECT DISTINCT resource_id
                FROM classes_resources
                WHERE class_id = :class_id
                """
            )

            result = session.exec(query, {"class_id": class_id})
            resource_ids = [row[0] for row in result.fetchall()]

            if not resource_ids:
                return None

            return resource_ids

        except Exception as e:
            raise Exception(f"Failed to get class resources: {str(e)}")
