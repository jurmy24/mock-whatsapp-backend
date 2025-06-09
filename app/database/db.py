from typing import Dict, List, Optional
from app.embeddings import get_embedding
from sqlmodel import Session, or_
from app.database.engine import db_engine
from app.database.models import Chunk, SubjectClassStatus
from sqlalchemy import text
from sqlmodel import select, desc, delete, exists, and_, insert
from sqlalchemy.orm import selectinload

from app.database.models import (
    User,
    Message,
    TeacherClass,
    Class,
    Chunk,
    Subject,
)


def update_user(user: User) -> User:
    """
    Update any information about an existing user and return the updated user.
    """
    with Session(db_engine) as session:
        try:
            # Add user to session and refresh to ensure we have latest data
            session.add(user)
            session.commit()
            session.refresh(user)
            print(f"Updated user {user.wa_id}: {user}")
            return user
        except Exception as e:
            print(f"Failed to update user {user.wa_id}: {str(e)}")
            raise Exception(f"Failed to update user: {str(e)}")


def get_class_ids_from_class_info(
    class_info: Dict[str, List[str]],
) -> Optional[List[int]]:
    """
    Get class IDs from a class_info dictionary structure in a single query

    Args:
        class_info: Dictionary mapping subject names to lists of grade levels
        Example: {"geography": ["os2"], "mathematics": ["os1", "os2"]}

    Returns:
        List of class IDs matching the subject-grade combinations
    """
    with Session(db_engine) as session:
        # Build conditions for each subject and its grade levels
        conditions = [
            and_(Subject.name == subject_name, Class.grade_level.in_(grade_levels))  # type: ignore
            for subject_name, grade_levels in class_info.items()
        ]

        query = (
            select(Class.id)
            .join(Subject, Class.subject_id == Subject.id)  # type: ignore
            .where(or_(*conditions), Class.status == SubjectClassStatus.active)
        )

        result = session.exec(query)
        class_ids = list(result.all())

        if not class_ids:
            print(f"No classes found for class info: {class_info}")
            return None

        return class_ids


def assign_teacher_to_classes(
    user: User, class_ids: List[int], subject_id: Optional[int] = None
):
    """
    Assign a teacher to a list of classes by creating teacher-class relationships.
    If subject_id is provided, only replaces classes with that subject_id.
    Otherwise replaces all teacher-class relationships.
    """
    with Session(db_engine) as session:
        try:
            # Construct delete query based on subject_id
            delete_query = delete(TeacherClass).where(
                TeacherClass.teacher_id == user.id  # type: ignore
            )

            if subject_id is not None:
                # Join with Class table to filter by subject_id
                delete_query = delete_query.where(
                    exists().where(
                        and_(
                            Class.id == TeacherClass.class_id,
                            Class.subject_id == subject_id,
                        )
                    )
                )

            # Delete existing relationships
            session.execute(delete_query)

            # Bulk insert new relationships
            if class_ids:
                values = [
                    {"teacher_id": user.id, "class_id": class_id}
                    for class_id in class_ids
                ]
                session.execute(insert(TeacherClass), values)
            else:
                print(f"No classes to assign for teacher {user.wa_id}")

            # Commit the transaction
            session.commit()

        except Exception as e:
            print(
                f"Failed to assign teacher {user.wa_id} to classes {class_ids}: {str(e)}"
            )
            raise Exception(f"Failed to assign teacher to classes: {str(e)}")


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

            result = session.execute(query, {"class_id": class_id})
            resource_ids = [row[0] for row in result.fetchall()]

            if not resource_ids:
                return None

            return resource_ids

        except Exception as e:
            raise Exception(f"Failed to get class resources: {str(e)}")
