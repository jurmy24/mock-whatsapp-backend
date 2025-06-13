from typing import Optional
import textwrap

box_width = 80  # Width of output boxes


def box_text(
    text: str, title: Optional[str] = None, emoji: Optional[str] = None
) -> str:
    """Create a boxed text with optional title and emoji."""
    # Handle None or empty text
    if not text:
        text = "No output"

    # Limit to 500 words for readability
    words = text.split()
    if len(words) > 500:
        words = words[:500]
        words.append("...")
        text = " ".join(words)

    # Wrap text at specified width
    wrapped_lines = []
    for line in text.split("\n"):
        if len(line) > box_width:
            wrapped_lines.extend(textwrap.wrap(line, width=box_width))
        else:
            wrapped_lines.append(line)

    # Handle empty wrapped_lines
    if not wrapped_lines:
        wrapped_lines = ["No output"]

    width = max(len(line) for line in wrapped_lines)
    width = max(width, len(title) if title else 0)

    if title and emoji:
        title = f" {emoji} {title} "
    elif title:
        title = f" {title} "
    elif emoji:
        title = f" {emoji} "

    result = []
    if title:
        result.append(f"╔{'═' * (width + 2)}╗")
        result.append(f"║ {title}{' ' * (width - len(title) + 2)}║")
        result.append(f"╠{'═' * (width + 2)}╣")
    else:
        result.append(f"╔{'═' * (width + 2)}╗")

    for line in wrapped_lines:
        result.append(f"║ {line}{' ' * (width - len(line))} ║")

    result.append(f"╚{'═' * (width + 2)}╝")
    return "\n".join(result)
