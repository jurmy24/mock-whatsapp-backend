from pathlib import Path
from typing import Dict


# EXAMPLE TOOL
def collect_files(directory) -> list[Dict[str, str]]:
    """
    Collects all files from the specified directory and its subdirectories.

    Args:
        directory: The directory to scan for files

    Returns:
        A list of file dictionaries ready for upload to the code interpreter
    """
    files = []
    path = Path(directory)

    if not path.exists():
        print(f"Directory '{directory}' does not exist, skipping file collection")
        return files

    for file_path in Path(directory).rglob("*"):
        if file_path.is_file() and not any(
            part.startswith(".") for part in file_path.parts
        ):
            try:
                # Handle different file types
                if file_path.suffix.lower() in [".csv", ".txt", ".json", ".py"]:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    files.append(
                        {
                            "name": str(file_path.relative_to(directory)),
                            "encoding": "string",
                            "content": content,
                        }
                    )
                elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                    # For Excel files, we'll let pandas handle them in the code
                    print(
                        f"Excel file detected: {file_path.name} - will be handled by pandas"
                    )

            except (UnicodeDecodeError, PermissionError) as e:
                print(f"Could not read file {file_path}: {e}")

    return files
