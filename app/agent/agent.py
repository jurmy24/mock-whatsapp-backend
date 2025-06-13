# Define the ReAct Loop

# 1. User query input sent to LLM
# 2. LLM makes a THOUGHT on the first steps to do
# 3. Content from steps 1 and 2 is sent to the LLM to execute the first step
# 4. If it calls a tool (ACTION), the tool is executed and the result (OBSERVATION) is sent back to the LLM that decides whether to continue with more thoughts or to write a final response
# 5. The process repeats until the LLM decides to write a final response

from typing import Dict
from langchain_together import Together
from dotenv import load_dotenv

import os

load_dotenv()

max_steps = 10
together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
for i in range(max_steps):














https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/DataScienceAgent/Together_Open_DataScience_Agent.ipynb#scrollTo=gTJlwLVEBGgo

# TODO: See if this is needed and how it might be beneficial
# This function creates a comprehensive summary of execution result for the model's history.
def get_execution_summary(execution_result: Dict) -> str:
    """
    Create a comprehensive summary of execution result for the model's history.
    This gives the model better context about what happened during code execution.

    Args:
        execution_result: The result dictionary from run_python

    Returns:
        A summary of the execution including status, outputs, and any errors
    """
    if not execution_result:
        return "Execution failed - no result returned"

    # Check execution status
    status = execution_result.get("status", "unknown")
    summary_parts = [f"Execution status: {status}"]

    # Process outputs
    stdout_outputs = []
    display_outputs = []
    other_outputs = []

    if "outputs" in execution_result:
        for output in execution_result["outputs"]:
            output_type = output.get("type", "unknown")
            output_data = output.get("data", "")

            if output_type == "stdout":
                stdout_outputs.append(output_data)
            elif output_type == "display_data":
                if isinstance(output_data, dict):
                    if "image/png" in output_data:
                        display_outputs.append("Generated plot/image")
                    if "text/plain" in output_data:
                        display_outputs.append(f"Display: {output_data['text/plain']}")
                else:
                    display_outputs.append("Generated display output")
            else:
                other_outputs.append(f"{output_type}: {str(output_data)[:100]}")

    # Add stdout outputs
    if stdout_outputs:
        summary_parts.append("Text output:")
        summary_parts.extend(stdout_outputs)

    # Add display outputs (plots, images)
    if display_outputs:
        summary_parts.append("Visual outputs:")
        summary_parts.extend(display_outputs)

    # Add other outputs
    if other_outputs:
        summary_parts.append("Other outputs:")
        summary_parts.extend(other_outputs)

    # Check for errors
    if "errors" in execution_result and execution_result["errors"]:
        summary_parts.append("Errors:")
        summary_parts.extend(execution_result["errors"])

    # If no outputs at all but status is success
    if not stdout_outputs and not display_outputs and not other_outputs and status == "success":
        summary_parts.append("Code executed successfully (no explicit output generated)")

    return "\n".join(summary_parts)