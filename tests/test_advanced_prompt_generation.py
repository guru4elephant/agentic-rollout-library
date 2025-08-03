#!/usr/bin/env python3
"""
Advanced example showing dynamic system prompt generation with variables.
This demonstrates how to create flexible, context-aware prompts for different scenarios.
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.agents.general_agent import GeneralAgent
from workers.core import create_tool
from workers.utils import create_llm_client


def generate_swe_bench_prompt(tools, **kwargs):
    """Generate prompt for SWE-bench style tasks."""
    issue_description = kwargs.get('issue_description', 'No issue description provided')
    repo_name = kwargs.get('repo_name', 'unknown')
    working_directory = kwargs.get('working_directory', '/testbed')
    test_command = kwargs.get('test_command', 'pytest')
    
    return f"""You are an expert software engineer working on the {repo_name} repository.

Issue Description:
{issue_description}

Environment:
- Working Directory: {working_directory}
- Test Command: {test_command}
- Repository: {repo_name}

Available Tools:
{tools['bash_executor'].get_description()}
{tools['file_editor'].get_description()}
{tools['search'].get_description()}
{tools['finish'].get_description()}

Instructions:
1. First, explore the repository structure to understand the codebase
2. Locate the relevant files mentioned in the issue
3. Analyze the problem and develop a solution
4. Implement the fix
5. Run tests to verify your solution
6. Submit your solution when complete

Use tools in JSON format:
{{
  "name": "tool_name",
  "parameters": {{...}}
}}
"""


def generate_code_review_prompt(tools, **kwargs):
    """Generate prompt for code review tasks."""
    file_path = kwargs.get('file_path', 'file.py')
    review_focus = kwargs.get('review_focus', ['bugs', 'performance', 'style'])
    language = kwargs.get('language', 'Python')
    
    focus_list = '\n'.join(f"- {focus}" for focus in review_focus)
    
    return f"""You are a senior {language} developer conducting a code review.

File to Review: {file_path}
Language: {language}

Review Focus Areas:
{focus_list}

Available Tools:
{tools['file_editor'].get_description()}
{tools['search'].get_description()}
{tools['finish'].get_description()}

Review Process:
1. Read and understand the entire file
2. Analyze for issues in the focus areas
3. Provide specific, actionable feedback
4. Suggest improvements with code examples
5. Submit your review when complete

Use tools in JSON format:
{{
  "name": "tool_name",
  "parameters": {{...}}
}}
"""


def generate_debugging_prompt(tools, **kwargs):
    """Generate prompt for debugging tasks."""
    error_message = kwargs.get('error_message', 'Unknown error')
    stack_trace = kwargs.get('stack_trace', 'No stack trace available')
    test_file = kwargs.get('test_file', None)
    context = kwargs.get('context', {})
    
    context_str = '\n'.join(f"- {k}: {v}" for k, v in context.items())
    
    return f"""You are debugging an error in the codebase.

Error Message:
{error_message}

Stack Trace:
{stack_trace}

{'Test File: ' + test_file if test_file else ''}

Additional Context:
{context_str}

Available Tools:
{tools['bash_executor'].get_description()}
{tools['file_editor'].get_description()}
{tools['search'].get_description()}
{tools['finish'].get_description()}

Debugging Strategy:
1. Analyze the error message and stack trace
2. Locate the source of the error
3. Understand the code flow leading to the error
4. Identify the root cause
5. Implement and test a fix
6. Verify the fix resolves the issue

Use tools in JSON format:
{{
  "name": "tool_name",
  "parameters": {{...}}
}}
"""


async def main():
    """Demonstrate different prompt generation scenarios."""
    print("\n" + "="*80)
    print("🚀 Advanced System Prompt Generation Demo")
    print("="*80)
    
    # Create tools
    tools = {
        "bash_executor": create_tool("BashExecutor"),
        "file_editor": create_tool("FileEditor"),
        "search": create_tool("Search"),
        "finish": create_tool("Finish")
    }
    
    # Example 1: SWE-bench style task
    print("\n📝 Example 1: SWE-bench Style Task")
    print("-"*40)
    
    swe_prompt = generate_swe_bench_prompt(
        tools,
        issue_description="The DataFrame.merge() function fails when merging on datetime columns with different timezones",
        repo_name="pandas",
        working_directory="/testbed/pandas",
        test_command="pytest tests/test_merge.py"
    )
    
    print("Generated prompt (first 500 chars):")
    print(swe_prompt[:500] + "...")
    
    # Example 2: Code review task
    print("\n\n📝 Example 2: Code Review Task")
    print("-"*40)
    
    review_prompt = generate_code_review_prompt(
        tools,
        file_path="/src/utils/data_processor.py",
        review_focus=['security', 'performance', 'error handling'],
        language="Python"
    )
    
    print("Generated prompt (first 500 chars):")
    print(review_prompt[:500] + "...")
    
    # Example 3: Debugging task
    print("\n\n📝 Example 3: Debugging Task")
    print("-"*40)
    
    debug_prompt = generate_debugging_prompt(
        tools,
        error_message="AttributeError: 'NoneType' object has no attribute 'split'",
        stack_trace="""Traceback (most recent call last):
  File "app.py", line 45, in process_data
    parts = result.split(',')
AttributeError: 'NoneType' object has no attribute 'split'""",
        test_file="tests/test_processor.py",
        context={
            "Function": "process_data",
            "Input type": "CSV string",
            "Last working version": "v2.1.0"
        }
    )
    
    print("Generated prompt (first 500 chars):")
    print(debug_prompt[:500] + "...")
    
    # Example 4: Dynamic prompt updates during execution
    print("\n\n📝 Example 4: Dynamic Prompt Updates")
    print("-"*40)
    
    agent = GeneralAgent(
        max_rounds=5,
        debug=False,
        termination_tool_names=["finish"]
    )
    agent.set_tools(tools)
    
    # Initial prompt for exploration
    exploration_prompt = f"""You are exploring a new codebase.
Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{tools['bash_executor'].get_description()}
{tools['file_editor'].get_description()}

Task: List the contents of the current directory and finish.

Use JSON format for tool calls."""
    
    agent.system_prompt = exploration_prompt
    print(f"Initial prompt set at {datetime.now().strftime('%H:%M:%S')}")
    
    # Simulate dynamic update (would normally happen based on results)
    await asyncio.sleep(1)
    
    # Update prompt with new context
    analysis_prompt = f"""You are analyzing code quality.
Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Previous findings: Found 3 Python files in src/

{tools['file_editor'].get_description()}
{tools['search'].get_description()}

Task: Search for TODO comments in Python files.

Use JSON format for tool calls."""
    
    agent.system_prompt = analysis_prompt
    print(f"Updated prompt set at {datetime.now().strftime('%H:%M:%S')}")
    
    print("\n" + "="*80)
    print("✅ Demo completed!")
    print("\nKey Takeaways:")
    print("1. Use functions to generate prompts with dynamic variables")
    print("2. Different tasks require different prompt structures")
    print("3. Prompts can be updated during execution based on context")
    print("4. Include relevant context (time, environment, previous results)")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())