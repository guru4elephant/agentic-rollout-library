#!/usr/bin/env python3
"""
Example: R2E GeneralAgent with K8S execution on swebench-xarray-pod.
This is a ready-to-use configuration with the custom LLM endpoint.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.agents.general_agent import GeneralAgent
from workers.core import create_tool
from workers.utils import create_llm_client

# LLM configuration
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = "gpt-4.1"


def create_r2e_agent_for_k8s():
    """Create a configured R2E GeneralAgent for K8S execution."""
    
    # K8S configuration for swebench-xarray-pod
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default"
    }
    
    # Create R2E tools
    tools = {
        "r2e_bash_executor": create_tool("R2EBashExecutor", k8s_config.copy()),
        "r2e_file_editor": create_tool("R2EFileEditor", k8s_config.copy()),
        "r2e_search": create_tool("R2ESearch", k8s_config.copy()),
        "r2e_submit": create_tool("R2ESubmit", {})
    }
    
    # Agent configuration - not used anymore, will pass params directly
    # agent_config = {
    #     "max_rounds": 15,
    #     "debug": False  # Set to True to see detailed LLM interactions
    # }
    
    # R2E-style system prompt
    custom_system_prompt = """You are an expert software engineer working with R2E tools in a Kubernetes pod.

You have access to these tools:
- r2e_bash_executor: Execute bash commands (blocks git, jupyter, etc.)
- r2e_file_editor: View/create/edit files (commands: view, create, str_replace, insert, undo_edit)
- r2e_search: Search in files and directories
- r2e_submit: Signal task completion

You're in pod 'swebench-xarray-pod' at /testbed.

IMPORTANT: Always use ReAct format with the EXACT JSON structure:

Thought: [Your reasoning here]
Action:
{
  "name": "r2e_bash_executor",
  "parameters": {
    "command": "pwd"
  }
}

Examples:
1. Bash command:
Action:
{
  "name": "r2e_bash_executor",
  "parameters": {
    "command": "ls -la"
  }
}

2. File editor:
Action:
{
  "name": "r2e_file_editor",
  "parameters": {
    "command": "view",
    "path": "/path/to/file"
  }
}

3. Search:
Action:
{
  "name": "r2e_search",
  "parameters": {
    "search_term": "def main",
    "path": "/testbed"
  }
}

4. Submit:
Action:
{
  "name": "r2e_submit",
  "parameters": {}
}

Complete tasks systematically, then use r2e_submit."""
    
    # Create agent
    agent = GeneralAgent(
        max_rounds=15,
        debug=False,  # Set to True to see detailed LLM interactions
        termination_tool_names=["r2e_submit"]  # Mark r2e_submit as termination tool
    )
    agent.set_tools(tools)
    agent.system_prompt = custom_system_prompt
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=False
    )
    
    return agent, llm_client


async def run_task(agent, llm_client, task, request_id="r2e_task"):
    """Run a task with the R2E agent."""
    
    print(f"\n📝 Task: {task.strip()}")
    print("\n🎯 Executing...")
    print("-" * 60)
    
    try:
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id=request_id
        )
        
        print("\n" + "-" * 60)
        print(f"✅ Task completed! Total steps: {len(trajectory.steps)}")
        
        # Show actions taken
        print("\n📊 Actions taken:")
        action_count = 0
        for step in trajectory.steps:
            if hasattr(step, 'step_type') and step.step_type.value == "action" and hasattr(step, 'tool_name') and step.tool_name:
                action_count += 1
                content_preview = step.content[:100] if hasattr(step, 'content') else "..."
                print(f"   {action_count}. {step.tool_name}: {content_preview}...")
        
        return trajectory
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


async def main():
    """Main example function."""
    
    print("\n" + "="*80)
    print("🚀 R2E GeneralAgent Example - K8S Execution")
    print("="*80)
    
    # Create agent and LLM client
    agent, llm_client = create_r2e_agent_for_k8s()
    print("\n✅ R2E Agent configured for K8S")
    print("   Pod: swebench-xarray-pod")
    print("   Model: gpt-4.1")
    
    # Example task 1: Simple exploration
    task1 = """
    Check the Python environment in the K8S pod:
    1. Show current directory and Python version
    2. List installed packages related to xarray
    3. Submit when done
    """
    
    await run_task(agent, llm_client, task1, "example_1")
    
    # Example task 2: File operations
    task2 = """
    Create a test file:
    1. Create /tmp/test_r2e.py with a simple hello world function
    2. View the file to confirm it was created correctly
    3. Submit when done
    """
    
    await run_task(agent, llm_client, task2, "example_2")
    
    print("\n" + "="*80)
    print("🎉 Examples completed!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())