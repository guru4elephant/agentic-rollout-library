#!/usr/bin/env python3
"""
Demo of R2E GeneralAgent with K8S execution on swebench-xarray-pod.
Uses the provided custom LLM endpoint.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.agents.general_agent import GeneralAgent
from workers.core import create_tool
from workers.utils import create_llm_client
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM configuration
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")


async def main():
    """Demo R2E GeneralAgent with K8S execution."""
    
    print("\n" + "="*80)
    print("🚀 R2E GeneralAgent Demo - K8S Execution on swebench-xarray-pod")
    print("="*80)
    
    # K8S configuration for swebench-xarray-pod
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default"
    }
    
    # Create R2E tools
    print("\n📦 Creating R2E tools for K8S...")
    tools = {
        "r2e_bash_executor": create_tool("R2EBashExecutor", k8s_config.copy()),
        "r2e_file_editor": create_tool("R2EFileEditor", k8s_config.copy()),
        "r2e_search": create_tool("R2ESearch", k8s_config.copy()),
        "r2e_submit": create_tool("R2ESubmit", {})
    }
    
    print(f"✅ Created {len(tools)} R2E tools")
    print(f"   Pod: {k8s_config['pod_name']}")
    print(f"   Namespace: {k8s_config['namespace']}")
    
    # Create GeneralAgent
    print("\n🤖 Creating GeneralAgent...")
    
    agent_config = {
        "max_rounds": 10,
        "debug": True  # Enable debug to see LLM interactions
    }
    
    # R2E-style system prompt
    custom_system_prompt = """You are an expert software engineer working with R2E tools in a Kubernetes pod.

You have access to these tools:
- r2e_bash_executor: Execute bash commands (blocks git, jupyter, etc.)
- r2e_file_editor: View/create/edit files (commands: view, create, str_replace, insert, undo_edit)
- r2e_search: Search in files and directories
- r2e_submit: Signal task completion

You're in pod 'swebench-xarray-pod' at /testbed.

Always use ReAct format:
- Thought: Analyze the task
- Action: {"tool": "tool_name", "parameters": {...}}
- Observation: Tool output

Complete tasks systematically, then use r2e_submit."""
    
    agent = GeneralAgent(config=agent_config)
    agent.set_tools(tools)
    agent.system_prompt = custom_system_prompt
    
    print("✅ Agent configured")
    
    # Create LLM client
    print(f"\n🔗 Connecting to LLM...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Endpoint: {BASE_URL}")
    
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=True
    )
    
    # Demo task
    print("\n" + "="*80)
    print("📝 Demo Task: Explore XArray Repository")
    print("="*80)
    
    task = """
    Explore the xarray repository in the K8S pod:
    1. Show the current directory and list main files/folders
    2. Check if there's a setup.py or pyproject.toml file
    3. Look for the main xarray module directory
    4. Create a simple test file /tmp/xarray_info.txt with what you found
    5. Submit when done
    """
    
    print(f"Task: {task.strip()}")
    print("\n🎯 Executing task...")
    print("-" * 60)
    
    try:
        # Run the agent
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id="r2e_k8s_demo"
        )
        
        print("\n" + "-" * 60)
        print("✅ Task completed!")
        print(f"   Total steps: {len(trajectory.steps)}")
        print(f"   Terminated: {'Yes' if not agent.should_continue(trajectory) else 'No'}")
        
        # Show summary of actions taken
        print("\n📊 Actions Summary:")
        action_count = 0
        for step in trajectory.steps:
            if step.step_type.value == "action" and step.tool_name:
                action_count += 1
                print(f"   {action_count}. {step.tool_name}: {step.tool_input.get('command', step.tool_input.get('search_term', 'action'))[:50]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("🎉 Demo completed!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())