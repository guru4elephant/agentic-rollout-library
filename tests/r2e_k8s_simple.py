#!/usr/bin/env python3
"""
Simple R2E GeneralAgent with K8S - using default system prompt
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.agents.general_agent import GeneralAgent
from workers.core import create_tool
from workers.utils import create_llm_client

# LLM configuration
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = "gpt-4.1"


async def main():
    print("\n" + "="*80)
    print("🚀 R2E GeneralAgent - Using Default System Prompt")
    print("="*80)
    
    # K8S configuration
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
    
    print(f"\n✅ Created {len(tools)} R2E tools for K8S")
    print(f"   Pod: {k8s_config['pod_name']}")
    
    # Create agent with default system prompt
    agent = GeneralAgent(
        max_rounds=10,
        debug=True,  # Enable debug to see what's happening
        termination_tool_names=["r2e_submit"]  # Set r2e_submit as termination tool
    )
    agent.set_tools(tools)
    
    # Don't set custom system prompt - use default
    print("\n✅ Using GeneralAgent's default system prompt")
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=True
    )
    
    # Simple task
    task = """
    Show the current directory in the K8S pod and then submit.
    """
    
    print(f"\n📝 Task: {task.strip()}")
    print("\n🎯 Executing...")
    print("-" * 60)
    
    try:
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id="r2e_simple"
        )
        
        print("\n" + "-" * 60)
        print(f"✅ Completed! Steps: {len(trajectory.steps)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())