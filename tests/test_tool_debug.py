#!/usr/bin/env python3
"""
Test tool execution with debug output
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
    print("🚀 Tool Execution Debug Test")
    print("="*80)
    
    # K8S configuration
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default"
    }
    
    # Create tools
    tools = {
        "r2e_bash_executor": create_tool("R2EBashExecutor", k8s_config.copy()),
        "r2e_search": create_tool("R2ESearch", k8s_config.copy()),
        "r2e_submit": create_tool("R2ESubmit", {})
    }
    
    print(f"\n✅ Created {len(tools)} tools")
    
    # Create agent with debug enabled
    agent = GeneralAgent(
        max_rounds=5,
        debug=True,  # Enable debug
        termination_tool_names=["r2e_submit"]
    )
    agent.set_tools(tools)
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=False  # Disable LLM debug to focus on tool debug
    )
    
    # Simple task that should trigger search
    task = """
    Use the search tool to find Python files in the current directory, then submit.
    """
    
    print(f"\n📝 Task: {task.strip()}")
    print("\n🎯 Executing with tool debug enabled...")
    print("-" * 60)
    
    try:
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id="tool_debug_test"
        )
        
        print("\n" + "-" * 60)
        print(f"✅ Completed! Steps: {len(trajectory.steps)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())