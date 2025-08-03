#!/usr/bin/env python3
"""
Simple test to verify R2E GeneralAgent debug output
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
    print("🚀 R2E GeneralAgent Simple Debug Test")
    print("="*80)
    
    # K8S configuration
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default"
    }
    
    # Create minimal tools
    tools = {
        "r2e_bash_executor": create_tool("R2EBashExecutor", k8s_config.copy()),
        "r2e_submit": create_tool("R2ESubmit", {})
    }
    
    print(f"\n✅ Created {len(tools)} tools")
    
    # Create agent with debug enabled
    agent = GeneralAgent(
        max_rounds=3,  # Very limited rounds
        debug=True,  # Enable agent debug
        termination_tool_names=["r2e_submit"]
    )
    agent.set_tools(tools)
    
    # Create LLM client with debug enabled  
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=True  # Enable LLM debug
    )
    
    # Very simple task
    task = """
    Execute these two steps:
    1. Run the command: echo "Hello from K8S pod"
    2. Submit when done
    """
    
    print(f"\n📝 Task: {task.strip()}")
    print("\n🎯 Starting execution with full debug...")
    print("="*80)
    
    try:
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id="debug_test"
        )
        
        print("\n" + "="*80)
        print(f"✅ Task completed!")
        print(f"   Steps taken: {len(trajectory.steps)}")
        print(f"   Termination: {'r2e_submit called' if trajectory.is_completed else 'max rounds'}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())