#!/usr/bin/env python3
"""
Test R2E GeneralAgent with a single task to verify termination
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
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")


async def main():
    print("\n" + "="*80)
    print("🚀 R2E GeneralAgent Single Task Test")
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
        "r2e_submit": create_tool("R2ESubmit", {})
    }
    
    print(f"\n✅ Created {len(tools)} tools")
    
    # Create agent
    agent = GeneralAgent(
        max_rounds=5,
        debug=True,
        termination_tool_names=["r2e_submit"]
    )
    agent.set_tools(tools)
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=False  # Disable LLM debug for cleaner output
    )
    
    # Single task
    task = """
    Complete these steps:
    1. Run: pwd
    2. Run: echo "Task completed successfully"
    3. Submit
    """
    
    print(f"\n📝 Task: {task.strip()}")
    print("\n🎯 Executing single task...")
    print("="*80)
    
    try:
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id="single_task"
        )
        
        print("\n" + "="*80)
        print(f"✅ Task completed!")
        print(f"   Total steps: {len(trajectory.steps)}")
        print(f"   Is completed: {trajectory.is_completed}")
        print(f"   Final reward: {trajectory.final_reward}")
        
        # Check if r2e_submit was called
        submit_called = False
        for step in trajectory.steps:
            if hasattr(step, 'tool_name') and step.tool_name == 'r2e_submit':
                submit_called = True
                break
        
        print(f"   Submit called: {submit_called}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("🎉 Single task test completed!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())