#!/usr/bin/env python3
"""
Test R2E file operations in K8S
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
    print("🚀 R2E File Operations Test")
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
        "r2e_file_editor": create_tool("R2EFileEditor", k8s_config.copy()),
        "r2e_submit": create_tool("R2ESubmit", {})
    }
    
    print(f"\n✅ Created {len(tools)} tools")
    
    # Create agent
    agent = GeneralAgent(
        max_rounds=10,
        debug=True,
        termination_tool_names=["r2e_submit"]
    )
    agent.set_tools(tools)
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=False
    )
    
    # Simple file task
    task = """
    Test file operations:
    1. First check if /tmp exists with bash: ls -la /tmp | head -5
    2. Create a file /tmp/test.py with content: print("Hello World")
    3. View the file to confirm it was created
    4. Submit
    """
    
    print(f"\n📝 Task: {task.strip()}")
    print("\n🎯 Executing...")
    print("="*80)
    
    try:
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id="file_ops_test"
        )
        
        print("\n" + "="*80)
        print(f"✅ Task completed!")
        print(f"   Total steps: {len(trajectory.steps)}")
        
        # Show what happened
        print("\n📊 Actions taken:")
        for step in trajectory.steps:
            if hasattr(step, 'tool_name') and step.tool_name:
                print(f"   - {step.tool_name}")
                if step.tool_name == "r2e_file_editor" and hasattr(step, 'tool_args'):
                    print(f"     Command: {step.tool_args.get('command', 'N/A')}")
                    print(f"     Path: {step.tool_args.get('path', 'N/A')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())