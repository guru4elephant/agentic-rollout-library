#!/usr/bin/env python3
"""
Final demo: R2E GeneralAgent with K8S execution
Shows the complete working configuration with:
1. Removed tool examples (saving tokens)
2. Proper termination with r2e_submit
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
    print("🚀 R2E GeneralAgent Final Demo - K8S Execution")
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
    print(f"   Namespace: {k8s_config['namespace']}")
    
    # Create agent with proper termination
    agent = GeneralAgent(
        max_rounds=10,
        debug=True,
        termination_tool_names=["r2e_submit"]  # Properly configured termination
    )
    agent.set_tools(tools)
    
    print(f"\n✅ Agent configured")
    print(f"   Max rounds: {agent.max_rounds}")
    print(f"   Termination tools: {agent.termination_tool_names}")
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=True
    )
    
    print(f"\n🔗 LLM configured")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Endpoint: {BASE_URL}")
    
    # Task
    task = """
    In the K8S pod:
    1. Show current directory
    2. List Python files in the current directory
    3. Create a simple test file /tmp/r2e_demo.txt with content "R2E Agent Demo"
    4. Verify the file was created
    5. Submit when done
    """
    
    print(f"\n📝 Task: {task.strip()}")
    print("\n🎯 Executing...")
    print("-" * 60)
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id="r2e_final_demo"
        )
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        print("\n" + "-" * 60)
        print(f"✅ Task completed!")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Total steps: {len(trajectory.steps)}")
        print(f"   Completed properly: {trajectory.is_completed}")
        
        # Count actions
        action_count = sum(1 for step in trajectory.steps 
                          if hasattr(step, 'step_type') and step.step_type.value == 'action')
        print(f"   Actions taken: {action_count}")
        
        # Show action summary
        print("\n📊 Actions summary:")
        for step in trajectory.steps:
            if hasattr(step, 'step_type') and step.step_type.value == 'action' and hasattr(step, 'tool_name'):
                print(f"   - {step.tool_name}: {step.content[:60]}...")
        
        # Verify termination
        last_action = None
        for step in reversed(trajectory.steps):
            if hasattr(step, 'tool_name') and step.tool_name:
                last_action = step.tool_name
                break
        
        print(f"\n🏁 Last action: {last_action}")
        print(f"   Terminated properly: {last_action == 'r2e_submit'}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("🎉 Demo completed!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
