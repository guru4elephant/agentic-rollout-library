#!/usr/bin/env python3
"""
Test R2E termination tool handling
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
    print("🚀 Testing R2E Submit Termination")
    print("="*80)
    
    # Local configuration for simpler test
    local_config = {
        "execution_mode": "local"
    }
    
    # Create R2E tools
    tools = {
        "r2e_bash_executor": create_tool("R2EBashExecutor", local_config.copy()),
        "r2e_submit": create_tool("R2ESubmit", {})
    }
    
    print(f"\n✅ Created {len(tools)} R2E tools")
    
    # Create agent with termination tool
    agent = GeneralAgent(
        max_rounds=10,  # More rounds to see if it continues after submit
        debug=False,
        termination_tool_names=["r2e_submit"]  # Mark r2e_submit as termination tool
    )
    agent.set_tools(tools)
    
    print(f"✅ Termination tools: {agent.termination_tool_names}")
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=False
    )
    
    # Simple task
    task = """
    Execute 'echo Hello' and then immediately submit to finish.
    """
    
    print(f"\n📝 Task: {task.strip()}")
    print("\n🎯 Starting execution...")
    print("-" * 60)
    
    try:
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id="test_termination"
        )
        
        print("\n" + "-" * 60)
        print(f"✅ Completed!")
        print(f"   Total steps: {len(trajectory.steps)}")
        print(f"   Is completed: {trajectory.is_completed}")
        
        # Show all steps
        print("\n📊 Trajectory steps:")
        for i, step in enumerate(trajectory.steps):
            print(f"   {i+1}. {step.step_type.value}: {step.content[:50]}...")
            if hasattr(step, 'tool_name') and step.tool_name:
                print(f"      Tool: {step.tool_name}")
        
        # Check if submit was called
        submit_called = any(
            hasattr(step, 'tool_name') and step.tool_name == 'r2e_submit' 
            for step in trajectory.steps
        )
        print(f"\n🔍 Submit tool called: {submit_called}")
        
        # Check last few steps
        print("\n🔍 Last 3 steps:")
        for step in trajectory.steps[-3:]:
            print(f"   - {step.step_type.value}: {step.content[:80]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())