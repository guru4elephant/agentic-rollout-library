#!/usr/bin/env python3
"""
Demo: R2E GeneralAgent with XML action format
Shows how custom action parsers work
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.agents.general_agent import GeneralAgent
from workers.core import create_tool
from workers.utils import create_llm_client
from workers.parsers.xml_action_parser import parse_xml_action

# LLM configuration
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = "gpt-4.1"


async def main():
    print("\n" + "="*80)
    print("🚀 R2E GeneralAgent with Custom XML Action Parser Demo")
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
    
    # System prompt that specifies XML format
    xml_system_prompt = """You are a helpful assistant that executes tasks using tools.

Available tools:
- execute_bash: Run bash commands
- finish: Complete the task

IMPORTANT: Use this EXACT XML format for tool calls:

<function=tool_name>
<parameter=param_name>param_value</parameter>
</function>

Example:
I need to check the current directory.

<function=execute_bash>
<parameter=cmd>pwd</parameter>
</function>

To complete a task:
<function=finish>
<parameter=command>submit</parameter>
</function>

Remember: Always include your reasoning before the function call."""
    
    # Create agent with XML parser
    agent = GeneralAgent(
        max_rounds=3,
        debug=True,  # Enable debug to see parsing
        termination_tool_names=["r2e_submit"],
        action_parser=parse_xml_action  # Custom XML parser
    )
    agent.set_tools(tools)
    agent.system_prompt = xml_system_prompt
    
    print("\n📌 Key Features Demonstrated:")
    print("   - Custom action parser (XML format)")
    print("   - Automatic tool name mapping")
    print("   - Debug output shows parsing details")
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=False
    )
    
    # Task
    task = """
    Execute these steps:
    1. Show the current working directory
    2. List files in /tmp (show first 5 files only)
    3. Complete the task
    """
    
    print(f"\n📝 Task: {task.strip()}")
    print("\n🎯 Starting execution...")
    print("="*80)
    
    try:
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id="xml_demo"
        )
        
        print("\n" + "="*80)
        print("🎉 Demo completed successfully!")
        print(f"\n📊 Summary:")
        print(f"   Total steps: {len(trajectory.steps)}")
        print(f"   Task completed: {trajectory.is_completed}")
        
        # Show which steps were parsed with XML
        xml_parsed_count = 0
        for step in trajectory.steps:
            if hasattr(step, 'metadata') and step.metadata.get('xml_parsed'):
                xml_parsed_count += 1
        
        print(f"   Steps parsed with XML: {xml_parsed_count}")
        
        print("\n💡 This demo shows how GeneralAgent can use custom action parsers")
        print("   to support different formats beyond the default JSON format.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())