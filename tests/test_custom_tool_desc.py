#!/usr/bin/env python3
"""
Demo: Custom tool descriptions and system prompts
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
    print("🚀 Custom Tool Descriptions Demo")
    print("="*80)
    
    # K8S configuration
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default"
    }
    
    # Create tools with custom description enabled for bash
    bash_config = k8s_config.copy()
    bash_config["use_custom_description"] = True
    
    tools = {
        "r2e_bash_executor": create_tool("R2EBashExecutor", bash_config),
        "r2e_file_editor": create_tool("R2EFileEditor", k8s_config.copy()),
        "r2e_submit": create_tool("R2ESubmit", {})
    }
    
    print(f"\n✅ Created {len(tools)} tools")
    
    # Get tool descriptions programmatically
    bash_desc = tools["r2e_bash_executor"].get_description()
    file_desc = tools["r2e_file_editor"].get_description()
    submit_desc = tools["r2e_submit"].get_description()
    
    # Custom system prompt that manually embeds tool descriptions
    custom_system_prompt = f"""You are a software engineer working with tools.

Available Tools:

1. Bash Executor Tool:
{bash_desc}

2. File Editor Tool (showing first 300 chars):
{file_desc[:300]}...

3. Submit Tool:
Use this to complete tasks.

IMPORTANT: Use XML format for all tool calls:
<function=tool_name>
<parameter=param>value</parameter>
</function>

Think step by step and use tools to complete the task."""
    
    # Create agent with custom prompt
    agent = GeneralAgent(
        max_rounds=3,
        debug=True,  # This will print the final system prompt
        termination_tool_names=["r2e_submit"],
        action_parser=parse_xml_action
    )
    agent.set_tools(tools)
    agent.system_prompt = custom_system_prompt
    
    print("\n📌 Key Features Demonstrated:")
    print("   - Custom system prompt with manual tool description embedding")
    print("   - Tool descriptions accessed via tool.get_description()")
    print("   - Debug mode prints the final system prompt")
    print("   - Custom XML action parser")
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=False
    )
    
    # Simple task
    task = """
    Show the current directory and then complete the task.
    """
    
    print(f"\n📝 Task: {task.strip()}")
    print("\n🎯 Starting execution...")
    print("="*80)
    
    try:
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id="custom_desc_demo"
        )
        
        print("\n" + "="*80)
        print("✅ Demo completed!")
        print(f"   Steps: {len(trajectory.steps)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())