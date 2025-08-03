#!/usr/bin/env python3
"""
Test R2E GeneralAgent with XML action format
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
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")


async def main():
    print("\n" + "="*80)
    print("🚀 R2E GeneralAgent with XML Action Format Test")
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
    
    # XML format system prompt
    xml_system_prompt = """You are a helpful assistant that uses tools to solve tasks.

Available tools:
- execute_bash: Run bash commands
- file_editor: Create/edit files
- finish: Complete the task

Use this EXACT format for tool calls:

<function=tool_name>
<parameter=param1>value1</parameter>
<parameter=param2>value2</parameter>
</function>

Example:
To run a command: 
<function=execute_bash>
<parameter=cmd>pwd</parameter>
</function>

To create a file:
<function=file_editor>
<parameter=command>create</parameter>
<parameter=path>/tmp/test.txt</parameter>
<parameter=file_text>Hello World</parameter>
</function>

To finish:
<function=finish>
<parameter=command>submit</parameter>
</function>

IMPORTANT: Each response must include reasoning text followed by a function call."""
    
    # Create agent with XML parser
    agent = GeneralAgent(
        max_rounds=5,
        debug=True,
        termination_tool_names=["r2e_submit"],
        action_parser=parse_xml_action
    )
    agent.set_tools(tools)
    agent.system_prompt = xml_system_prompt
    
    # Create LLM client
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=False
    )
    
    # Simple task
    task = """
    Complete these steps:
    1. Show current directory with pwd
    2. Create a file /tmp/xml_test.txt with content "XML parsing works!"
    3. Finish the task
    """
    
    print(f"\n📝 Task: {task.strip()}")
    print("\n🎯 Executing with XML format...")
    print("="*80)
    
    try:
        trajectory = await agent.run_trajectory(
            prompt=task,
            llm_generate_func=llm_client.generate,
            request_id="xml_test"
        )
        
        print("\n" + "="*80)
        print(f"✅ Task completed!")
        print(f"   Total steps: {len(trajectory.steps)}")
        
        # Show parsed actions
        print("\n📊 Actions executed:")
        for step in trajectory.steps:
            if hasattr(step, 'tool_name') and step.tool_name:
                print(f"   - {step.tool_name}")
                if hasattr(step, 'metadata') and step.metadata.get('xml_parsed'):
                    print(f"     (Parsed with XML parser)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())