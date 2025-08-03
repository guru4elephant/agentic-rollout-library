#!/usr/bin/env python3
"""
Simple test for GeneralAgent with R2E tools in K8S.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.agents.general_agent import GeneralAgent
from workers.core import create_tool
from workers.utils import create_llm_client
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Create and display R2E GeneralAgent configuration for K8S."""
    
    print("\n" + "="*80)
    print("🚀 R2E GeneralAgent Configuration for K8S")
    print("="*80)
    
    # K8S configuration for swebench-xarray-pod
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default"
    }
    
    # Create R2E tools with K8S configuration
    print("\n📦 Creating R2E tools for K8S execution...")
    tools = {
        "r2e_bash_executor": create_tool("R2EBashExecutor", k8s_config.copy()),
        "r2e_file_editor": create_tool("R2EFileEditor", k8s_config.copy()),
        "r2e_search": create_tool("R2ESearch", k8s_config.copy()),
        "r2e_submit": create_tool("R2ESubmit", {})
    }
    
    print(f"✅ Created {len(tools)} R2E tools")
    print(f"   Pod: {k8s_config['pod_name']}")
    print(f"   Namespace: {k8s_config['namespace']}")
    
    # Display tool schemas
    print("\n📋 Tool Configurations:")
    print("-" * 40)
    for name, tool in tools.items():
        schema = tool.get_openai_tool_schema()
        if hasattr(schema, 'model_dump'):
            schema_dict = schema.model_dump()
        else:
            schema_dict = schema.dict()
        func = schema_dict.get('function', {})
        print(f"\n🔧 {name}:")
        print(f"   Name: {func.get('name')}")
        print(f"   Description: {func.get('description', '')[:80]}...")
        
        # Show execution info if available
        if hasattr(tool, 'get_execution_info'):
            exec_info = tool.get_execution_info()
            print(f"   Execution Mode: {exec_info.get('execution_mode', 'local')}")
            if exec_info.get('execution_mode') == 'k8s':
                print(f"   Pod: {exec_info.get('pod_name')}")
                print(f"   Namespace: {exec_info.get('namespace')}")
    
    # Create GeneralAgent
    print("\n" + "="*80)
    print("🤖 Creating GeneralAgent with R2E tools...")
    print("="*80)
    
    # Agent configuration
    agent_config = {
        "max_rounds": 15,
        "debug": False
    }
    
    # Custom system prompt for R2E-style agent
    custom_system_prompt = """You are an expert software engineer working with R2E tools in a Kubernetes pod.

You have access to the following tools:
- r2e_bash_executor: Execute bash commands in the K8S pod (blocks dangerous commands like git)
- r2e_file_editor: View, create, and edit files in the K8S pod (supports str_replace, insert, undo)
- r2e_search: Search for terms in files and directories in the K8S pod
- r2e_submit: Signal task completion

You are working in pod 'swebench-xarray-pod' in the 'default' namespace.
All file operations and commands execute inside this pod at /testbed.

Use the ReAct framework for your responses:
- Thought: Analyze what needs to be done
- Action: Execute a tool with JSON parameters
- Observation: Review the tool output

When your task is complete, use r2e_submit to signal completion."""
    
    # Create agent instance
    agent = GeneralAgent(config=agent_config)
    agent.set_tools(tools)
    agent.system_prompt = custom_system_prompt
    
    print("✅ GeneralAgent created successfully")
    print(f"   Max rounds: {agent.max_rounds}")
    print(f"   Tools configured: {len(agent.tools)}")
    print(f"   System prompt length: {len(agent.system_prompt)} chars")
    
    # Test a simple command
    print("\n" + "="*80)
    print("🧪 Testing R2E Bash Executor in K8S")
    print("="*80)
    
    test_tool = tools["r2e_bash_executor"]
    result = await test_tool.execute_tool(
        "test_instance",
        {"command": "pwd && ls -la | head -5"}
    )
    
    print(f"\n✅ Test Result:")
    print(f"   Success: {result.success}")
    if result.success:
        output = result.result.get('output', '')
        print(f"   Output:\n{output[:500]}")
    else:
        print(f"   Error: {result.error}")
    
    print("\n" + "="*80)
    print("🎉 R2E GeneralAgent configuration complete!")
    print("="*80)
    
    print("\n📝 Summary:")
    print("   - GeneralAgent configured with 4 R2E tools")
    print(f"   - K8S execution on pod: {k8s_config['pod_name']}")
    print("   - Tools support both viewing and editing files")
    print("   - Bash executor blocks dangerous commands (git, jupyter, etc.)")
    print("   - Ready for ReAct-style task execution")
    
    print("\n💡 To use this agent with an LLM:")
    print("   1. Create an LLM client with your API credentials")
    print("   2. Call agent.run_trajectory(prompt, llm_client.generate, request_id)")
    print("   3. The agent will use R2E tools to complete tasks in the K8S pod")


if __name__ == "__main__":
    asyncio.run(main())