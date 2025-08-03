#!/usr/bin/env python3
"""
Test the system prompt generation with K8S tools
"""

import sys
import os

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.core import create_tool, create_agent

def test_system_prompt_with_k8s_tools():
    """Test system prompt generation with K8S tools"""
    print("🔧 Testing System Prompt with K8S Tools")
    print("=" * 80)
    
    # K8S configuration
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default",
        "timeout": 30
    }
    
    # Create tools
    print("📋 Step 1: Creating K8S tools...")
    tools = {
        "bash_executor": create_tool("BashExecutor", k8s_config.copy()),
        "file_editor": create_tool("FileEditor", k8s_config.copy()),
        "search": create_tool("Search", k8s_config.copy()),
        "finish": create_tool("Finish")
    }
    print(f"   ✅ Created {len(tools)} tools")
    
    # Create agent with custom system prompt
    print("\n📋 Step 2: Creating GeneralAgent...")
    custom_prompt = """You are an AI assistant working in a Kubernetes environment.
You have access to various tools to help complete tasks."""
    
    agent = create_agent("General", {
        "max_rounds": 5,
        "system_prompt": custom_prompt,
        "termination_tool_names": ["finish"]
    })
    
    # Set tools
    print("\n📋 Step 3: Setting tools...")
    agent.set_tools(tools)
    
    # Generate system prompt
    print("\n📋 Step 4: Generating system prompt...")
    full_prompt = agent.create_system_prompt()
    
    # Save to file for easier reading
    output_file = "test_system_prompt_output.txt"
    with open(output_file, 'w') as f:
        f.write(full_prompt)
    
    print(f"\n✅ System prompt saved to: {output_file}")
    print(f"   Length: {len(full_prompt):,} characters")
    print(f"   Lines: {len(full_prompt.splitlines())} lines")
    
    # Check for key elements
    print("\n📊 Checking system prompt elements:")
    checks = [
        ("Custom prompt included", custom_prompt in full_prompt),
        ("ReAct format", "Thought:" in full_prompt and "Action:" in full_prompt),
        ("Tool documentation", "Available Tools" in full_prompt),
        ("bash_executor schema", '"name": "bash_executor"' in full_prompt),
        ("file_editor schema", '"name": "file_editor"' in full_prompt),
        ("search schema", '"name": "search"' in full_prompt),
        ("finish schema", '"name": "finish"' in full_prompt),
        ("Parameter details", '"command"' in full_prompt and '"path"' in full_prompt),
        ("Required fields", '"required"' in full_prompt),
        ("K8S execution mode", "executing in k8s mode" in full_prompt)
    ]
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
    
    # Show a snippet of the tool documentation
    print("\n📝 Tool documentation snippet:")
    lines = full_prompt.splitlines()
    in_tools_section = False
    tool_lines = []
    
    for i, line in enumerate(lines):
        if "Available Tools" in line:
            in_tools_section = True
        if in_tools_section:
            tool_lines.append(line)
            if len(tool_lines) > 50:  # Show first 50 lines of tools section
                break
    
    if tool_lines:
        print("\n".join(tool_lines[:30]) + "\n...")
    
    return full_prompt


if __name__ == "__main__":
    test_system_prompt_with_k8s_tools()