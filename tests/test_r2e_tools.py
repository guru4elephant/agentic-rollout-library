#!/usr/bin/env python3
"""
Test R2E tools implementation.
Tests both local and K8S execution modes.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.core import create_tool
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_r2e_tools_local():
    """Test R2E tools in local execution mode."""
    print("\n" + "="*80)
    print("🧪 Testing R2E Tools - Local Execution Mode")
    print("="*80)
    
    # Create R2E tools with local execution
    local_config = {"execution_mode": "local"}
    
    tools = {
        "r2e_bash": create_tool("R2EBashExecutor", local_config),
        "r2e_search": create_tool("R2ESearch", local_config),
        "r2e_file_editor": create_tool("R2EFileEditor", local_config),
        "r2e_str_replace": create_tool("R2EStrReplaceEditor", local_config),
        "r2e_submit": create_tool("R2ESubmit", {})
    }
    
    print(f"\n✅ Created {len(tools)} R2E tools in local mode")
    
    # Test 1: R2E Bash Executor
    print("\n📋 Test 1: R2E Bash Executor")
    print("-" * 40)
    result = await tools["r2e_bash"].execute_tool(
        "test_instance",
        {"command": "echo 'Hello from R2E Bash!'"}
    )
    print(f"Success: {result.success}")
    if result.success:
        print(f"Output:\n{result.result['output']}")
    else:
        print(f"Error: {result.error}")
    
    # Test blocked command
    print("\n📋 Test 1b: R2E Bash - Blocked Command")
    result = await tools["r2e_bash"].execute_tool(
        "test_instance",
        {"command": "git status"}
    )
    print(f"Success: {result.success}")
    print(f"Error (expected): {result.error}")
    
    # Test 2: R2E Search
    print("\n📋 Test 2: R2E Search Tool")
    print("-" * 40)
    # Create a test file
    test_file = "/tmp/r2e_test.py"
    with open(test_file, "w") as f:
        f.write("def hello():\n    print('Hello from R2E!')\n\nhello()\n")
    
    result = await tools["r2e_search"].execute_tool(
        "test_instance",
        {"search_term": "hello", "path": test_file}
    )
    print(f"Success: {result.success}")
    if result.success:
        print(f"Output:\n{result.result['output']}")
    
    # Test 3: R2E File Editor - View
    print("\n📋 Test 3: R2E File Editor - View")
    print("-" * 40)
    result = await tools["r2e_file_editor"].execute_tool(
        "test_instance",
        {"command": "view", "path": test_file}
    )
    print(f"Success: {result.success}")
    if result.success:
        print(f"Output:\n{result.result['output'][:200]}...")
    
    # Test 4: R2E File Editor - Create
    print("\n📋 Test 4: R2E File Editor - Create")
    print("-" * 40)
    new_file = "/tmp/r2e_new.py"
    result = await tools["r2e_file_editor"].execute_tool(
        "test_instance",
        {
            "command": "create",
            "path": new_file,
            "file_text": "# R2E Test File\nprint('Created by R2E!')\n"
        }
    )
    print(f"Success: {result.success}")
    if result.success:
        print(f"Created file: {result.result.get('path')}")
    
    # Test 5: R2E String Replace Editor
    print("\n📋 Test 5: R2E String Replace Editor")
    print("-" * 40)
    result = await tools["r2e_str_replace"].execute_tool(
        "test_instance",
        {
            "command": "str_replace",
            "path": new_file,
            "old_str": "print('Created by R2E!')",
            "new_str": "print('Modified by R2E String Replace!')"
        }
    )
    print(f"Success: {result.success}")
    if result.success:
        print("String replaced successfully")
    
    # Test 6: R2E Submit
    print("\n📋 Test 6: R2E Submit Tool")
    print("-" * 40)
    result = await tools["r2e_submit"].execute_tool(
        "test_instance",
        {}
    )
    print(f"Success: {result.success}")
    if result.success:
        print(f"Message: {result.result['message']}")
    
    # Cleanup
    try:
        os.remove(test_file)
        os.remove(new_file)
    except:
        pass
    
    print("\n✅ Local R2E tools test completed!")


async def test_r2e_tools_k8s():
    """Test R2E tools in K8S execution mode."""
    print("\n" + "="*80)
    print("🧪 Testing R2E Tools - K8S Execution Mode")
    print("="*80)
    
    # K8S configuration
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default"
    }
    
    try:
        tools = {
            "r2e_bash": create_tool("R2EBashExecutor", k8s_config.copy()),
            "r2e_search": create_tool("R2ESearch", k8s_config.copy()),
            "r2e_file_editor": create_tool("R2EFileEditor", k8s_config.copy()),
            "r2e_str_replace": create_tool("R2EStrReplaceEditor", k8s_config.copy()),
            "r2e_submit": create_tool("R2ESubmit", {})
        }
        
        print(f"\n✅ Created {len(tools)} R2E tools in K8S mode")
        print(f"   Pod: {k8s_config['pod_name']}")
        print(f"   Namespace: {k8s_config['namespace']}")
        
        # Test K8S bash executor
        print("\n📋 Test: R2E Bash Executor in K8S")
        print("-" * 40)
        result = await tools["r2e_bash"].execute_tool(
            "test_instance",
            {"command": "pwd && ls -la"}
        )
        print(f"Success: {result.success}")
        if result.success:
            print(f"Output:\n{result.result['output'][:500]}...")
        else:
            print(f"Error: {result.error}")
        
        print("\n✅ K8S R2E tools test completed!")
        
    except ImportError as e:
        print(f"❌ K8S tools not available: {e}")
        print("   Make sure kodo library is installed for K8S support")
    except Exception as e:
        print(f"❌ K8S test failed: {e}")


def display_tool_schemas():
    """Display R2E tool schemas."""
    print("\n" + "="*80)
    print("📚 R2E Tool Schemas")
    print("="*80)
    
    tool_names = [
        "R2EBashExecutor",
        "R2ESearch",
        "R2EFileEditor",
        "R2EStrReplaceEditor",
        "R2ESubmit"
    ]
    
    for tool_name in tool_names:
        try:
            tool = create_tool(tool_name, {})
            schema = tool.get_openai_tool_schema()
            
            print(f"\n🔧 {tool_name}")
            print("-" * 40)
            
            if hasattr(schema, 'model_dump'):
                schema_dict = schema.model_dump()
            else:
                schema_dict = schema.dict()
            
            func = schema_dict.get('function', {})
            print(f"Name: {func.get('name')}")
            print(f"Description: {func.get('description')}")
            
            params = func.get('parameters', {})
            if 'properties' in params:
                print("Parameters:")
                for param_name, param_def in params['properties'].items():
                    required = param_name in params.get('required', [])
                    req_str = " (required)" if required else " (optional)"
                    print(f"  - {param_name}{req_str}: {param_def.get('type')} - {param_def.get('description', '')[:60]}...")
                    if 'enum' in param_def:
                        print(f"    Allowed values: {param_def['enum']}")
        except Exception as e:
            print(f"\n❌ Failed to create {tool_name}: {e}")


async def main():
    """Main test function."""
    print("🚀 R2E Tools Test Suite")
    print("Testing R2E tools converted to official tool implementation")
    
    # Display schemas
    display_tool_schemas()
    
    # Test local execution
    await test_r2e_tools_local()
    
    # Test K8S execution (if available)
    await test_r2e_tools_k8s()
    
    print("\n" + "="*80)
    print("🎉 R2E Tools test suite completed!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())