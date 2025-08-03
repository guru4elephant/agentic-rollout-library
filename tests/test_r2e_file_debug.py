#!/usr/bin/env python3
"""
Debug R2E file editor K8S issues
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.core import create_tool


async def main():
    print("\n" + "="*80)
    print("🔍 R2E File Editor Debug Test")
    print("="*80)
    
    # K8S configuration
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default"
    }
    
    # Create file editor tool
    file_editor = create_tool("R2EFileEditor", k8s_config)
    
    print("\n📝 Test 1: Create a simple file")
    result = await file_editor.execute_tool(
        instance_id="test1",
        parameters={
            "command": "create",
            "path": "/tmp/debug_test.txt",
            "file_text": "Hello from debug test"
        }
    )
    print(f"Success: {result.success}")
    if result.success:
        print(f"Result: {result.result}")
    else:
        print(f"Error: {result.error}")
    
    print("\n📝 Test 2: View the file")
    result = await file_editor.execute_tool(
        instance_id="test2",
        parameters={
            "command": "view",
            "path": "/tmp/debug_test.txt"
        }
    )
    print(f"Success: {result.success}")
    if result.success:
        print(f"Result: {result.result.get('output', '')[:200]}")
    else:
        print(f"Error: {result.error}")
    
    print("\n📝 Test 3: Check with bash")
    bash = create_tool("R2EBashExecutor", k8s_config)
    result = await bash.execute_tool(
        instance_id="test3",
        parameters={
            "command": "ls -la /tmp/debug_test.txt && cat /tmp/debug_test.txt"
        }
    )
    print(f"Success: {result.success}")
    print(f"Output: {result.result}")


if __name__ == "__main__":
    asyncio.run(main())