#!/usr/bin/env python3
"""
R2E Agent K8S Example - R2E-style tool execution in Kubernetes pods.

This example demonstrates an R2E-style agent that executes commands
in Kubernetes pods using the R2E prompt format and tools (bash, file_editor, search, finish).
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent / "src"))

from core import (
    ContextEngineeringNode,
    LLMNode,
    ToolParsingNode,
    K8SToolExecutionNode,
    K8S_AVAILABLE,
    get_timeline
)
from utils import create_openai_api_handle_async
from r2e_configs import (
    CUSTOM_TOOL_DESCRIPTIONS,
    parse_xml_action_custom,
    SYSTEM_PROMPT_TEMPLATE,
    QUERY_PROMPT_TEMPLATE,
    DEFAULT_TEMPLATE_VARIABLES
)

def create_r2e_parser():
    """
    Create R2E-style XML parser for tool calls.
    Uses parse_xml_action_custom from r2e_configs.
    """
    def parse_tool_calls(llm_response: Dict) -> List[Dict]:
        content = llm_response.get("content", "")

        # Use the custom XML parser from r2e_configs
        parsed = parse_xml_action_custom(content)

        if parsed is None:
            return []

        # Convert to our internal tool call format
        tool_call = {
            "tool": parsed["tool_name"],
            "parameters": parsed["tool_args"]
        }

        # Store thought content if present
        if parsed.get("has_thought"):
            tool_call["thought"] = parsed["thought_content"]

        return [tool_call]

    return parse_tool_calls


async def process_single_instance(
    instance_data: Dict,
    pod_suffix: str,
    enable_timeline: bool = False
) -> Dict:
    """Process a single instance from the JSONL file.

    Args:
        instance_data: Data for a single instance from JSONL
        pod_suffix: Unique suffix for the pod name
        enable_timeline: Enable timeline tracking

    Returns:
        Result dictionary with instance_id and execution status
    """
    instance_id = instance_data.get("instance_id", "unknown")
    print(f"\n🚀 Processing instance: {instance_id} (pod: r2e-agent-{pod_suffix})")

    result = {
        "instance_id": instance_id,
        "status": "failed",
        "error": None
    }

    try:
        # Context Engineering Node
        context = ContextEngineeringNode(name=f"R2EK8SContext-{pod_suffix}", timeline_enabled=enable_timeline)

        # LLM Node (async)
        llm_handle = create_openai_api_handle_async(
            base_url="http://211.23.3.237:27544/v1",
            api_key="sk-qq7xJtnAdB1Gv6IkHTQhDAPuUAT700vF3CMmGinILsmP2HuY",
            model="deepseek-v3-1-terminus"
        )

        llm_node = LLMNode(
            name=f"R2ELLM-{pod_suffix}",
            function_handle=llm_handle,
            model_config={
                "temperature": 0.7,
                "max_tokens": 4000
            },
            timeline_enabled=enable_timeline
        )
        llm_node.set_retry_config(max_retries=3)

        # Tool Parsing Node (R2E-style XML parser)
        parser = ToolParsingNode(
            name=f"R2EParser-{pod_suffix}",
            parse_function=create_r2e_parser(),
            timeline_enabled=enable_timeline
        )

        # Use the image from the instance data if provided
        image = instance_data.get("image", "python:3.11-slim")

        # Use context manager for automatic cleanup
        with K8SToolExecutionNode(
            name=f"R2EK8SExecutor-{pod_suffix}",
            namespace="default",
            image=image,
            pod_name=f"r2e-agent-{pod_suffix}",
            environment={
                "PYTHONUNBUFFERED": "1"
            },
            timeline_enabled=enable_timeline
        ) as k8s_executor:

            # Register R2E tools
            k8s_executor.register_tool(
                "r2e_bash_executor",
                "src/tools/r2e/bash_func.py"
            )

            k8s_executor.register_tool(
                "r2e_file_editor",
                "src/tools/r2e/file_editor.py"
            )

            k8s_executor.register_tool(
                "r2e_search",
                "src/tools/r2e/search_func.py"
            )

            # Finish/Submit tool
            def finish_parse_result(result):
                """Parse finish tool result."""
                if isinstance(result, dict):
                    return result
                return {
                    "message": "<<<Finish>>>",
                    "status": "stop"
                }

            k8s_executor.register_tool(
                "r2e_submit",
                "src/tools/r2e/finish.py",
                finish_parse_result
            )

            # Build system prompt with tool descriptions
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                r2e_file_editor=CUSTOM_TOOL_DESCRIPTIONS['r2e_file_editor'],
                r2e_bash_executor=CUSTOM_TOOL_DESCRIPTIONS['r2e_bash_executor'],
                r2e_search=CUSTOM_TOOL_DESCRIPTIONS['r2e_search'],
                r2e_submit=CUSTOM_TOOL_DESCRIPTIONS['r2e_submit'],
                **DEFAULT_TEMPLATE_VARIABLES
            )

            # Add system prompt message
            context.add_message(
                message_content=system_prompt,
                message_role="system",
                message_type="system_prompt"
            )

            # Get problem statement from instance data
            issue = instance_data.get("problem_statement", "No problem statement provided")

            # Build and add user query
            query = QUERY_PROMPT_TEMPLATE.format(issue=issue)
            context.add_message(
                message_content=query,
                message_role="user",
                message_type="query"
            )

            max_iterations = 10
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                messages = context.get_llm_context()
                try:
                    if enable_timeline:
                        llm_response = await llm_node.process_with_timing(messages, event_type="llm_call")
                        tool_calls = await parser.process_with_timing(llm_response, event_type="parse")
                    else:
                        llm_response = await llm_node.process_async(messages)
                        tool_calls = await parser.process_async(llm_response)

                    context.add_message(
                        message_content=llm_response['content'],
                        message_role="assistant",
                        message_type="tool_call"
                    )
                    tool_call = tool_calls[0]

                    if enable_timeline:
                        results = await k8s_executor.process_with_timing([tool_call], event_type="tool_execute", tool_name=tool_call.get('tool', 'unknown'))
                    else:
                        results = await k8s_executor.process_async([tool_call])
                    tool_result = results[0] if results else {}

                    # Check for stop signal
                    if isinstance(tool_result, dict) and tool_result.get("status") == "stop":
                        result["status"] = "success"
                        print(f"✅ Instance {instance_id} completed successfully")
                        return result

                    # Get formatted result from parser
                    formatted_result = tool_result.get('result', '')

                    # Ensure formatted_result is a string
                    if isinstance(formatted_result, dict):
                        formatted_result = json.dumps(formatted_result)
                    elif not isinstance(formatted_result, str):
                        formatted_result = str(formatted_result)

                    # Add steps remaining
                    steps_remaining = max_iterations - iteration
                    steps_remaining_content = f"Steps Remaining: {steps_remaining}"

                    context.add_message(
                        message_content=formatted_result + "\n" + steps_remaining_content,
                        message_role="user",
                        message_type="tool_result"
                    )

                except Exception as e:
                    print(f"\n❌ Error in agent loop for {instance_id}: {str(e)}")
                    result["error"] = str(e)
                    break

            if iteration >= max_iterations:
                print(f"\n⚠️ Instance {instance_id} reached maximum iterations ({max_iterations})")
                result["status"] = "max_iterations"

    except Exception as e:
        print(f"❌ Error processing instance {instance_id}: {str(e)}")
        result["error"] = str(e)

    return result


async def main(
    jsonl_file: str,
    max_concurrent: int = 3,
    enable_timeline: bool = False
):
    """Main function to process JSONL file with concurrent execution.

    Args:
        jsonl_file: Path to the JSONL file containing instances
        max_concurrent: Maximum number of concurrent executions
        enable_timeline: Enable timeline tracking for profiling
    """
    print("=== R2E Agent K8S Concurrent Executor ===\n")
    print(f"📁 JSONL file: {jsonl_file}")
    print(f"🔧 Max concurrent: {max_concurrent}")
    print(f"⏱️ Timeline tracking: {'ENABLED' if enable_timeline else 'DISABLED'}\n")

    # Load instances from JSONL file
    instances = []
    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    instances.append(json.loads(line))
        print(f"📊 Loaded {len(instances)} instances from file\n")
    except Exception as e:
        print(f"❌ Error loading JSONL file: {e}")
        return

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(instance_data, index):
        """Process an instance with semaphore control."""
        async with semaphore:
            # Use index as pod suffix to ensure uniqueness
            pod_suffix = f"{index:04d}"
            return await process_single_instance(
                instance_data,
                pod_suffix,
                enable_timeline
            )

    # Create tasks for all instances
    tasks = [
        process_with_semaphore(instance, idx)
        for idx, instance in enumerate(instances)
    ]

    # Execute all tasks
    print(f"🚀 Starting concurrent processing...\n")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Print summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)

    successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
    max_iter = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "max_iterations")
    exceptions = sum(1 for r in results if isinstance(r, Exception))

    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Max iterations: {max_iter}")
    print(f"🔥 Exceptions: {exceptions}")
    print(f"📊 Total: {len(results)}")

    # Print detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    for idx, result in enumerate(results):
        if isinstance(result, dict):
            status_emoji = {
                "success": "✅",
                "failed": "❌",
                "max_iterations": "⚠️"
            }.get(result.get("status"), "❓")
            print(f"{status_emoji} [{idx:04d}] {result.get('instance_id', 'unknown')}: {result.get('status', 'unknown')}")
            if result.get("error"):
                print(f"   Error: {result['error'][:100]}...")
        elif isinstance(result, Exception):
            print(f"🔥 [{idx:04d}] Exception: {str(result)[:100]}...")

    # Print timeline if enabled
    if enable_timeline:
        print("\n" + "="*60)
        print("TIMELINE PROFILING")
        print("="*60)
        timeline_data = {
            "stats": get_timeline().get_stats(),
            "events": get_timeline().get_timeline()
        }
        print(json.dumps(timeline_data, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="R2E Agent K8S Concurrent Executor")
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to JSONL file containing instances to process"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=3,
        help="Maximum number of concurrent executions (default: 3)"
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Enable timeline tracking for profiling"
    )
    args = parser.parse_args()

    # Set up event loop with proper configuration for high concurrency
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            main(
                jsonl_file=args.jsonl,
                max_concurrent=args.concurrent,
                enable_timeline=args.timeline
            )
        )
    finally:
        loop.close()
