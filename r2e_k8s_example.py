#!/usr/bin/env python3
"""
R2E Agent K8S Example - R2E-style tool execution in Kubernetes pods.

This example demonstrates an R2E-style agent that executes commands
in Kubernetes pods using the R2E prompt format and tools (bash, file_editor, search, finish).
"""

import sys
import json
import asyncio
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections import defaultdict

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


@dataclass
class TaskProgress:
    """Track progress for a single task."""
    task_id: int
    instance_id: str
    start_time: float = field(default_factory=time.time)
    iterations: int = 0
    llm_success: int = 0
    llm_error: int = 0
    llm_timeout: int = 0
    status: str = "running"  # running, success, failed, max_iter

    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time


class ProgressTracker:
    """Thread-safe progress tracker for concurrent tasks."""

    def __init__(self):
        self.tasks: Dict[int, TaskProgress] = {}
        self.lock = threading.Lock()
        self.display_running = False
        self.display_thread = None

    def create_task(self, task_id: int, instance_id: str) -> None:
        """Create a new task entry."""
        with self.lock:
            self.tasks[task_id] = TaskProgress(task_id=task_id, instance_id=instance_id)

    def update_iteration(self, task_id: int, iteration: int) -> None:
        """Update iteration count."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].iterations = iteration

    def increment_llm_success(self, task_id: int) -> None:
        """Increment successful LLM calls."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].llm_success += 1

    def increment_llm_error(self, task_id: int) -> None:
        """Increment LLM errors."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].llm_error += 1

    def increment_llm_timeout(self, task_id: int) -> None:
        """Increment LLM timeouts."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].llm_timeout += 1

    def set_status(self, task_id: int, status: str) -> None:
        """Set task status."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = status

    def get_snapshot(self) -> List[TaskProgress]:
        """Get a thread-safe snapshot of all tasks."""
        with self.lock:
            return list(self.tasks.values())

    def print_table(self) -> None:
        """Print progress table."""
        snapshot = self.get_snapshot()
        if not snapshot:
            return

        # Clear screen and print header
        print("\033[2J\033[H", end="")  # Clear screen, move cursor to top
        print("=" * 120)
        print("CONCURRENT TASK PROGRESS")
        print("=" * 120)

        # Table header
        header = f"{'Task':<6} {'Instance ID':<20} {'Iter':<5} {'Time(s)':<8} {'LLM✓':<6} {'LLM✗':<6} {'LLM⏱':<6} {'Status':<12}"
        print(header)
        print("-" * 120)

        # Sort by task_id
        snapshot.sort(key=lambda t: t.task_id)

        # Print each task
        for task in snapshot:
            status_emoji = {
                "running": "🔄",
                "success": "✅",
                "failed": "❌",
                "max_iter": "⚠️"
            }.get(task.status, "❓")

            row = (
                f"{task.task_id:<6} "
                f"{task.instance_id[:20]:<20} "
                f"{task.iterations:<5} "
                f"{task.elapsed_time():<8.1f} "
                f"{task.llm_success:<6} "
                f"{task.llm_error:<6} "
                f"{task.llm_timeout:<6} "
                f"{status_emoji} {task.status:<10}"
            )
            print(row)

        print("-" * 120)

        # Summary stats
        total = len(snapshot)
        running = sum(1 for t in snapshot if t.status == "running")
        completed = total - running
        total_llm_calls = sum(t.llm_success + t.llm_error + t.llm_timeout for t in snapshot)

        print(f"Total: {total} | Running: {running} | Completed: {completed} | Total LLM calls: {total_llm_calls}")
        print("=" * 120)
        print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def start_display(self, interval: float = 2.0) -> None:
        """Start background thread to display progress."""
        self.display_running = True

        def display_loop():
            while self.display_running:
                self.print_table()
                time.sleep(interval)

        self.display_thread = threading.Thread(target=display_loop, daemon=True)
        self.display_thread.start()

    def stop_display(self) -> None:
        """Stop background display thread."""
        self.display_running = False
        if self.display_thread:
            self.display_thread.join(timeout=3)
        # Print final table
        self.print_table()

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
    task_id: int,
    progress_tracker: ProgressTracker,
    enable_timeline: bool = False
) -> Dict:
    """Process a single instance from the JSONL file.

    Args:
        instance_data: Data for a single instance from JSONL
        pod_suffix: Unique suffix for the pod name (derived from instance_id)
        enable_timeline: Enable timeline tracking

    Returns:
        Result dictionary with instance_id and execution status
    """
    instance_id = instance_data.get("instance_id", "unknown")

    # Register task with progress tracker
    progress_tracker.create_task(task_id, instance_id)

    # Derive pod name from instance_id for idempotency
    # Replace all underscores and double-dashes with single dash for valid K8S naming
    # K8S pod names must match: [a-z0-9]([-a-z0-9]*[a-z0-9])?
    safe_instance_id = instance_id.replace('__', '-').replace('_', '-').replace('--', '-')
    pod_name = f"r2e-{safe_instance_id}".lower()

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
            pod_name=pod_name,
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
                progress_tracker.update_iteration(task_id, iteration)

                messages = context.get_llm_context()
                try:
                    # Call LLM and track success/error/timeout
                    try:
                        if enable_timeline:
                            llm_response = await llm_node.process_with_timing(messages, event_type="llm_call")
                        else:
                            llm_response = await llm_node.process_async(messages)
                        progress_tracker.increment_llm_success(task_id)
                    except asyncio.TimeoutError:
                        progress_tracker.increment_llm_timeout(task_id)
                        raise
                    except Exception:
                        progress_tracker.increment_llm_error(task_id)
                        raise

                    # Parse tool calls
                    if enable_timeline:
                        tool_calls = await parser.process_with_timing(llm_response, event_type="parse")
                    else:
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
                        progress_tracker.set_status(task_id, "success")
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
                    result["error"] = str(e)
                    progress_tracker.set_status(task_id, "failed")
                    break

            if iteration >= max_iterations:
                result["status"] = "max_iterations"
                progress_tracker.set_status(task_id, "max_iter")

    except Exception as e:
        result["error"] = str(e)
        progress_tracker.set_status(task_id, "failed")

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
    # Create progress tracker
    progress_tracker = ProgressTracker()

    print("=== R2E Agent K8S Concurrent Executor ===")
    print(f"📁 JSONL file: {jsonl_file}")
    print(f"🔧 Max concurrent: {max_concurrent}")
    print(f"⏱️ Timeline tracking: {'ENABLED' if enable_timeline else 'DISABLED'}")
    print()

    # Load instances from JSONL file
    instances = []
    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    instances.append(json.loads(line))
        print(f"📊 Loaded {len(instances)} instances from file")
    except Exception as e:
        print(f"❌ Error loading JSONL file: {e}")
        return

    # Start progress display
    print("\n🔄 Starting progress display...\n")
    time.sleep(1)
    progress_tracker.start_display(interval=2.0)

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(instance_data, index):
        """Process an instance with semaphore control."""
        async with semaphore:
            # Use instance_id to derive pod suffix for idempotency
            instance_id = instance_data.get("instance_id", f"unknown-{index}")
            # Sanitize for K8S naming: replace all underscores and double-dashes
            pod_suffix = instance_id.replace('__', '-').replace('_', '-').replace('--', '-')
            return await process_single_instance(
                instance_data,
                pod_suffix,
                task_id=index,
                progress_tracker=progress_tracker,
                enable_timeline=enable_timeline
            )

    # Create tasks for all instances
    tasks = [
        process_with_semaphore(instance, idx)
        for idx, instance in enumerate(instances)
    ]

    # Execute all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Stop progress display
    progress_tracker.stop_display()

    # Print summary
    print("\n\n" + "="*60)
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
