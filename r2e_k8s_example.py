#!/usr/bin/env python3
"""
R2E Agent K8S Example - R2E-style tool execution in Kubernetes pods.

This example demonstrates an R2E-style agent that executes commands
in Kubernetes pods using the R2E prompt format and tools (bash, file_editor, search, finish).
"""
import os
import sys
import json
import asyncio
import time
import threading
import warnings
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections import defaultdict

# Suppress aiohttp ResourceWarning about unclosed client sessions
# These warnings are triggered when the program exits and sessions are cleaned up by GC
# The sessions are properly managed by connection pooling and will be closed on exit
warnings.filterwarnings('ignore', category=ResourceWarning)

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
    end_time: float = None  # 任务完成时间
    iterations: int = 0
    llm_success: int = 0
    llm_error: int = 0
    llm_timeout: int = 0
    tool_parse_fail: int = 0
    tool_exec_fail: int = 0
    status: str = "running"  # running, success, failed, max_iter

    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time:
            # 任务已完成，返回总耗时
            return self.end_time - self.start_time
        else:
            # 任务进行中，返回当前耗时
            return time.time() - self.start_time


class ProgressTracker:
    """Thread-safe progress tracker for concurrent tasks."""

    def __init__(self):
        self.tasks: Dict[int, TaskProgress] = {}
        self.lock = threading.Lock()
        self.display_running = False
        self.display_thread = None
        self.program_start_time = time.time()

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

    def increment_tool_parse_fail(self, task_id: int) -> None:
        """Increment tool parsing failures."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].tool_parse_fail += 1

    def increment_tool_exec_fail(self, task_id: int) -> None:
        """Increment tool execution failures."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].tool_exec_fail += 1

    def set_status(self, task_id: int, status: str) -> None:
        """Set task status."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = status
                # Record end time when task completes
                if status in ["success", "failed", "max_iter"]:
                    self.tasks[task_id].end_time = time.time()

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
        print("=" * 175)
        print("CONCURRENT TASK PROGRESS")
        print("=" * 175)

        # Table header
        header = f"{'Task':<6} {'Instance ID':<40} {'Iter':<5} {'Time(s)':<8} {'LLM✓':<6} {'LLM✗':<6} {'LLM⏱':<6} {'TPF':<5} {'TEF':<5} {'Status':<12}"
        print(header)
        print("-" * 175)

        # Sort by task_id
        snapshot.sort(key=lambda t: t.task_id)

        # Print each task
        for task in snapshot:
            status_emoji = {
                "initializing": "⏳",
                "running": "🔄",
                "success": "✅",
                "failed": "❌",
                "max_iter": "⚠️"
            }.get(task.status, "❓")

            row = (
                f"{task.task_id:<6} "
                f"{task.instance_id[:40]:<40} "
                f"{task.iterations:<5} "
                f"{task.elapsed_time():<8.1f} "
                f"{task.llm_success:<6} "
                f"{task.llm_error:<6} "
                f"{task.llm_timeout:<6} "
                f"{task.tool_parse_fail:<5} "
                f"{task.tool_exec_fail:<5} "
                f"{status_emoji} {task.status:<10}"
            )
            print(row)

        print("-" * 175)

        # Summary stats
        total = len(snapshot)
        initializing = sum(1 for t in snapshot if t.status == "initializing")
        running = sum(1 for t in snapshot if t.status == "running")
        completed = total - running - initializing
        total_llm_calls = sum(t.llm_success + t.llm_error + t.llm_timeout for t in snapshot)

        total_parse_fails = sum(t.tool_parse_fail for t in snapshot)
        total_exec_fails = sum(t.tool_exec_fail for t in snapshot)

        print(f"Total: {total} | Initializing: {initializing} | Running: {running} | Completed: {completed} | Total LLM calls: {total_llm_calls} | Parse fails: {total_parse_fails} | Exec fails: {total_exec_fails}")
        print("=" * 175)
        
        # Calculate total program runtime
        program_runtime = time.time() - self.program_start_time
        hours = int(program_runtime // 3600)
        minutes = int((program_runtime % 3600) // 60)
        seconds = int(program_runtime % 60)
        
        runtime_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"
        
        print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')} | Program runtime: {runtime_str}")
        print(f"TPF=Tool Parse Fail | TEF=Tool Exec Fail")

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
    output_dir: str = None,
    enable_timeline: bool = False,
    debug: bool = False,
    cpu_request: str = "0.3",
    memory_request: str = "1Gi",
    max_execution_time: float = None,
    llm_timeout: float = 120.0,
    tool_timeout: float = 300.0) -> Dict:
    """Process a single instance from the JSONL file.

    Args:
        instance_data: Data for a single instance from JSONL
        pod_suffix: Unique suffix for the pod name (derived from instance_id)
        output_dir: Directory to save context and log files
        enable_timeline: Enable timeline tracking
        debug: Enable debug mode (detailed logging)
        max_execution_time: Maximum execution time in seconds (None for no limit)
        llm_timeout: LLM call timeout in seconds (default: 120s)
        tool_timeout: Tool execution timeout in seconds (default: 300s)

    Returns:
        Result dictionary with instance_id and execution status
    """
    instance_id = instance_data.get("instance_id", "unknown")

    # Register task with progress tracker
    progress_tracker.create_task(task_id, instance_id)
    progress_tracker.set_status(task_id, "initializing")
    
    # Setup output files if output_dir is provided
    log_file = None
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        log_file_path = os.path.join(output_dir, f"{instance_id}.log")
        log_file = open(log_file_path, 'w', encoding='utf-8')

    # Derive pod name from instance_id with random suffix
    # Replace all underscores and double-dashes with single dash for valid K8S naming
    # K8S pod names must match: [a-z0-9]([-a-z0-9]*[a-z0-9])?
    import random
    safe_instance_id = instance_id.replace('__', '-').replace('_', '-').replace('--', '-')
    random_suffix = random.randint(1000, 9999)
    pod_name = f"r2e-{safe_instance_id}-{random_suffix}".lower()
    
    # Truncate if too long (K8S max is 253 chars, leave margin)
    if len(pod_name) > 200:
        # Keep prefix and suffix, truncate middle
        pod_name = f"{pod_name[:100]}-{random_suffix}"

    result = {
        "instance_id": instance_id,
        "status": "failed",
        "error": None
    }
    
    def log(message: str):
        """Helper to write to log file if available."""
        if log_file:
            log_file.write(message + "\n")
            log_file.flush()

    try:
        log(f"=== Starting task for instance: {instance_id} ===")
        log(f"Task ID: {task_id}")
        log(f"Pod name: {pod_name if 'pod_name' in locals() else 'not yet assigned'}")
        if max_execution_time:
            log(f"Max execution time: {max_execution_time}s ({max_execution_time/60:.1f} minutes)")
        log(f"LLM timeout: {llm_timeout}s")
        log(f"Tool timeout: {tool_timeout}s")
        
        # Track execution start time
        execution_start_time = time.time()
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
            timeline_enabled=enable_timeline,
            timeout=llm_timeout
        )
        #llm_node.set_retry_config(max_retries=3)

        # Tool Parsing Node (R2E-style XML parser)
        parser = ToolParsingNode(
            name=f"R2EParser-{pod_suffix}",
            parse_function=create_r2e_parser(),
            timeline_enabled=enable_timeline
        )

        # Use the image from the instance data if provided
        image = instance_data.get("image", "python:3.11-slim")
        
        log(f"Pod name: {pod_name}")
        log(f"Using image: {image}")

        # Use async context manager for automatic cleanup
        async with K8SToolExecutionNode(
                name=f"R2EK8SExecutor-{pod_suffix}",
                namespace="qianfan-train-cpu-ns",
                node_selector={"nvme": "ok"},
                kubeconfig_path="./cpu_config2",
                image=image,
                pod_name=pod_name,
                environment={
                    "PYTHONPATH": "/testbed",
                    "PYTHONIOENCODING": "utf-8",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8"
                },
                cpu_request=cpu_request,
                memory_request=memory_request,
                timeline_enabled=enable_timeline,
                tool_timeout=tool_timeout
        ) as k8s_executor:
            log(f"K8S executor initialized")

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

            max_iterations = instance_data.get("max_iterations", 100)
            iteration = 0

            # Mark as running after pod is ready
            progress_tracker.set_status(task_id, "running")

            while iteration < max_iterations:
                # Check total execution time limit
                if max_execution_time:
                    elapsed_time = time.time() - execution_start_time
                    if elapsed_time > max_execution_time:
                        log(f"\n⏱️  Execution time limit reached: {elapsed_time:.1f}s > {max_execution_time}s")
                        print(f"\n⏱️  Task {task_id} ({instance_id}): Execution time limit reached ({elapsed_time/60:.1f} minutes)")
                        result["status"] = "timeout"
                        result["error"] = f"Execution time limit reached: {elapsed_time:.1f}s"
                        progress_tracker.set_status(task_id, "failed")
                        
                        # Add timeout message to context
                        context.add_message(
                            message_content=f"Task execution stopped: Maximum execution time ({max_execution_time}s) exceeded.",
                            message_role="system",
                            message_type="error"
                        )
                        break
                
                iteration += 1
                progress_tracker.update_iteration(task_id, iteration)
                log(f"\n=== Iteration {iteration}/{max_iterations} ===")

                messages = context.get_llm_context()
                try:
                    # Call LLM and track success/error/timeout
                    try:
                        log(f"Calling LLM with timeout={llm_timeout}s...")
                        if enable_timeline:
                            llm_response = await asyncio.wait_for(
                                llm_node.process_with_timing(messages, event_type="llm_call"),
                                timeout=llm_timeout
                            )
                        else:
                            llm_response = await asyncio.wait_for(
                                llm_node.process_async(messages),
                                timeout=llm_timeout
                            )

                        log(f"LLM Response: {llm_response.get('content', '')[:200]}...")
                        
                        # Print raw LLM response for debugging
                        if debug:
                            print(f"\n🤖 Task {task_id} iter {iteration} - LLM Response (FULL):")
                            print(f"{llm_response.get('content', '')}")
                            print("-" * 80)

                        progress_tracker.increment_llm_success(task_id)
                    except asyncio.TimeoutError:
                        progress_tracker.increment_llm_timeout(task_id)
                        error_msg = f"LLM call timeout after {llm_timeout}s"
                        log(f"❌ {error_msg}")
                        print(f"\n❌ Task {task_id} ({instance_id}) iter {iteration}: {error_msg}")
                        
                        # Add timeout error to context
                        context.add_message(
                            message_content=f"ERROR: {error_msg}. Task terminated.",
                            message_role="system",
                            message_type="error"
                        )
                        
                        result["status"] = "failed"
                        result["error"] = error_msg
                        progress_tracker.set_status(task_id, "failed")
                        break
                    except Exception:
                        progress_tracker.increment_llm_error(task_id)
                        raise

                    # Parse tool calls
                    try:
                        if enable_timeline:
                            tool_calls = await parser.process_with_timing(llm_response, event_type="parse")
                        else:
                            tool_calls = await parser.process_async(llm_response)

                        # Print parsed tool calls for debugging
                        if debug and tool_calls and len(tool_calls) > 0:
                            print(f"📝 Parsed tool: {tool_calls[0].get('tool', 'unknown')}")
                            print(f"📝 Tool params (FULL): {tool_calls[0].get('parameters', {})}")
                            print("-" * 80)

                        if not tool_calls or len(tool_calls) == 0:
                            progress_tracker.increment_tool_parse_fail(task_id)
                            log(f"Tool parsing returned empty list - treating as completion")
                            print(f"\n⚠️  Task {task_id} ({instance_id}): Tool parsing returned empty list - treating as task completion")
                            print(f"   LLM Response: {llm_response.get('content', '')[:500]}")
                            
                            # Add LLM response to context
                            context.add_message(
                                message_content=llm_response['content'],
                                message_role="assistant",
                                message_type="tool_call"
                            )
                            
                            # Terminate loop when LLM doesn't call any tool
                            result["status"] = "success"
                            progress_tracker.set_status(task_id, "success")
                            break

                    except Exception as e:
                        progress_tracker.increment_tool_parse_fail(task_id)
                        log(f"Tool parse error: {str(e)}")
                        print(f"\n❌ Task {task_id} ({instance_id}): Tool parse error: {str(e)}")
                        print(f"   LLM Response: {llm_response.get('content', '')[:500]}")
                        
                        # Add error response to context and terminate
                        context.add_message(
                            message_content=llm_response.get('content', ''),
                            message_role="assistant",
                            message_type="tool_call"
                        )
                        
                        result["status"] = "failed"
                        result["error"] = str(e)
                        progress_tracker.set_status(task_id, "failed")
                        break

                    context.add_message(
                        message_content=llm_response['content'],
                        message_role="assistant",
                        message_type="tool_call"
                    )
                    tool_call = tool_calls[0]
                    
                    tool_name = tool_call.get('tool', 'unknown')
                    log(f"Executing tool: {tool_name}")
                    log(f"Tool parameters: {json.dumps(tool_call.get('parameters', {}), indent=2)}")

                    # Execute tool
                    try:
                        if enable_timeline:
                            results = await k8s_executor.process_with_timing([tool_call], event_type="tool_execute", tool_name=tool_name)
                        else:
                            results = await k8s_executor.process_async([tool_call])
                        tool_result = results[0] if results else {}
                        
                        log(f"Tool execution status: {tool_result.get('status', 'unknown')}")
                        if 'stdout' in tool_result:
                            log(f"Tool stdout:\n{tool_result.get('stdout', '')}")
                        if 'stderr' in tool_result and tool_result.get('stderr'):
                            log(f"Tool stderr:\n{tool_result.get('stderr', '')}")

                        # Check if tool execution had errors
                        if isinstance(tool_result, dict) and tool_result.get("status") == "error":
                            progress_tracker.increment_tool_exec_fail(task_id)
                            error_msg = tool_result.get('error', 'Unknown error')
                            print(f"\n⚠️  Task {task_id} ({instance_id}) iter {iteration}: Tool execution error")
                            print(f"   Tool: {tool_call.get('tool', 'unknown')}")
                            print(f"   Error: {error_msg[:300]}")
                            # Don't continue with more error details to avoid log spam

                    except Exception as e:
                        progress_tracker.increment_tool_exec_fail(task_id)
                        print(f"\n❌ Task {task_id} ({instance_id}): Tool execution exception: {str(e)}")
                        print(f"   Tool: {tool_call.get('tool', 'unknown')}")
                        raise

                    # Get formatted result from parser
                    formatted_result = tool_result.get('result', '')

                    # Print formatted result for debugging
                    if debug:
                        print(f"🔧 Tool result status: {tool_result.get('status', 'unknown')}")
                        print(f"📤 Formatted result (FULL):\n{str(formatted_result)}")
                        print("=" * 80)

                    # Ensure formatted_result is a string
                    if isinstance(formatted_result, dict):
                        formatted_result = json.dumps(formatted_result)
                    elif not isinstance(formatted_result, str):
                        formatted_result = str(formatted_result)

                    # Add steps remaining
                    steps_remaining = max_iterations - iteration
                    steps_remaining_content = f"Steps Remaining: {steps_remaining}"

                    # Add tool result to context
                    context.add_message(
                        message_content=formatted_result + "\n" + steps_remaining_content,
                        message_role="user",
                        message_type="tool_result"
                    )

                    # Check for stop signal AFTER adding result to context
                    if isinstance(tool_result, dict) and tool_result.get("status") == "stop":
                        log(f"Finish tool executed - terminating successfully")
                        print(f"\n✅ Task {task_id} ({instance_id}): Finish tool executed, terminating successfully")
                        result["status"] = "success"
                        progress_tracker.set_status(task_id, "success")
                        break

                except Exception as e:
                    import traceback
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    log(f"Error in agent loop: {error_msg}")
                    log(f"Traceback: {traceback.format_exc()}")
                    print(f"\n❌ Error in agent loop for task {task_id} ({instance_id}): {error_msg}")
                    print(f"   Traceback: {traceback.format_exc()[:500]}")
                    result["error"] = error_msg
                    progress_tracker.set_status(task_id, "failed")
                    break

            if iteration >= max_iterations:
                log(f"Reached max iterations ({max_iterations})")
                result["status"] = "max_iterations"
                progress_tracker.set_status(task_id, "max_iter")
            
            # Generate and save patch before pod is deleted (must be inside async with block)
            if output_dir:
                try:
                    log(f"Generating patch from testbed changes...")
                    
                    # Stage all changes
                    await k8s_executor._execute_kubectl_async("cd /testbed && git add -A")
                    
                    # Get base_commit from instance data
                    base_commit = instance_data.get('base_commit', None)
                    
                    # Generate patch
                    if base_commit:
                        log(f"Generating patch against base_commit: {base_commit}")
                        patch_output, exit_code = await k8s_executor._execute_kubectl_async(
                            f"cd /testbed && git diff {base_commit}"
                        )
                    else:
                        log("Generating patch against staged changes")
                        patch_output, exit_code = await k8s_executor._execute_kubectl_async(
                            "cd /testbed && git diff --cached"
                        )
                    
                    if exit_code == "0" or exit_code == 0:
                        patch = patch_output.strip()
                        log(f"Patch generated successfully, size: {len(patch)} characters")
                        
                        if patch:
                            # Log patch preview
                            lines = patch.split('\n')
                            log(f"Patch preview (first 5 lines):")
                            for line in lines[:5]:
                                log(f"  {line}")
                            if len(lines) > 5:
                                log(f"  ... ({len(lines) - 5} more lines)")
                            
                            # Save patch directly to output directory as {instance_id}.patch
                            patch_filepath = os.path.join(output_dir, f"{instance_id}.patch")
                            with open(patch_filepath, 'w', encoding='utf-8') as f:
                                f.write(patch)
                            log(f"Saved patch to: {patch_filepath}")
                            result["patch_file"] = patch_filepath
                        else:
                            log("No changes detected, patch is empty")
                    else:
                        log(f"Failed to generate patch, exit code: {exit_code}")
                        
                except Exception as e:
                    log(f"Error generating patch: {str(e)}")

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        log(f"Fatal error: {error_msg}")
        log(f"Traceback: {traceback.format_exc()}")
        print(f"\n❌ Fatal error processing instance {instance_id}: {error_msg}")
        print(f"   Traceback: {traceback.format_exc()[:500]}")
        result["error"] = error_msg
        progress_tracker.set_status(task_id, "failed")
    
    finally:
        # Log pod cleanup (the async context manager will handle actual deletion)
        log(f"Task finished, pod {pod_name} will be deleted by context manager")
        
        # Save context to file
        if output_dir and 'context' in locals():
            try:
                import os
                context_file_path = os.path.join(output_dir, f"{instance_id}.context")
                with open(context_file_path, 'w', encoding='utf-8') as f:
                    messages = context.get_llm_context()
                    for msg in messages:
                        f.write(json.dumps(msg, ensure_ascii=False) + '\n')
                if log_file:
                    log(f"Context saved to {context_file_path}")
            except Exception as e:
                if log_file:
                    log(f"Error saving context: {e}")
        
        # Close aiohttp session in LLM node
        if 'llm_node' in locals() and llm_node:
            try:
                await llm_node.close_async()
            except Exception as e:
                if log_file:
                    log(f"Error closing LLM node: {e}")
        
        # Close log file
        if log_file:
            try:
                log(f"=== Task completed with status: {result.get('status', 'unknown')} ===")
                log(f"=== Pod {pod_name} cleanup delegated to context manager ===")
                log_file.close()
            except:
                pass

    return result


async def main(
    jsonl_file: str,
    max_concurrent: int = 3,
    output_dir: str = None,
    enable_timeline: bool = False,
    debug: bool = False,
    cpu_request: str = "0.3",
    memory_request: str = "1Gi",
    max_execution_time: float = None,
    llm_timeout: float = 120.0,
    tool_timeout: float = 300.0
):
    """Main function to process JSONL file with concurrent execution.

    Args:
        jsonl_file: Path to the JSONL file containing instances
        max_concurrent: Maximum number of concurrent executions
        output_dir: Directory to save context and log files
        enable_timeline: Enable timeline tracking for profiling
        debug: Enable debug mode (detailed logging, no progress table)
        max_execution_time: Maximum execution time per instance in seconds (None for no limit)
        llm_timeout: LLM call timeout in seconds (default: 120s)
        tool_timeout: Tool execution timeout in seconds (default: 300s)
    """
    # Create progress tracker
    progress_tracker = ProgressTracker()

    print("=== R2E Agent K8S Concurrent Executor ===")
    print(f"📁 JSONL file: {jsonl_file}")
    print(f"🔧 Max concurrent: {max_concurrent}")
    print(f"⏱️  Timeline tracking: {'ENABLED' if enable_timeline else 'DISABLED'}")
    print(f"🐛 Debug mode: {'ENABLED' if debug else 'DISABLED'}")
    if max_execution_time:
        print(f"⏰ Max execution time: {max_execution_time}s ({max_execution_time/60:.1f} minutes)")
    else:
        print(f"⏰ Max execution time: No limit")
    print(f"🕐 LLM timeout: {llm_timeout}s")
    print(f"🔧 Tool timeout: {tool_timeout}s")
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

    # Start progress display (only if not in debug mode)
    if not debug:
        print("\n🔄 Starting progress display...\n")
        time.sleep(1)
        progress_tracker.start_display(interval=2.0)
    else:
        print("\n🐛 Debug mode: Progress table disabled, detailed logging enabled\n")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    active_tasks = {'count': 0}
    lock = asyncio.Lock()

    async def process_with_semaphore(instance_data, index):
        """Process an instance with semaphore control."""
        async with semaphore:
            async with lock:
                active_tasks['count'] += 1
                current_active = active_tasks['count']

            # Log when we reach peak concurrency
            if current_active >= max_concurrent * 0.8:
                print(f"⚡ High concurrency: {current_active}/{max_concurrent} tasks active")

            try:
                # Use instance_id to derive pod suffix for idempotency
                instance_id = instance_data.get("instance_id", f"unknown-{index}")
                # Sanitize for K8S naming: replace all underscores and double-dashes
                pod_suffix = instance_id.replace('__', '-').replace('_', '-').replace('--', '-')
                return await process_single_instance(
                    instance_data,
                    pod_suffix,
                    task_id=index,
                    progress_tracker=progress_tracker,
                    output_dir=output_dir,
                    enable_timeline=enable_timeline,
                    debug=debug,
                    cpu_request=cpu_request,
                    memory_request=memory_request,
                    max_execution_time=max_execution_time,
                    llm_timeout=llm_timeout,
                    tool_timeout=tool_timeout
                )
            finally:
                async with lock:
                    active_tasks['count'] -= 1

    # Create tasks for all instances
    tasks = [
        process_with_semaphore(instance, idx)
        for idx, instance in enumerate(instances)
    ]

    # Execute all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Wait a bit for cleanup to complete
    await asyncio.sleep(0.5)

    # Stop progress display (only if it was started)
    if not debug:
        progress_tracker.stop_display()

    # Print summary
    print("\n\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)

    successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
    timeout = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "timeout")
    max_iter = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "max_iterations")
    exceptions = sum(1 for r in results if isinstance(r, Exception))

    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Timeout: {timeout}")
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
                "timeout": "⏱️",
                "max_iterations": "⚠️"
            }.get(result.get("status"), "❓")
            print(f"{status_emoji} [{idx:04d}] {result.get('instance_id', 'unknown')}: {result.get('status', 'unknown')}")
            if result.get("error"):
                print(f"   Error: {result['error'][:100]}...")
        elif isinstance(result, Exception):
            print(f"🔥 [{idx:04d}] Exception: {str(result)[:100]}...")

    # Print and save timeline if enabled
    if enable_timeline:
        print("\n" + "="*60)
        print("TIMELINE PROFILING")
        print("="*60)
        timeline_data = {
            "stats": get_timeline().get_stats(),
            "events": get_timeline().get_timeline()
        }
        print(json.dumps(timeline_data, indent=2))
        
        # Save timeline to output directory
        if output_dir:
            try:
                timeline_file = os.path.join(output_dir, "timeline_profile.json")
                with open(timeline_file, 'w', encoding='utf-8') as f:
                    json.dump(timeline_data, f, indent=2, ensure_ascii=False)
                print(f"\n✓ Timeline saved to: {timeline_file}")
            except Exception as e:
                print(f"\n✗ Error saving timeline: {str(e)}")


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
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save context and log files (default: None)"
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Enable timeline tracking for profiling"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (detailed logging, no progress table)"
    )
    parser.add_argument(
        "--cpu",
        type=str,
        default="0.3",
        help="CPU resource request per pod (default: 0.3 core)"
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="1Gi",
        help="Memory resource request per pod (default: 1Gi)"
    )
    parser.add_argument(
        "--max-execution-time",
        type=float,
        default=None,
        help="Maximum execution time per instance in seconds (default: no limit)"
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=120.0,
        help="LLM call timeout in seconds (default: 120s)"
    )
    parser.add_argument(
        "--tool-timeout",
        type=float,
        default=300.0,
        help="Tool execution timeout in seconds (default: 300s)"
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
                output_dir=args.output_dir,
                enable_timeline=args.timeline,
                debug=args.debug,
                cpu_request=args.cpu,
                memory_request=args.memory,
                max_execution_time=args.max_execution_time,
                llm_timeout=args.llm_timeout,
                tool_timeout=args.tool_timeout
            )
        )
        
        # Wait for all pending tasks to complete
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user, cleaning up...")
        # Cancel all running tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        # Wait for cancellation to complete
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        # Give time for cleanup
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.close()
