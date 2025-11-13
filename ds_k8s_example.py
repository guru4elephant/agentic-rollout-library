#!/usr/bin/env python3
"""
DeepSeek V3.1 Agent K8S Example - DeepSeek-style tool execution in Kubernetes pods.

This example demonstrates a DeepSeek-style agent that executes commands
in Kubernetes pods using the DeepSeek completion format and tools (bash, str_replace_editor).
"""
import os
import sys
import json
import asyncio
import time
import threading
import warnings
import re
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections import defaultdict

# Suppress aiohttp ResourceWarning about unclosed client sessions
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


# DeepSeek special tokens
TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
TOOL_CALL_END = "<｜tool▁call▁end｜>"
TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
TOOL_SEP = "<｜tool▁sep｜>"
TOOL_OUTPUT_BEGIN = "<｜tool▁output▁begin｜>"
TOOL_OUTPUT_END = "<｜tool▁output▁end｜>"
END_OF_SENTENCE = "<｜end▁of▁sentence｜>"


# System prompt for DeepSeek
DEEPSEEK_SYSTEM_PROMPT = """You are a helpful software engineer assistant.

## Tools
You have access to the following tools:

### bash
Description: Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You don't have access to the internet via this tool.
* You do have access to a mirror of common linux and python packages via apt and pip.
* State is persistent across command calls and discussions with the user.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.

Parameters: {"title": "BashInput", "type": "object", "properties": {"command": {"title": "Command", "description": "The bash command to run. Relative path is preferred in the command.", "type": "string"}}, "required": ["command"], "additionalProperties": false}

### str_replace_editor
Description: Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`

Parameters: {"title": "SweEditorInput", "type": "object", "properties": {"command": {"title": "Command", "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`.", "enum": ["view", "create", "str_replace", "insert"], "type": "string"}, "path": {"title": "Path", "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.", "type": "string"}, "file_text": {"title": "File Text", "description": "Required parameter of `create` command, with the content of the file to be created.", "type": "string"}, "insert_line": {"title": "Insert Line", "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.", "type": "integer"}, "new_str": {"title": "New Str", "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.", "type": "string"}, "old_str": {"title": "Old Str", "description": "Required parameter of `str_replace` command containing the string in `path` to replace.", "type": "string"}, "view_range": {"title": "View Range", "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.", "type": "array", "items": {"type": "integer"}}}, "required": ["command", "path"], "additionalProperties": false}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{additional_tool_calls}<｜tool▁calls▁end｜>

Where:
- `tool_call_name` must be an exact match to one of the available tools
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces"""


def create_ds_parser():
    """
    Create DeepSeek-style parser for tool calls.
    Parses the special token format used by DeepSeek V3.1.
    
    Format:
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_name<｜tool▁sep｜>{"param": "value"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    def parse_tool_calls(llm_response: Dict) -> List[Dict]:
        content = llm_response.get("content", "")
        
        # Check if content ends with TOOL_CALLS_END
        if TOOL_CALLS_END not in content:
            return []
        
        # Extract everything between TOOL_CALLS_BEGIN and TOOL_CALLS_END
        if TOOL_CALLS_BEGIN not in content:
            return []
        
        # Find the tool calls section
        begin_idx = content.find(TOOL_CALLS_BEGIN)
        end_idx = content.find(TOOL_CALLS_END)
        
        if begin_idx == -1 or end_idx == -1:
            return []
        
        tool_calls_section = content[begin_idx + len(TOOL_CALLS_BEGIN):end_idx]
        
        # Parse individual tool calls
        tool_calls = []
        
        # Find all tool calls between TOOL_CALL_BEGIN and TOOL_CALL_END
        pattern = re.escape(TOOL_CALL_BEGIN) + r'(.*?)' + re.escape(TOOL_CALL_END)
        matches = re.findall(pattern, tool_calls_section, re.DOTALL)
        
        for match in matches:
            # Split by TOOL_SEP to get tool name and arguments
            if TOOL_SEP not in match:
                continue
            
            parts = match.split(TOOL_SEP, 1)
            if len(parts) != 2:
                continue
            
            tool_name = parts[0].strip()
            tool_args_str = parts[1].strip()
            
            # Parse JSON arguments
            try:
                tool_args = json.loads(tool_args_str)
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to parse tool arguments as JSON: {e}")
                print(f"   Tool: {tool_name}")
                print(f"   Args string: {tool_args_str[:200]}")
                continue
            
            # Map DeepSeek tool names to our internal tool names
            internal_tool_name = tool_name
            if tool_name == "bash":
                internal_tool_name = "ds_bash_executor"
            elif tool_name == "str_replace_editor":
                internal_tool_name = "ds_file_editor"
            
            tool_call = {
                "tool": internal_tool_name,
                "parameters": tool_args
            }
            
            tool_calls.append(tool_call)
        
        return tool_calls
    
    return parse_tool_calls


@dataclass
class TaskProgress:
    """Track progress for a single task."""
    task_id: int
    instance_id: str
    start_time: float = field(default_factory=time.time)
    end_time: float = None
    iterations: int = 0
    llm_success: int = 0
    llm_error: int = 0
    llm_timeout: int = 0
    tool_parse_fail: int = 0
    tool_exec_fail: int = 0
    status: str = "running"

    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        else:
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
        print("\033[2J\033[H", end="")
        print("=" * 175)
        print("DEEPSEEK V3.1 CONCURRENT TASK PROGRESS")
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
        self.print_table()


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
    """Process a single instance using DeepSeek V3.1 completion format.

    Args:
        instance_data: Data for a single instance from JSONL
        pod_suffix: Unique suffix for the pod name
        task_id: Task ID for progress tracking
        progress_tracker: Progress tracker instance
        output_dir: Directory to save context and log files
        enable_timeline: Enable timeline tracking
        debug: Enable debug mode
        cpu_request: CPU resource request
        memory_request: Memory resource request
        max_execution_time: Maximum execution time in seconds
        llm_timeout: LLM call timeout in seconds
        tool_timeout: Tool execution timeout in seconds

    Returns:
        Result dictionary with instance_id and execution status
    """
    instance_id = instance_data.get("instance_id", "unknown")
    
    # Check if patch file already exists
    if output_dir:
        import os
        patch_filepath = os.path.join(output_dir, f"{instance_id}.patch")
        if os.path.exists(patch_filepath):
            print(f"⏭️  Task {task_id} ({instance_id}): Patch file already exists, skipping")
            return {
                "task_id": task_id,
                "instance_id": instance_id,
                "status": "skipped",
                "reason": "patch_exists",
                "patch_file": patch_filepath
            }

    # Register task with progress tracker
    progress_tracker.create_task(task_id, instance_id)
    progress_tracker.set_status(task_id, "initializing")
    
    # Setup output files
    log_file = None
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        log_file_path = os.path.join(output_dir, f"{instance_id}.log")
        log_file = open(log_file_path, 'w', encoding='utf-8')

    # Derive pod name
    import random
    safe_instance_id = instance_id.replace('__', '-').replace('_', '-').replace('--', '-')
    random_suffix = random.randint(1000, 9999)
    pod_name = f"ds-{safe_instance_id}-{random_suffix}".lower()
    
    if len(pod_name) > 200:
        pod_name = f"{pod_name[:100]}-{random_suffix}"

    result = {
        "instance_id": instance_id,
        "status": "failed",
        "error": None
    }
    
    def log(message: str):
        """Helper to write to log file."""
        if log_file:
            log_file.write(message + "\n")
            log_file.flush()

    try:
        log(f"=== Starting DeepSeek task for instance: {instance_id} ===")
        log(f"Task ID: {task_id}")
        log(f"Pod name: {pod_name}")
        if max_execution_time:
            log(f"Max execution time: {max_execution_time}s ({max_execution_time/60:.1f} minutes)")
        log(f"LLM timeout: {llm_timeout}s")
        log(f"Tool timeout: {tool_timeout}s")
        
        execution_start_time = time.time()
        
        # Context Engineering Node
        context = ContextEngineeringNode(name=f"DSK8SContext-{pod_suffix}", timeline_enabled=enable_timeline)

        # LLM Node - using completion endpoint
        llm_handle = create_openai_api_handle_async(
            base_url="",
            api_key="",
            model="",
            use_completion=True  # Use completion endpoint instead of chat
        )

        llm_node = LLMNode(
            name=f"DSLLM-{pod_suffix}",
            function_handle=llm_handle,
            model_config={
                "temperature": 0.7,
                "max_tokens": 8000
            },
            timeline_enabled=enable_timeline,
            timeout=llm_timeout
        )

        # Tool Parsing Node (DeepSeek-style parser)
        parser = ToolParsingNode(
            name=f"DSParser-{pod_suffix}",
            parse_function=create_ds_parser(),
            timeline_enabled=enable_timeline
        )

        # Use the image from instance data
        image = instance_data.get("image", "python:3.11-slim")
        
        log(f"Using image: {image}")

        # Use async context manager for automatic cleanup
        async with K8SToolExecutionNode(
                name=f"DSK8SExecutor-{pod_suffix}",
                namespace="rl-training",
                kubeconfig_path="./swe-bench-verified-workspace/config_cce_new",
                image=image,
                pod_name=pod_name,
                environment={
                    "PYTHONPATH": "/testbed",
                    "PYTHONIOENCODING": "utf-8",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "http_proxy": "http://agent.baidu.com:8891",
                    "https_proxy": "http://agent.baidu.com:8891",
                    "PIP_INDEX_URL": "http://pip.baidu.com/pypi/simple",
                    "PIP_TRUSTED_HOST": "pip.baidu.com"                    
                },
                cpu_request=cpu_request,
                memory_request=memory_request,
                timeline_enabled=enable_timeline,
                tool_timeout=tool_timeout
        ) as k8s_executor:
            log(f"K8S executor initialized")

            # Register DeepSeek tools
            k8s_executor.register_tool(
                "ds_bash_executor",
                "src/tools/deepseek/bash_func.py"
            )

            k8s_executor.register_tool(
                "ds_file_editor",
                "src/tools/deepseek/file_editor.py"
            )

            # Build system prompt
            system_prompt = DEEPSEEK_SYSTEM_PROMPT

            # Add system prompt message
            context.add_message(
                message_content=system_prompt,
                message_role="system",
                message_type="system_prompt"
            )

            # Get problem statement
            issue = instance_data.get("problem_statement", "No problem statement provided")

            # Build user query in DeepSeek format
            query = f"""<｜User｜><uploaded_files>
/testbed
</uploaded_files>
I've uploaded a code repository in the directory /testbed (not in /tmp/inputs). Consider the following PR description:

<pr_description>
{issue}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!

Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <pr_description> is satisfied.

You are only allowed to call **ONE** function each time!

<｜Assistant｜></think>"""

            context.add_message(
                message_content=query,
                message_role="user",
                message_type="query"
            )

            max_iterations = instance_data.get("max_iterations", 100)
            iteration = 0

            # Mark as running after pod is ready
            progress_tracker.set_status(task_id, "running")

            # Accumulated prompt for completion API
            accumulated_prompt = system_prompt + "\n\n" + query

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
                        break
                
                iteration += 1
                progress_tracker.update_iteration(task_id, iteration)
                log(f"\n=== Iteration {iteration}/{max_iterations} ===")

                try:
                    # Call LLM with accumulated prompt
                    try:
                        log(f"Calling LLM completion with timeout={llm_timeout}s...")
                        
                        # Create a single message with the accumulated prompt for completion API
                        completion_messages = [{"role": "user", "content": accumulated_prompt}]
                        
                        if enable_timeline:
                            llm_response = await asyncio.wait_for(
                                llm_node.process_with_timing(completion_messages, event_type="llm_call"),
                                timeout=llm_timeout
                            )
                        else:
                            llm_response = await asyncio.wait_for(
                                llm_node.process_async(completion_messages),
                                timeout=llm_timeout
                            )

                        log(f"LLM Response: {llm_response.get('content', '')[:200]}...")
                        
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
                        result["status"] = "failed"
                        result["error"] = error_msg
                        progress_tracker.set_status(task_id, "failed")
                        break
                    except Exception:
                        progress_tracker.increment_llm_error(task_id)
                        raise

                    # Append LLM response to accumulated prompt
                    llm_content = llm_response.get('content', '')
                    accumulated_prompt += llm_content

                    # Check if response ends with TOOL_CALLS_END
                    if not llm_content.endswith(TOOL_CALLS_END):
                        # No tool call, task is complete
                        log(f"LLM response doesn't end with tool calls - treating as completion")
                        print(f"\n✅ Task {task_id} ({instance_id}): LLM completed without tool call")
                        
                        context.add_message(
                            message_content=llm_content,
                            message_role="assistant",
                            message_type="completion"
                        )
                        
                        result["status"] = "success"
                        progress_tracker.set_status(task_id, "success")
                        break

                    # Append end of sentence marker
                    accumulated_prompt += END_OF_SENTENCE

                    # Parse tool calls
                    try:
                        if enable_timeline:
                            tool_calls = await parser.process_with_timing(llm_response, event_type="parse")
                        else:
                            tool_calls = await parser.process_async(llm_response)

                        if debug and tool_calls and len(tool_calls) > 0:
                            print(f"📝 Parsed tool: {tool_calls[0].get('tool', 'unknown')}")
                            print(f"📝 Tool params: {tool_calls[0].get('parameters', {})}")
                            print("-" * 80)

                        if not tool_calls or len(tool_calls) == 0:
                            progress_tracker.increment_tool_parse_fail(task_id)
                            log(f"Tool parsing returned empty list")
                            print(f"\n⚠️  Task {task_id} ({instance_id}): Tool parsing failed")
                            
                            context.add_message(
                                message_content=llm_content,
                                message_role="assistant",
                                message_type="tool_call"
                            )
                            
                            result["status"] = "failed"
                            result["error"] = "Tool parsing failed"
                            progress_tracker.set_status(task_id, "failed")
                            break

                    except Exception as e:
                        progress_tracker.increment_tool_parse_fail(task_id)
                        log(f"Tool parse error: {str(e)}")
                        print(f"\n❌ Task {task_id} ({instance_id}): Tool parse error: {str(e)}")
                        
                        result["status"] = "failed"
                        result["error"] = str(e)
                        progress_tracker.set_status(task_id, "failed")
                        break

                    context.add_message(
                        message_content=llm_content,
                        message_role="assistant",
                        message_type="tool_call"
                    )

                    # Execute tool
                    tool_call = tool_calls[0]
                    tool_name = tool_call.get('tool', 'unknown')
                    log(f"Executing tool: {tool_name}")
                    log(f"Tool parameters: {json.dumps(tool_call.get('parameters', {}), indent=2)}")

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

                        if isinstance(tool_result, dict) and tool_result.get("status") == "error":
                            progress_tracker.increment_tool_exec_fail(task_id)
                            error_msg = tool_result.get('error', 'Unknown error')
                            print(f"\n⚠️  Task {task_id} ({instance_id}) iter {iteration}: Tool execution error")
                            print(f"   Tool: {tool_name}")
                            print(f"   Error: {error_msg[:300]}")

                    except Exception as e:
                        progress_tracker.increment_tool_exec_fail(task_id)
                        print(f"\n❌ Task {task_id} ({instance_id}): Tool execution exception: {str(e)}")
                        print(f"   Tool: {tool_name}")
                        raise

                    # Format tool output for DeepSeek
                    formatted_result = tool_result.get('result', '')

                    if debug:
                        print(f"🔧 Tool result status: {tool_result.get('status', 'unknown')}")
                        print(f"📤 Formatted result:\n{str(formatted_result)[:500]}")
                        print("=" * 80)

                    # Ensure formatted_result is a string
                    if isinstance(formatted_result, dict):
                        formatted_result = json.dumps(formatted_result)
                    elif not isinstance(formatted_result, str):
                        formatted_result = str(formatted_result)

                    # Append tool output to accumulated prompt
                    tool_output = f"{TOOL_OUTPUT_BEGIN}{formatted_result}{TOOL_OUTPUT_END}"
                    accumulated_prompt += tool_output

                    # Add tool result to context
                    context.add_message(
                        message_content=tool_output,
                        message_role="user",
                        message_type="tool_result"
                    )

                except Exception as e:
                    import traceback
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    log(f"Error in agent loop: {error_msg}")
                    log(f"Traceback: {traceback.format_exc()}")
                    print(f"\n❌ Error in agent loop for task {task_id} ({instance_id}): {error_msg}")
                    result["error"] = error_msg
                    progress_tracker.set_status(task_id, "failed")
                    break

            if iteration >= max_iterations:
                log(f"Reached max iterations ({max_iterations})")
                result["status"] = "max_iterations"
                progress_tracker.set_status(task_id, "max_iter")
            
            # Generate and save patch
            if output_dir:
                try:
                    log(f"Generating patch from testbed changes...")
                    
                    await k8s_executor._execute_kubectl_async("cd /testbed && git add -A")
                    
                    base_commit = instance_data.get('base_commit', None)
                    
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
                            lines = patch.split('\n')
                            log(f"Patch preview (first 5 lines):")
                            for line in lines[:5]:
                                log(f"  {line}")
                            if len(lines) > 5:
                                log(f"  ... ({len(lines) - 5} more lines)")
                            
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
        result["error"] = error_msg
        progress_tracker.set_status(task_id, "failed")
    
    finally:
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
        
        # Close aiohttp session
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
    """Main function to process JSONL file with concurrent execution."""
    
    progress_tracker = ProgressTracker()

    print("=== DeepSeek V3.1 Agent K8S Concurrent Executor ===")
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
        
        if output_dir:
            import os
            existing_patches = 0
            for instance in instances:
                instance_id = instance.get("instance_id", "unknown")
                patch_filepath = os.path.join(output_dir, f"{instance_id}.patch")
                if os.path.exists(patch_filepath):
                    existing_patches += 1
            
            if existing_patches > 0:
                print(f"⏭️  Found {existing_patches} existing patch files, will skip those")
                print(f"🎯 Will process {len(instances) - existing_patches} new instances")
        
    except Exception as e:
        print(f"❌ Error loading JSONL file: {e}")
        return

    # Start progress display
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

            if current_active >= max_concurrent * 0.8:
                print(f"⚡ High concurrency: {current_active}/{max_concurrent} tasks active")

            try:
                instance_id = instance_data.get("instance_id", f"unknown-{index}")
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
    
    await asyncio.sleep(0.5)

    # Stop progress display
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
    skipped = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "skipped")
    exceptions = sum(1 for r in results if isinstance(r, Exception))

    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Timeout: {timeout}")
    print(f"⚠️  Max iterations: {max_iter}")
    print(f"⏭️  Skipped (patch exists): {skipped}")
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
                "max_iterations": "⚠️",
                "skipped": "⏭️"
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

    parser = argparse.ArgumentParser(description="DeepSeek V3.1 Agent K8S Concurrent Executor")
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

    # Set up event loop
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
        
        # Wait for all pending tasks
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user, cleaning up...")
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.close()
