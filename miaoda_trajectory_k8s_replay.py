#!/usr/bin/env python3
"""
Miaoda Trajectory K8S Replay Tool - Replay trajectories from existing rollout results.

This tool replays trajectories from already completed rollouts by loading
LLM outputs from saved context files instead of calling the actual LLM.
The tool execution and parsing remain the same as the original flow.
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

warnings.filterwarnings('ignore', category=ResourceWarning)

sys.path.insert(0, str(Path(__file__).parent / "src"))

from core import (
    ContextEngineeringNode,
    ToolParsingNode,
    K8SToolExecutionNode,
    K8S_AVAILABLE,
    get_timeline
)
from r2e_configs import (
    CUSTOM_TOOL_DESCRIPTIONS,
    parse_xml_action_custom,
    SYSTEM_PROMPT_TEMPLATE,
    QUERY_PROMPT_TEMPLATE,
    DEFAULT_TEMPLATE_VARIABLES
)


# Import Miaoda parser from miaoda_k8s_example.py
def create_miaoda_parser():
    """
    Create Miaoda-style parser for tool calls.
    Parses the function-call format used by Miaoda.

    Format:
    <function=tool_name>
    <parameter=param1>value1</parameter>
    <parameter=param2>value2</parameter>
    </function>
    """
    def parse_tool_calls(llm_response: Dict) -> List[Dict]:
        content = llm_response.get("content", "")

        # Check if content contains function calls
        if "<function=" not in content or "</function>" not in content:
            return []

        tool_calls = []

        # Pattern to match function blocks
        pattern = r'<function=([^>]+)>(.*?)</function>'
        matches = re.findall(pattern, content, re.DOTALL)

        for tool_name, params_block in matches:
            tool_name = tool_name.strip()

            # Parse parameters
            param_pattern = r'<parameter=([^>]+)>(.*?)</parameter>'
            param_matches = re.findall(param_pattern, params_block, re.DOTALL)

            parameters = {}
            for param_name, param_value in param_matches:
                param_name = param_name.strip()
                param_value = param_value.strip()
                parameters[param_name] = param_value

            # Map Miaoda tool names to internal tool names
            internal_tool_name = tool_name
            if tool_name == "bash":
                internal_tool_name = "miaoda_bash_executor"
            elif tool_name == "str_replace_editor":
                internal_tool_name = "miaoda_file_editor"
            elif tool_name == "think":
                internal_tool_name = "miaoda_think"
            elif tool_name == "image_search":
                internal_tool_name = "miaoda_image_search"
            elif tool_name == "api_rag":
                internal_tool_name = "miaoda_api_rag"
            elif tool_name == "api_desc":
                internal_tool_name = "miaoda_api_desc"
            elif tool_name == "supabase_init":
                internal_tool_name = "miaoda_supabase_init"
            elif tool_name == "supabase_apply_migration":
                internal_tool_name = "miaoda_supabase_migration"
            elif tool_name == "supabase_execute_sql":
                internal_tool_name = "miaoda_supabase_sql"
            elif tool_name == "finish":
                internal_tool_name = "miaoda_finish"

            tool_call = {
                "tool": internal_tool_name,
                "parameters": parameters
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
                if status in ["success", "failed", "max_iter", "context_missing"]:
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
        print("=" * 120)
        print("MIAODA TRAJECTORY REPLAY PROGRESS")
        print("=" * 120)

        # Table header
        header = f"{'Task':<6} {'Instance ID':<40} {'Iter':<5} {'Time(s)':<8} {'TPF':<5} {'TEF':<5} {'Status':<12}"
        print(header)
        print("-" * 120)

        # Sort by task_id
        snapshot.sort(key=lambda t: t.task_id)

        # Print each task
        for task in snapshot:
            status_emoji = {
                "initializing": "⏳",
                "running": "🔄",
                "success": "✅",
                "failed": "❌",
                "max_iter": "⚠️",
                "context_missing": "🚫"
            }.get(task.status, "❓")

            row = (
                f"{task.task_id:<6} "
                f"{task.instance_id:<40} "
                f"{task.iterations:<5} "
                f"{task.elapsed_time():<8.1f} "
                f"{task.tool_parse_fail:<5} "
                f"{task.tool_exec_fail:<5} "
                f"{status_emoji} {task.status:<10}"
            )
            print(row)

        print("-" * 120)

        # Summary stats
        total = len(snapshot)
        initializing = sum(1 for t in snapshot if t.status == "initializing")
        running = sum(1 for t in snapshot if t.status == "running")
        completed = total - running - initializing

        total_parse_fails = sum(t.tool_parse_fail for t in snapshot)
        total_exec_fails = sum(t.tool_exec_fail for t in snapshot)

        print(f"Total: {total} | Initializing: {initializing} | Running: {running} | Completed: {completed} | Parse fails: {total_parse_fails} | Exec fails: {total_exec_fails}")
        print("=" * 120)

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


def load_context_trajectory(output_dir: str, instance_id: str) -> Optional[List[Dict]]:
    """
    Load trajectory from context file.

    Args:
        output_dir: Directory containing context files
        instance_id: Instance ID to load

    Returns:
        List of message dictionaries from the context file, or None if not found
    """
    context_file = os.path.join(output_dir, f"{instance_id}.context")

    if not os.path.exists(context_file):
        return None

    messages = []
    try:
        with open(context_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    msg = json.loads(line)
                    messages.append(msg)
        return messages
    except Exception as e:
        print(f"Error loading context file {context_file}: {e}")
        return None


async def process_single_instance(
    instance_data: Dict,
    pod_suffix: str,
    task_id: int,
    progress_tracker: ProgressTracker,
    output_dir: str,
    new_output_dir: str = None,
    enable_timeline: bool = False,
    debug: bool = False,
    cpu_request: str = "0.3",
    memory_request: str = "1Gi",
    tool_timeout: float = 300.0) -> Dict:
    """Process a single instance by replaying trajectory from context file.

    Args:
        instance_data: Data for a single instance from JSONL
        pod_suffix: Unique suffix for the pod name
        task_id: Task ID for progress tracking
        progress_tracker: Progress tracker instance
        output_dir: Directory containing existing context files to replay
        new_output_dir: Directory to save new context and log files (optional)
        enable_timeline: Enable timeline tracking
        debug: Enable debug mode
        cpu_request: CPU resource request
        memory_request: Memory resource request
        tool_timeout: Tool execution timeout in seconds

    Returns:
        Result dictionary with instance_id and execution status
    """
    instance_id = instance_data.get("qid", instance_data.get("instance_id", "unknown"))
    extra_info = instance_data.get("extra_info", {})
    app_id = f"app-{instance_id}"
    requirement_type = extra_info.get("requirement_type", "Web")

    # Register task with progress tracker
    progress_tracker.create_task(task_id, instance_id)
    progress_tracker.set_status(task_id, "initializing")

    # Load trajectory from context file
    trajectory = load_context_trajectory(output_dir, instance_id)

    if trajectory is None:
        print(f"⏭️  Task {task_id} ({instance_id}): Context file not found, skipping")
        progress_tracker.set_status(task_id, "context_missing")
        return {
            "task_id": task_id,
            "instance_id": instance_id,
            "status": "skipped",
            "reason": "context_missing"
        }

    # Setup output files if new_output_dir is specified
    log_file = None
    if new_output_dir:
        os.makedirs(new_output_dir, exist_ok=True)
        log_file_path = os.path.join(new_output_dir, f"{instance_id}.log")
        log_file = open(log_file_path, 'w', encoding='utf-8')

    # Derive pod name using app_id
    import random
    import uuid
    random_uuid = str(uuid.uuid4())[:6]
    pod_name = f"{app_id}-{random_uuid}".lower()

    if len(pod_name) > 200:
        pod_name = f"{pod_name[:100]}-{random_uuid}"

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
        log(f"=== Starting Miaoda trajectory replay for instance: {instance_id} ===")
        log(f"Task ID: {task_id}")
        log(f"Pod name: {pod_name}")
        log(f"Tool timeout: {tool_timeout}s")
        log(f"Loaded {len(trajectory)} messages from context file")

        execution_start_time = time.time()

        # Context Engineering Node
        context = ContextEngineeringNode(name=f"MiaodaReplayContext-{pod_suffix}", timeline_enabled=enable_timeline)

        # Tool Parsing Node (Miaoda-style parser)
        parser = ToolParsingNode(
            name=f"MiaodaReplayParser-{pod_suffix}",
            parse_function=create_miaoda_parser(),
            timeline_enabled=enable_timeline
        )

        # Use Miaoda image
        image = "iregistry.baidu-int.com/acg-agi/reward:miaoda_reward_1106"

        log(f"Using image: {image}")
        log(f"App ID: {app_id}")
        log(f"Requirement type: {requirement_type}")

        # Use async context manager for automatic cleanup
        async with K8SToolExecutionNode(
                name=f"MiaodaReplayK8SExecutor-{pod_suffix}",
                namespace="rl-training",
                kubeconfig_path="./swe-bench-verified-workspace/config_cce_new",
                image=image,
                pod_name=pod_name,
                environment={
                    "PATH": "/usr/local/jupyter:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin:/root/.local/share/mise/installs/python/latest/bin:/root/.local/share/mise/installs/node/latest/bin:/pnpm-store",
                    "PYTHONPATH": "/workspace",
                    "PYTHONIOENCODING": "utf-8",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "http_proxy": "http://mt:mtstudio@10.224.65.111:8234",
                    "https_proxy": "http://mt:mtstudio@10.224.65.111:8234",
                    "PIP_INDEX_URL": "http://pip.baidu.com/pypi/simple",
                    "PIP_TRUSTED_HOST": "pip.baidu.com"
                },
                cpu_request=cpu_request,
                memory_request=memory_request,
                timeline_enabled=enable_timeline,
                tool_timeout=tool_timeout
        ) as k8s_executor:
            log(f"K8S executor initialized")

            # Initialize Miaoda pod environment
            try:
                log(f"Initializing Miaoda pod environment...")
                await k8s_executor._execute_kubectl_async("rm -f /workspace/node_modules")

                if requirement_type == "Mini Program":
                    await k8s_executor._execute_kubectl_async("ln -sf /data/wechat/node_modules /workspace/node_modules")
                    await k8s_executor._execute_kubectl_async(f"mv /code-template/taro-weapp-template /workspace/{app_id}")
                else:
                    await k8s_executor._execute_kubectl_async("ln -sf /data/shadcn/node_modules /workspace/node_modules")
                    await k8s_executor._execute_kubectl_async(f"mv /code-template/react-shadcn-lite-template /workspace/{app_id}")

                await k8s_executor._execute_kubectl_async("pip install chardet --break-system-packages")

                log(f"Miaoda pod initialization completed")
            except Exception as e:
                log(f"Warning: Pod initialization encountered error: {e}")

            # Register Miaoda tools
            k8s_executor.register_tool("miaoda_bash_executor", "src/tools/miaoda/bash_func.py")
            k8s_executor.register_tool("miaoda_file_editor", "src/tools/miaoda/file_editor.py")
            k8s_executor.register_tool("miaoda_finish", "src/tools/miaoda/finish.py", execution_mode="local")
            k8s_executor.register_tool("miaoda_think", "src/tools/miaoda/think.py", execution_mode="local")
            k8s_executor.register_tool("miaoda_image_search", "src/tools/miaoda/image_search.py")
            k8s_executor.register_tool("miaoda_api_rag", "src/tools/miaoda/api_rag.py")
            k8s_executor.register_tool("miaoda_api_desc", "src/tools/miaoda/api_desc.py")
            k8s_executor.register_tool("miaoda_supabase_init", "src/tools/miaoda/supabase_init.py", execution_mode="local")
            k8s_executor.register_tool("miaoda_supabase_migration", "src/tools/miaoda/supabase_migration.py", execution_mode="local")
            k8s_executor.register_tool("miaoda_supabase_sql", "src/tools/miaoda/supabase_sql_execution.py", execution_mode="local")

            # Replay trajectory
            iteration = 0
            trajectory_idx = 0

            # Mark as running after pod is ready
            progress_tracker.set_status(task_id, "running")

            # Process messages from trajectory
            while trajectory_idx < len(trajectory):
                current_msg = trajectory[trajectory_idx]
                msg_role = current_msg.get("role")

                # Add system and user messages to context
                if msg_role == "system":
                    context.add_message(
                        message_content=current_msg.get("content", ""),
                        message_role="system",
                        message_type="system_prompt"
                    )
                    log(f"Added system message to context")
                    trajectory_idx += 1
                    continue

                elif msg_role == "user":
                    # Check if this is a query or tool_result
                    # The first user message is the query, subsequent ones are tool results
                    if iteration == 0:
                        context.add_message(
                            message_content=current_msg.get("content", ""),
                            message_role="user",
                            message_type="query"
                        )
                        log(f"Added user query to context")
                    else:
                        context.add_message(
                            message_content=current_msg.get("content", ""),
                            message_role="user",
                            message_type="tool_result"
                        )
                        log(f"Added tool result to context")
                    trajectory_idx += 1
                    continue

                elif msg_role == "assistant":
                    iteration += 1
                    progress_tracker.update_iteration(task_id, iteration)
                    log(f"\n=== Iteration {iteration} (from trajectory) ===")

                    # Use the assistant message content as LLM response
                    llm_content = current_msg.get("content", "")
                    llm_response = {"content": llm_content}

                    log(f"Loaded LLM response from trajectory: {llm_content[:200]}...")

                    if debug:
                        print(f"\n🤖 Task {task_id} iter {iteration} - Replaying LLM Response:")
                        print(f"{llm_content}")
                        print("-" * 80)

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
                            # No tool call, task is complete
                            log(f"No tool calls parsed - treating as completion")
                            log(llm_content)
                            print(f"\n✅ Task {task_id} ({instance_id}): Completed without tool call")

                            context.add_message(
                                message_content=llm_content,
                                message_role="assistant",
                                message_type="completion"
                            )

                            result["status"] = "success"
                            progress_tracker.set_status(task_id, "success")
                            trajectory_idx += 1
                            break

                    except Exception as e:
                        progress_tracker.increment_tool_parse_fail(task_id)
                        log(f"Tool parse error: {str(e)}")
                        print(llm_content)
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

                    # Check if finish tool was called
                    if tool_name == "miaoda_finish":
                        log(f"Finish tool called - task complete")
                        print(f"\n✅ Task {task_id} ({instance_id}): Finish tool called")
                        result["status"] = "success"
                        progress_tracker.set_status(task_id, "success")
                        trajectory_idx += 1
                        break

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
                            print(f"   Error: {error_msg}")

                    except Exception as e:
                        progress_tracker.increment_tool_exec_fail(task_id)
                        print(f"\n❌ Task {task_id} ({instance_id}): Tool execution exception: {str(e)}")
                        print(f"   Tool: {tool_name}")
                        raise

                    # Format tool output (note: we don't use it from trajectory, we use the actual execution result)
                    formatted_result = tool_result.get('result', '')

                    if debug:
                        print(f"🔧 Tool result status: {tool_result.get('status', 'unknown')}")
                        print(f"📤 Formatted result:\n{str(formatted_result)}")
                        print("=" * 80)

                    # Ensure formatted_result is a string
                    if isinstance(formatted_result, dict):
                        formatted_result = json.dumps(formatted_result)
                    elif not isinstance(formatted_result, str):
                        formatted_result = str(formatted_result)

                    trajectory_idx += 1

                else:
                    # Unknown message role, skip
                    log(f"Warning: Unknown message role: {msg_role}")
                    trajectory_idx += 1
                    continue

            if iteration >= len([m for m in trajectory if m.get("role") == "assistant"]):
                log(f"Completed all iterations from trajectory")
                if result["status"] != "success":
                    result["status"] = "completed_trajectory"
                    progress_tracker.set_status(task_id, "success")

            # Generate and save patch
            if new_output_dir:
                try:
                    log(f"Generating patch from workspace changes...")

                    await k8s_executor._execute_kubectl_async(f"cd /workspace/{app_id} && git add -A")

                    base_commit = extra_info.get('base_commit', None)

                    if requirement_type == "Mini Program":
                        base_commit = "9b0ed8ea29ec81a8b563360928b164f25161acac"
                    else:
                        base_commit = "413c2a93c661f4896acc1a35e59233e02bf9c924"

                    if base_commit:
                        log(f"Generating patch against base_commit: {base_commit}")
                        patch_output, exit_code = await k8s_executor._execute_kubectl_async(
                            f"cd /workspace/{app_id} && git diff {base_commit}"
                        )
                    else:
                        log("Generating patch against staged changes")
                        patch_output, exit_code = await k8s_executor._execute_kubectl_async(
                            f"cd /workspace/{app_id} && git diff --cached"
                        )

                    if exit_code == "0" or exit_code == 0:
                        patch = patch_output.strip()
                        log(f"Patch generated successfully, size: {len(patch)} characters")

                        if patch:
                            patch_filepath = os.path.join(new_output_dir, f"{instance_id}.patch")
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

            # Calculate and save reward (similar to original)
            if new_output_dir:
                try:
                    log(f"Calculating Miaoda reward...")

                    prd = instance_data.get("prd_description", "")
                    function_list = extra_info.get("function_list", [])
                    func_num = len(function_list) if function_list else 1

                    func_json = json.dumps(function_list, ensure_ascii=False)
                    func_escaped = func_json.replace("'", "'\"'\"'")

                    exec_command = f"""timeout 300 bash /workspace/text_reward_model/run.sh --qid "{instance_id}" --prd_id "825" --repo_path /workspace/{app_id} --prd_description "{prd}" --func_list '{func_escaped}'"""

                    heredoc_cmd = f"""cat > /workspace/test.sh << 'EOF'
{exec_command}
EOF"""

                    await k8s_executor._execute_kubectl_async(heredoc_cmd)
                    await k8s_executor._execute_kubectl_async("chmod +x /workspace/test.sh")

                    log(f"Executing reward calculation command...")
                    output, error_code = await k8s_executor._execute_kubectl_async("timeout 300 bash /workspace/test.sh")
                    log(f"Reward calculation output: {output[:200]}...")
                    log(f"Reward calculation error code: {error_code}")

                    score_output, score_error = await k8s_executor._execute_kubectl_async(
                        "timeout 60 cat /workspace/text_reward_model/result_dir/result_score.jsonl"
                    )

                    reward_result = {
                        "instance_id": instance_id,
                        "app_id": app_id,
                        "reward": 0.0,
                        "status": "error",
                        "raw_output": score_output if score_output else "",
                        "run_sh_output": output if output else "",
                        "run_sh_error_code": error_code
                    }

                    if score_error == "0" or score_error == 0:
                        try:
                            scores = []
                            for func_line in score_output.strip().split("\n"):
                                if func_line.strip():
                                    function_score = json.loads(func_line)
                                    scores.append(function_score.get("function_score", 0))

                            if scores:
                                reward = sum(scores) / len(scores)
                            else:
                                reward = 0.0

                            reward_result["reward"] = reward
                            reward_result["status"] = "success"
                            reward_result["scores"] = scores
                            log(f"Reward calculated successfully: {reward}")
                        except Exception as e:
                            log(f"Error parsing reward scores: {e}")
                            reward_result["error"] = str(e)
                    else:
                        log(f"Reward calculation failed with error code: {score_error}")
                        reward_result["error"] = f"Command failed with exit code {score_error}"

                    reward_filepath = os.path.join(new_output_dir, f"{instance_id}.reward")
                    with open(reward_filepath, 'w', encoding='utf-8') as f:
                        json.dump(reward_result, f, ensure_ascii=False, indent=2)
                    log(f"Saved reward to: {reward_filepath}")
                    result["reward_file"] = reward_filepath
                    result["reward"] = reward_result["reward"]

                except Exception as e:
                    log(f"Error calculating reward: {str(e)}")
                    reward_result = {
                        "instance_id": instance_id,
                        "app_id": app_id,
                        "reward": 0.0,
                        "status": "error",
                        "error": str(e)
                    }
                    try:
                        reward_filepath = os.path.join(new_output_dir, f"{instance_id}.reward")
                        with open(reward_filepath, 'w', encoding='utf-8') as f:
                            json.dump(reward_result, f, ensure_ascii=False, indent=2)
                        result["reward_file"] = reward_filepath
                        result["reward"] = 0.0
                    except:
                        pass

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

        # Save context to file if new_output_dir is specified
        if new_output_dir and 'context' in locals():
            try:
                context_file_path = os.path.join(new_output_dir, f"{instance_id}.context")
                with open(context_file_path, 'w', encoding='utf-8') as f:
                    messages = context.get_llm_context()
                    for msg in messages:
                        f.write(json.dumps(msg, ensure_ascii=False) + '\n')
                if log_file:
                    log(f"Context saved to {context_file_path}")
            except Exception as e:
                if log_file:
                    log(f"Error saving context: {e}")

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
    output_dir: str,
    max_concurrent: int = 3,
    new_output_dir: str = None,
    enable_timeline: bool = False,
    debug: bool = False,
    cpu_request: str = "0.3",
    memory_request: str = "1Gi",
    tool_timeout: float = 300.0
):
    """Main function to replay trajectories from context files."""

    progress_tracker = ProgressTracker()

    print("=== Miaoda Trajectory K8S Replay ===")
    print(f"📁 JSONL file: {jsonl_file}")
    print(f"📂 Output directory (source): {output_dir}")
    if new_output_dir:
        print(f"📂 New output directory: {new_output_dir}")
    print(f"🔧 Max concurrent: {max_concurrent}")
    print(f"⏱️  Timeline tracking: {'ENABLED' if enable_timeline else 'DISABLED'}")
    print(f"🐛 Debug mode: {'ENABLED' if debug else 'DISABLED'}")
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
                    new_output_dir=new_output_dir,
                    enable_timeline=enable_timeline,
                    debug=debug,
                    cpu_request=cpu_request,
                    memory_request=memory_request,
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

    successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") in ["success", "completed_trajectory"])
    failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
    skipped = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "skipped")
    exceptions = sum(1 for r in results if isinstance(r, Exception))

    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️  Skipped (context missing): {skipped}")
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
                "completed_trajectory": "✅",
                "failed": "❌",
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

        if new_output_dir:
            try:
                timeline_file = os.path.join(new_output_dir, "timeline_profile.json")
                with open(timeline_file, 'w', encoding='utf-8') as f:
                    json.dump(timeline_data, f, indent=2, ensure_ascii=False)
                print(f"\n✓ Timeline saved to: {timeline_file}")
            except Exception as e:
                print(f"\n✗ Error saving timeline: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Miaoda Trajectory K8S Replay Tool")
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to JSONL file containing instances to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing existing context files to replay (required)"
    )
    parser.add_argument(
        "--new-output-dir",
        type=str,
        default=None,
        help="Directory to save new context, logs, patches and rewards (optional)"
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
                output_dir=args.output_dir,
                max_concurrent=args.concurrent,
                new_output_dir=args.new_output_dir,
                enable_timeline=args.timeline,
                debug=args.debug,
                cpu_request=args.cpu,
                memory_request=args.memory,
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
