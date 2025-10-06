"""
K8S Tool Execution Node implementation.

This module provides a Kubernetes-based tool execution node that executes
tools in Kubernetes pods using the Kodo library.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from .tool_execution_node import ToolExecutionNode

try:
    from kodo import ContainerRunner
    KODO_AVAILABLE = True
except ImportError:
    KODO_AVAILABLE = False
    ContainerRunner = None


class K8SToolExecutionNode(ToolExecutionNode):
    """
    Kubernetes-based tool execution node.

    Executes tools in Kubernetes pods instead of local processes.
    Uses the Kodo library for Kubernetes operations.
    """

    def __init__(self,
                 name: str = None,
                 namespace: str = "default",
                 kubeconfig_path: str = None,
                 image: str = "python:3.9-slim",
                 node_selector: Dict[str, str] = None,
                 pod_name: str = None,
                 environment: Dict[str, str] = None,
                 image_pull_policy: str = "IfNotPresent",
                 timeline_enabled: bool = False,
                 timeout: float = None):
        """
        Initialize the K8S Tool Execution Node.

        Args:
            name: Optional name for the node
            namespace: Kubernetes namespace
            kubeconfig_path: Path to kubeconfig file
            image: Container image for tool execution pod
            node_selector: Optional node selector for pod placement
            pod_name: Optional pod name (auto-generated if not provided)
            environment: Optional environment variables for the pod
            image_pull_policy: Image pull policy (Always, Never, IfNotPresent)
            timeline_enabled: Enable automatic timeline tracking for this node
            timeout: Timeout in seconds for tool execution (None = no timeout)
        """
        if not KODO_AVAILABLE:
            raise ImportError(
                "Kodo library is required for K8S tool execution. "
                "Install it with: pip install kodo"
            )

        super().__init__(name, timeline_enabled=timeline_enabled, timeout=timeout)

        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path
        self.image = image
        self.node_selector = node_selector or {}
        self.environment = environment or {}
        self.image_pull_policy = image_pull_policy

        # Initialize Kodo container runner
        self.runner = ContainerRunner(
            backend="kubernetes",
            namespace=self.namespace,
            kubeconfig_path=self.kubeconfig_path
        )

        # Generate pod name if not provided
        self.pod_name = pod_name or self._generate_pod_name()

        # Pod reference (will be created lazily on first use)
        self.pod = None
        self._pod_ready = asyncio.Event()
        self._pod_creation_started = False

        self.logger.info(f"K8S Tool Execution Node initialized (pod will be created on first use): {self.pod_name}")

    async def _ensure_pod_ready(self) -> None:
        """
        Ensure pod is created and ready (lazy initialization).

        Uses asyncio.Event to ensure only one coroutine creates the pod.
        """
        # If pod already ready, return immediately
        if self._pod_ready.is_set():
            return

        # If another coroutine is creating the pod, wait for it
        if self._pod_creation_started:
            await self._pod_ready.wait()
            return

        # This coroutine will create the pod
        self._pod_creation_started = True

        try:
            # Run synchronous pod creation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._create_pod)

            # Signal that pod is ready
            self._pod_ready.set()

        except Exception as e:
            self.logger.error(f"Failed to create pod: {str(e)}")
            self._pod_creation_started = False
            raise

    async def _execute_single_tool_async(self, tool_call: Dict) -> Dict:
        """
        Execute a single tool call in a Kubernetes pod asynchronously.

        Args:
            tool_call: Tool call dictionary

        Returns:
            Execution result dictionary
        """
        # Ensure pod is ready before executing
        await self._ensure_pod_ready()
        tool_name = (tool_call.get("tool") or
                    tool_call.get("function") or
                    tool_call.get("name") or
                    tool_call.get("action"))

        arguments = (tool_call.get("arguments") or
                    tool_call.get("parameters") or
                    tool_call.get("args") or
                    tool_call.get("input") or
                    {})

        if not isinstance(arguments, dict):
            arguments = {"input": arguments}

        if not tool_name:
            return {
                "error": "No tool name specified",
                "tool_call": tool_call,
                "status": "error"
            }

        if tool_name not in self.tools:
            self.logger.warning(f"Tool not found: {tool_name}")
            return {
                "error": f"Tool '{tool_name}' not registered",
                "tool": tool_name,
                "status": "error"
            }

        tool = self.tools[tool_name]
        try:
            self.logger.info(f"Executing tool '{tool_name}' in K8S pod: {self.pod_name}")

            command = self._build_tool_command(tool_name, arguments)

            # Execute kubectl command asynchronously (no GIL blocking)
            output, kodo_status = await self._execute_kubectl_async(command)

            import json
            try:
                result_data = json.loads(output)
                stdout = result_data.get("stdout", "")
                stderr = result_data.get("stderr", "")
                exit_code = result_data.get("exit_code", 0)
            except (json.JSONDecodeError, AttributeError) as e:
                self.logger.warning(f"Failed to parse tool output as JSON: {e}")
                self.logger.debug(f"Raw output: {output[:200]}")
                stdout = str(output) if output is not None else ""
                stderr = ""
                try:
                    exit_code = int(kodo_status) if kodo_status and kodo_status != "0" else 0
                except (ValueError, TypeError):
                    exit_code = 0 if str(kodo_status) == "0" else 1

            raw_result = {
                "output": stdout,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "returncode": exit_code,
                "pod_name": self.pod_name,
                "success": exit_code == 0
            }

            parsed_result = tool.parse_result(raw_result)

            result = {
                "tool": tool_name,
                "result": parsed_result,
                "status": "success" if exit_code == 0 else "error",
                "execution_mode": "k8s",
                "pod": self.pod_name,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code
            }

            if exit_code != 0:
                error_msg = stderr.strip() if stderr.strip() else None
                if not error_msg and isinstance(parsed_result, dict):
                    error_msg = parsed_result.get("output", "")

                result["error"] = error_msg or f"Command failed with exit code {exit_code}"
                result["error_type"] = "CommandExecutionError"

            if tool_name == "stop" and isinstance(parsed_result, dict):
                result["status"] = parsed_result.get("status", "success")

            return result

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self.logger.error(f"Error executing tool {tool_name} in K8S: {str(e)}")
            self.logger.error(f"Full traceback: {error_detail}")

            error_msg = str(e) if str(e) else f"{type(e).__name__}: No error message"

            error_context = f"Tool '{tool_name}' failed in pod '{self.pod_name}'"
            if 'command' in locals():
                error_context += f"\nCommand preview: {command[:100]}..."

            full_error = f"{error_context}\nError: {error_msg}"

            print(f"\n❌ K8S Execution Error:")
            print(f"  Tool: {tool_name}")
            print(f"  Pod: {self.pod_name}")
            print(f"  Error: {error_msg}")
            print(f"  Type: {type(e).__name__}")

            return {
                "tool": tool_name,
                "error": full_error,
                "error_type": type(e).__name__,
                "error_message": error_msg,
                "traceback": error_detail,
                "status": "error",
                "execution_mode": "k8s",
                "pod": self.pod_name
            }

    def _execute_single_tool(self, tool_call: Dict) -> Dict:
        """
        Execute a single tool call in a Kubernetes pod.

        Args:
            tool_call: Tool call dictionary

        Returns:
            Execution result dictionary
        """
        # Extract tool name and arguments
        tool_name = (tool_call.get("tool") or
                    tool_call.get("function") or
                    tool_call.get("name") or
                    tool_call.get("action"))

        arguments = (tool_call.get("arguments") or
                    tool_call.get("parameters") or
                    tool_call.get("args") or
                    tool_call.get("input") or
                    {})

        if not isinstance(arguments, dict):
            arguments = {"input": arguments}

        # Check if tool is registered
        if not tool_name:
            return {
                "error": "No tool name specified",
                "tool_call": tool_call,
                "status": "error"
            }

        if tool_name not in self.tools:
            self.logger.warning(f"Tool not found: {tool_name}")
            return {
                "error": f"Tool '{tool_name}' not registered",
                "tool": tool_name,
                "status": "error"
            }

        # Execute tool in K8S pod
        tool = self.tools[tool_name]
        try:
            self.logger.info(f"Executing tool '{tool_name}' in K8S pod: {self.pod_name}")

            # Build command to execute (returns JSON with stdout/stderr/exit_code)
            command = self._build_tool_command(tool_name, arguments)

            # Execute command in the persistent pod
            output, kodo_status = self.runner.execute_command(self.pod, command)

            # Parse JSON output to extract stdout, stderr, exit_code
            import json
            try:
                result_data = json.loads(output)
                stdout = result_data.get("stdout", "")
                stderr = result_data.get("stderr", "")
                exit_code = result_data.get("exit_code", 0)
            except (json.JSONDecodeError, AttributeError) as e:
                # Fallback if output is not valid JSON
                self.logger.warning(f"Failed to parse tool output as JSON: {e}")
                self.logger.debug(f"Raw output: {output[:200]}")
                stdout = str(output) if output is not None else ""
                stderr = ""
                # Try to extract exit code from kodo_status
                try:
                    exit_code = int(kodo_status) if kodo_status and kodo_status != "0" else 0
                except (ValueError, TypeError):
                    exit_code = 0 if str(kodo_status) == "0" else 1

            # Build raw_result with both stdout and stderr
            # Provide both formats for compatibility with different result_parsers
            raw_result = {
                "output": stdout,      # For parsers expecting "output"
                "stdout": stdout,      # For parsers expecting "stdout"
                "stderr": stderr,      # Stderr is now available
                "exit_code": exit_code,
                "returncode": exit_code,  # For parsers expecting "returncode"
                "pod_name": self.pod_name,
                "success": exit_code == 0
            }

            parsed_result = tool.parse_result(raw_result)

            # Determine status: use parser's status if provided, otherwise default
            if isinstance(parsed_result, dict) and "status" in parsed_result:
                status = parsed_result.get("status")
            else:
                status = "success" if exit_code == 0 else "error"

            # Format response - always include stdout/stderr/exit_code
            result = {
                "tool": tool_name,
                "result": parsed_result,
                "status": status,
                "execution_mode": "k8s",
                "pod": self.pod_name,
                "stdout": stdout,        # Always provide subprocess stdout
                "stderr": stderr,        # Always provide subprocess stderr
                "exit_code": exit_code  # Always provide exit code
            }

            # If execution failed, add error information
            if exit_code != 0:
                # Try to get error message from stderr or parsed result
                error_msg = stderr.strip() if stderr.strip() else None
                if not error_msg and isinstance(parsed_result, dict):
                    error_msg = parsed_result.get("output", "")

                result["error"] = error_msg or f"Command failed with exit code {exit_code}"
                result["error_type"] = "CommandExecutionError"

            # Check if this is the stop tool
            if tool_name == "stop" and isinstance(parsed_result, dict):
                result["status"] = parsed_result.get("status", "success")

            return result

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self.logger.error(f"Error executing tool {tool_name} in K8S: {str(e)}")
            self.logger.error(f"Full traceback: {error_detail}")

            # Build detailed error message
            error_msg = str(e) if str(e) else f"{type(e).__name__}: No error message"

            # Add context to error message
            error_context = f"Tool '{tool_name}' failed in pod '{self.pod_name}'"
            if 'command' in locals():
                error_context += f"\nCommand preview: {command[:100]}..."

            full_error = f"{error_context}\nError: {error_msg}"

            print(f"\n❌ K8S Execution Error:")
            print(f"  Tool: {tool_name}")
            print(f"  Pod: {self.pod_name}")
            print(f"  Error: {error_msg}")
            print(f"  Type: {type(e).__name__}")

            return {
                "tool": tool_name,
                "error": full_error,
                "error_type": type(e).__name__,
                "error_message": error_msg,
                "traceback": error_detail,
                "status": "error",
                "execution_mode": "k8s",
                "pod": self.pod_name
            }

    async def _execute_kubectl_async(self, command: str) -> tuple:
        """
        Execute kubectl command asynchronously without GIL blocking.

        Args:
            command: Command to execute in the pod

        Returns:
            Tuple of (stdout, exit_code)
        """
        # Build kubectl exec command
        kubectl_cmd = [
            "kubectl", "exec", self.pod_name,
            "-n", self.namespace,
            "--",
            "sh", "-c", command
        ]

        try:
            # Use asyncio subprocess for true concurrency
            process = await asyncio.create_subprocess_exec(
                *kubectl_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')

            # Log errors if present
            if stderr_str:
                self.logger.debug(f"kubectl stderr: {stderr_str[:200]}")

            # Return in Kodo-compatible format (output, status)
            return (stdout_str, str(process.returncode))

        except Exception as e:
            self.logger.error(f"kubectl exec failed: {str(e)}")
            raise

    def _create_pod(self) -> None:
        """
        Create the persistent pod for tool execution.

        This pod will be used throughout the node's lifecycle.
        First deletes any existing pod with the same name to ensure idempotency.
        """
        self.logger.info(f"Creating persistent K8S pod: {self.pod_name}")

        try:
            # Delete existing pod with same name if it exists
            self._cleanup_existing_pod()

            # Add PYTHONPATH to environment
            env = self.environment.copy()
            env['PYTHONPATH'] = '/workspace:$PYTHONPATH'

            self.pod = self.runner.start_container(
                self.image,
                name=self.pod_name,
                environment=env,
                node_selector=self.node_selector
            )
            self.logger.info(f"Pod {self.pod_name} created successfully")

            # Copy tools to the pod
            self._copy_tools_to_pod()

        except Exception as e:
            self.logger.error(f"Failed to create pod {self.pod_name}: {str(e)}")
            raise

    def _cleanup_existing_pod(self) -> None:
        """
        Clean up any existing pod with the same name.

        This ensures idempotency - if a previous run left a pod with the same name,
        we delete it before creating a new one.
        """
        try:
            import subprocess

            # Check if pod exists
            check_cmd = [
                "kubectl", "get", "pod", self.pod_name,
                "-n", self.namespace,
                "--ignore-not-found"
            ]

            result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            # If pod exists (non-empty output), delete it
            if result.stdout.strip():
                self.logger.info(f"Found existing pod {self.pod_name}, deleting...")

                delete_cmd = [
                    "kubectl", "delete", "pod", self.pod_name,
                    "-n", self.namespace,
                    "--force", "--grace-period=0"
                ]

                subprocess.run(
                    delete_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                self.logger.info(f"Deleted existing pod {self.pod_name}")

                # Wait a bit for pod to be fully deleted
                import time
                time.sleep(2)
            else:
                self.logger.debug(f"No existing pod {self.pod_name} found")

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout while checking/deleting existing pod {self.pod_name}")
        except Exception as e:
            self.logger.warning(f"Error checking/deleting existing pod {self.pod_name}: {str(e)}")
            # Don't raise - we'll try to create the pod anyway

    def _copy_tools_to_pod(self) -> None:
        """
        Copy the tools directory to the pod by copying individual files.
        """
        import os
        import glob

        self.logger.info("Copying tools to pod...")

        try:
            # Get the tools directory path
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tools_dir = os.path.join(current_dir, "tools")

            if not os.path.exists(tools_dir):
                self.logger.warning(f"Tools directory not found: {tools_dir}")
                return

            # Create directory structure
            self.runner.execute_command(self.pod, "mkdir -p /workspace/src/tools/r2e")
            self.runner.execute_command(self.pod, "mkdir -p /workspace/src/tools/mini_swe")

            # Copy all Python files from tools/r2e
            r2e_dir = os.path.join(tools_dir, "r2e")
            if os.path.exists(r2e_dir):
                for py_file in glob.glob(os.path.join(r2e_dir, "*.py")):
                    filename = os.path.basename(py_file)
                    pod_path = f"/workspace/src/tools/r2e/{filename}"
                    self._copy_file_to_pod(py_file, pod_path)
                    self.logger.debug(f"Copied {filename} to pod")

            # Copy all Python files from tools/mini_swe
            mini_swe_dir = os.path.join(tools_dir, "mini_swe")
            if os.path.exists(mini_swe_dir):
                for py_file in glob.glob(os.path.join(mini_swe_dir, "*.py")):
                    filename = os.path.basename(py_file)
                    pod_path = f"/workspace/src/tools/mini_swe/{filename}"
                    self._copy_file_to_pod(py_file, pod_path)
                    self.logger.debug(f"Copied {filename} to pod")

            # Copy base_tool.py if exists
            base_tool_path = os.path.join(tools_dir, "base_tool.py")
            if os.path.exists(base_tool_path):
                self._copy_file_to_pod(base_tool_path, "/workspace/src/tools/base_tool.py")

            # Copy __init__.py files
            for init_file in ["__init__.py", "r2e/__init__.py", "mini_swe/__init__.py"]:
                local_init = os.path.join(tools_dir, init_file)
                if os.path.exists(local_init):
                    pod_init = f"/workspace/src/tools/{init_file}"
                    self._copy_file_to_pod(local_init, pod_init)

            self.logger.info("Tools copied to pod at /workspace/src/tools")

        except Exception as e:
            self.logger.warning(f"Failed to copy tools to pod: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            # Continue anyway - tools might not be needed

    def _copy_file_to_pod(self, local_path: str, pod_path: str) -> None:
        """
        Copy a single file to the pod using heredoc.

        Args:
            local_path: Path to local file
            pod_path: Destination path in pod
        """
        try:
            # Read file content
            with open(local_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Use base64 encoding to avoid shell escaping issues
            import base64
            content_b64 = base64.b64encode(content.encode('utf-8')).decode('utf-8')

            # Create file in pod using base64 decoding
            cmd = f"echo '{content_b64}' | base64 -d > {pod_path}"
            self.runner.execute_command(self.pod, cmd)

        except Exception as e:
            self.logger.warning(f"Failed to copy {local_path} to pod: {str(e)}")

    def _generate_pod_name(self) -> str:
        """
        Generate a unique pod name for this node.

        Returns:
            Unique pod name
        """
        import time
        import random
        timestamp = int(time.time())
        random_suffix = random.randint(1000, 9999)
        node_name = self.name.lower().replace(" ", "-") if self.name else "tool-execution"
        return f"{node_name}-{timestamp}-{random_suffix}"

    def _local_to_pod_path(self, local_path: str) -> str:
        """
        Convert local script path to pod path.

        The local path is what developers use when registering tools.
        The pod path is the internal implementation detail of where files
        are copied to in the Kubernetes pod.

        Args:
            local_path: Local file path (e.g., "src/tools/r2e/bash_func.py")

        Returns:
            Pod path (e.g., "/workspace/src/tools/r2e/bash_func.py")

        Examples:
            >>> _local_to_pod_path("src/tools/r2e/bash_func.py")
            "/workspace/src/tools/r2e/bash_func.py"

            >>> _local_to_pod_path("./src/tools/r2e/bash_func.py")
            "/workspace/src/tools/r2e/bash_func.py"

            >>> _local_to_pod_path("/abs/path/src/tools/r2e/bash_func.py")
            "/workspace/src/tools/r2e/bash_func.py"
        """
        import os

        # Normalize path
        local_path = os.path.normpath(local_path)

        # Extract the relevant part starting from 'src/tools/'
        if 'src/tools/' in local_path:
            # Find the index of 'src/tools/' and extract from there
            idx = local_path.find('src/tools/')
            relative_path = local_path[idx:]
            return f"/workspace/{relative_path}"

        # If path starts with 'src/'
        if local_path.startswith('src/'):
            return f"/workspace/{local_path}"

        # Fallback: assume it's just a filename, place in tools directory
        filename = os.path.basename(local_path)
        self.logger.warning(
            f"Could not determine pod path for '{local_path}', "
            f"using fallback: /workspace/src/tools/{filename}"
        )
        return f"/workspace/src/tools/{filename}"

    def _build_tool_command(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Build command to execute tool in pod with stdout/stderr capture.

        All R2E tools are executable Python scripts that can be run via command line.
        We use subprocess to capture stdout and stderr separately, then output as JSON.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Command string that outputs JSON with stdout, stderr, exit_code
        """
        import json
        import base64

        # Get the tool to access its script_path
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Get local script path from tool and convert to pod path
        if tool.script_path:
            local_script_path = tool.script_path
            pod_script_path = self._local_to_pod_path(local_script_path)
        else:
            # Fallback: try to construct path from tool name
            self.logger.warning(f"Tool '{tool_name}' has no script_path, using fallback")
            pod_script_path = f"/workspace/src/tools/{tool_name}.py"

        # Build Python code that calls the tool with arguments
        # Encode arguments as JSON and pass via Python
        args_json = json.dumps(arguments)
        args_b64 = base64.b64encode(args_json.encode('utf-8')).decode('utf-8')

        # Python script that:
        # 1. Decodes arguments
        # 2. Builds command line
        # 3. Executes it
        # 4. Captures stdout/stderr
        # 5. Returns JSON
        python_script = f"""
import subprocess
import json
import base64
import sys

# Decode arguments
args_json = base64.b64decode('{args_b64}').decode('utf-8')
args = json.loads(args_json)

# Build command line arguments
cmd_parts = ['python3', '{pod_script_path}']

# Handle 'command' as positional argument if present
if 'command' in args:
    cmd_parts.append(str(args['command']))

# Add other arguments as named parameters
for key, value in args.items():
    if key == 'command':
        continue  # Already handled as positional

    if isinstance(value, bool):
        if value:
            cmd_parts.append(f'--{{key}}')
    elif isinstance(value, (list, dict)):
        cmd_parts.extend([f'--{{key}}', json.dumps(value)])
    else:
        cmd_parts.extend([f'--{{key}}', str(value)])

# Execute the command
result = subprocess.run(
    cmd_parts,
    capture_output=True,
    text=True
)

# Output result as JSON
output = {{
    'stdout': result.stdout,
    'stderr': result.stderr,
    'exit_code': result.returncode
}}
print(json.dumps(output))
"""

        # Encode the Python script as base64 to avoid any escaping issues
        script_b64 = base64.b64encode(python_script.encode('utf-8')).decode('utf-8')

        # Final command: decode and execute the Python script
        wrapper_cmd = f"echo '{script_b64}' | base64 -d | python3"

        return wrapper_cmd

    def cleanup_pod(self) -> None:
        """
        Clean up the persistent Kubernetes pod.

        This should be called when the node is no longer needed.
        """
        if self.pod is None:
            return

        try:
            self.logger.info(f"Cleaning up pod: {self.pod_name}")
            self.runner.cleanup()
            self.pod = None
            self.logger.info(f"Pod {self.pod_name} cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error cleaning up pod {self.pod_name}: {str(e)}")

    def reset(self) -> None:
        """Reset the node state (does not cleanup pod)."""
        super().reset()

    def close(self) -> None:
        """
        Close the node and cleanup the persistent pod.

        Should be called when the node is no longer needed.
        """
        self.cleanup_pod()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup_pod()
        except:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup pod."""
        self.close()
        return False