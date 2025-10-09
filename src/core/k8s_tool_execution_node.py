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
    from kodo import ContainerRunner, AsyncKubernetesManager
    KODO_AVAILABLE = True
except ImportError:
    KODO_AVAILABLE = False
    ContainerRunner = None
    AsyncKubernetesManager = None


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
                 cleanup_existing_pod: bool = False,
                 cpu_request: str = "0.3",
                 memory_request: str = "1Gi",
                 timeline_enabled: bool = False,
                 tool_timeout: float = None):
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
            cleanup_existing_pod: Whether to delete existing pod before creating (default: False)
            cpu_request: CPU resource request (default: "0.3" for 0.3 core)
            memory_request: Memory resource request (default: "1Gi" for 1GB)
            timeline_enabled: Enable automatic timeline tracking for this node
            tool_timeout: Timeout in seconds for tool execution (None = no timeout)
        """
        if not KODO_AVAILABLE:
            raise ImportError(
                "Kodo library is required for K8S tool execution. "
                "Install it with: pip install kodo"
            )

        super().__init__(name, timeline_enabled=timeline_enabled, timeout=tool_timeout)

        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path
        self.image = image
        self.node_selector = node_selector or {}
        self.environment = environment or {}
        self.image_pull_policy = image_pull_policy
        self.cleanup_existing_pod = cleanup_existing_pod
        self.cpu_request = cpu_request
        self.memory_request = memory_request

        # Initialize Kodo async manager (will be initialized lazily)
        self.async_manager = None
        
        # Keep sync runner for backward compatibility
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

    async def _ensure_async_manager(self) -> None:
        """Initialize async manager if not already initialized."""
        if self.async_manager is None:
            if self.timeline_enabled and self._timeline:
                event_id = self._timeline.start_event(self.name, "async_manager_create", {})
            try:
                self.async_manager = AsyncKubernetesManager(
                    namespace=self.namespace,
                    kubeconfig_path=self.kubeconfig_path,
                    disable_proxy=True
                )
                await self.async_manager._init_client()
            finally:
                if self.timeline_enabled and self._timeline:
                    self._timeline.end_event(event_id)
    
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
            # Ensure async manager is ready
            await self._ensure_async_manager()
            
            # Create pod asynchronously
            await self._create_pod_async()

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

            # Execute command using kubectl directly (faster and more reliable)
            # Note: Using kubectl instead of kodo async API due to performance issues
            try:
                if self.timeout:
                    stdout, exit_code_str = await asyncio.wait_for(
                        self._execute_kubectl_async(command),
                        timeout=self.timeout
                    )
                else:
                    stdout, exit_code_str = await self._execute_kubectl_async(command)
            except asyncio.TimeoutError:
                self.logger.error(f"Tool '{tool_name}' execution timeout after {self.timeout}s")
                return {
                    "tool": tool_name,
                    "result": "",
                    "status": "error",
                    "error": f"Tool execution timeout after {self.timeout}s",
                    "stdout": "",
                    "stderr": f"ERROR: Tool execution timeout after {self.timeout}s",
                    "exit_code": 124  # Standard timeout exit code
                }
            
            # Parse exit code
            try:
                exit_code = int(exit_code_str)
            except (ValueError, TypeError):
                # If exit_code_str is "Error: Exit code X", extract X
                if isinstance(exit_code_str, str) and "Exit code" in exit_code_str:
                    try:
                        exit_code = int(exit_code_str.split("Exit code")[-1].strip())
                    except:
                        exit_code = 1
                else:
                    exit_code = 1 if exit_code_str != "0" else 0
            
            # For tools, stdout is the actual output, stderr is empty
            # (kodo combines stdout/stderr into the first return value)
            stderr = ""

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

            # Execute command using kodo directly
            # kodo's execute_command returns (stdout, exit_code_str)
            stdout, exit_code_str = self.runner.execute_command(self.pod, command)
            
            # Parse exit code
            try:
                exit_code = int(exit_code_str)
            except (ValueError, TypeError):
                # If exit_code_str is "Error: Exit code X", extract X
                if isinstance(exit_code_str, str) and "Exit code" in exit_code_str:
                    try:
                        exit_code = int(exit_code_str.split("Exit code")[-1].strip())
                    except:
                        exit_code = 1
                else:
                    exit_code = 1 if exit_code_str != "0" else 0
            
            # For tools, stdout is the actual output, stderr is empty
            # (kodo combines stdout/stderr into the first return value)
            stderr = ""

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
            "--kubeconfig", self.kubeconfig_path,
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

            # Combine stdout and stderr (like kodo does)
            combined_output = stdout_str
            if stderr_str:
                self.logger.debug(f"kubectl stderr: {stderr_str[:200]}")
                combined_output += stderr_str

            # Return in Kodo-compatible format (output, status)
            return (combined_output, str(process.returncode))

        except Exception as e:
            self.logger.error(f"kubectl exec failed: {str(e)}")
            raise

    async def _create_pod_async(self) -> None:
        """
        Create the persistent pod for tool execution asynchronously.

        This pod will be used throughout the node's lifecycle.
        First deletes any existing pod with the same name to ensure idempotency.
        Includes retry logic for network errors.
        """
        self.logger.info(f"Creating persistent K8S pod: {self.pod_name}")

        max_retries = 3
        retry_delay = 5
        
        for attempt in range(1, max_retries + 1):
            try:
                # Delete existing pod with same name if it exists (only if cleanup_existing_pod=True)
                if self.cleanup_existing_pod:
                    if self.timeline_enabled and self._timeline:
                        event_id = self._timeline.start_event(self.name, "pod_cleanup", {})
                    try:
                        await self._cleanup_existing_pod_async()
                    finally:
                        if self.timeline_enabled and self._timeline:
                            self._timeline.end_event(event_id)

                # Add PYTHONPATH to environment
                env = self.environment.copy()
                env['PYTHONPATH'] = '/workspace:$PYTHONPATH'

                # Initialize async manager with timing
                if self.timeline_enabled and self._timeline:
                    event_id = self._timeline.start_event(self.name, "async_manager_init", {})
                try:
                    # Async manager should already be initialized by _ensure_async_manager
                    # This is just a placeholder for timing
                    pass
                finally:
                    if self.timeline_enabled and self._timeline:
                        self._timeline.end_event(event_id)

                # Create pod using async kodo manager
                self.logger.info(f"Creating pod (attempt {attempt}/{max_retries})...")
                
                # Prepare resource requests
                resources = {
                    "requests": {
                        "cpu": self.cpu_request,
                        "memory": self.memory_request
                    }
                }
                
                # Pod creation with timing
                if self.timeline_enabled and self._timeline:
                    event_id = self._timeline.start_event(self.name, "pod_creation", {"attempt": attempt})
                try:
                    self.pod = await self.async_manager.start_pod(
                        name=self.pod_name,
                        image=self.image,
                        command="sleep infinity",
                        environment=env,
                        resources=resources,
                        node_selector=self.node_selector
                    )
                    self.logger.info(f"Pod {self.pod_name} created successfully")
                finally:
                    if self.timeline_enabled and self._timeline:
                        self._timeline.end_event(event_id)

                # Run post-creation initialization commands
                await self._run_pod_init_commands_async()

                # Copy tools to the pod (async)
                await self._copy_tools_to_pod_async()
                
                # Success - break out of retry loop
                return

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Check if this is a retryable error
                is_retryable = (
                    'ProtocolError' in error_type or
                    'Response ended prematurely' in error_msg or
                    'ConnectionError' in error_type or
                    'Timeout' in error_type or
                    'ReadTimeoutError' in error_type
                )
                
                if is_retryable and attempt < max_retries:
                    self.logger.warning(
                        f"Retryable error creating pod {self.pod_name} "
                        f"(attempt {attempt}/{max_retries}): {error_type}: {error_msg}"
                    )
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)  # Exponential backoff, max 30s
                    continue
                else:
                    # Non-retryable error or max retries exceeded
                    self.logger.error(
                        f"Failed to create pod {self.pod_name} after {attempt} attempts: "
                        f"{error_type}: {error_msg}"
                    )
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

            # Run post-creation initialization commands
            self._run_pod_init_commands()

            # Copy tools to the pod
            self._copy_tools_to_pod()

        except Exception as e:
            self.logger.error(f"Failed to create pod {self.pod_name}: {str(e)}")
            raise

    def _run_pod_init_commands(self) -> None:
        """
        Run initialization commands in the pod after creation (sync version).

        These commands set up the environment for tool execution:
        - Install required Python packages
        - Create necessary symlinks
        - Set up permissions
        """
        init_commands = [
            # Install common dependencies
            "python3 -m pip install --quiet --no-warn-script-location chardet 2>/dev/null || true",

            # Create workspace directory if not exists
            "mkdir -p /workspace",

            # Make conda env symlink if conda exists
            "if [ -d /opt/miniconda3/envs/testbed ]; then ln -sf /opt/miniconda3/envs/testbed /root/.venv; fi",
        ]

        self.logger.info("Running pod initialization commands...")

        for cmd in init_commands:
            try:
                self.runner.execute_command(self.pod, cmd)
                self.logger.debug(f"Init command succeeded: {cmd[:50]}...")
            except Exception as e:
                # Don't fail pod creation if init commands fail - they're optional
                self.logger.warning(f"Init command failed (non-fatal): {cmd[:50]}... - {str(e)}")

        self.logger.info("Pod initialization completed")

    async def _run_pod_init_commands_async(self) -> None:
        """
        Run initialization commands in the pod after creation (async version).

        These commands set up the environment for tool execution:
        - Install required Python packages
        - Create necessary symlinks
        - Set up permissions
        """
        import os
        
        # Check if local pip packages exist
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pip_packages_dir = os.path.join(os.path.dirname(current_dir), "pip_packages")
        
        init_commands = []
        
        
        init_commands.extend([
            # Create workspace directory if not exists
            "mkdir -p /workspace",
            "chmod +x /run_tests.sh",
            # Make conda env symlink if conda exists
            "ln -s /opt/miniconda3/envs/testbed /root/.venv"
        ])
        # If local pip packages exist, copy and install from local
        if os.path.exists(pip_packages_dir) and os.listdir(pip_packages_dir):
            init_commands.extend([
                # Create pip packages directory in pod
                "mkdir -p /workspace/pip_packages",
            ])
            # Note: wheel files will be copied before this
            init_commands.append(
                # Install from local wheels (no network needed)
                "python3 -m pip install --quiet --no-warn-script-location --no-index --find-links=/workspace/pip_packages chardet 2>/dev/null || true"
            )
        else:
            # Fallback: install from network
            init_commands.append(
                "python3 -m pip install --quiet --no-warn-script-location chardet 2>/dev/null || true"
            )


        self.logger.info("Running pod initialization commands...")

        for idx, cmd in enumerate(init_commands):
            if self.timeline_enabled and self._timeline:
                event_id = self._timeline.start_event(self.name, f"init_cmd_{idx}", {"command": cmd[:50]})
            try:
                output, status = await self.async_manager.execute_command(self.pod_name, cmd)
                self.logger.debug(f"Init command succeeded: {cmd[:50]}...")
            except Exception as e:
                # Don't fail pod creation if init commands fail - they're optional
                self.logger.warning(f"Init command failed (non-fatal): {cmd[:50]}... - {str(e)}")
            finally:
                if self.timeline_enabled and self._timeline:
                    self._timeline.end_event(event_id)

        self.logger.info("Pod initialization completed")

    async def _cleanup_existing_pod_async(self) -> None:
        """
        Clean up any existing pod with the same name asynchronously.

        This ensures idempotency - if a previous run left a pod with the same name,
        we delete it before creating a new one.
        """
        try:
            # Check if pod exists
            check_cmd = [
                "kubectl", "get", "pod", self.pod_name,
                "-n", self.namespace,
                "--kubeconfig", self.kubeconfig_path,
                "--ignore-not-found"
            ]

            process = await asyncio.create_subprocess_exec(
                *check_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10
            )

            # If pod exists (non-empty output), delete it
            if stdout.strip():
                self.logger.info(f"Found existing pod {self.pod_name}, deleting...")

                delete_cmd = [
                    "kubectl", "delete", "pod", self.pod_name,
                    "-n", self.namespace,
                    "--kubeconfig", self.kubeconfig_path,
                    "--force", "--grace-period=0"
                ]

                delete_process = await asyncio.create_subprocess_exec(
                    *delete_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                await asyncio.wait_for(
                    delete_process.communicate(),
                    timeout=30
                )

                self.logger.info(f"Deleted existing pod {self.pod_name}")

                # Wait asynchronously for pod to be fully deleted
                await asyncio.sleep(2)
            else:
                self.logger.debug(f"No existing pod {self.pod_name} found")

        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout while checking/deleting existing pod {self.pod_name}")
        except Exception as e:
            self.logger.warning(f"Error checking/deleting existing pod {self.pod_name}: {str(e)}")
            # Don't raise - we'll try to create the pod anyway

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
                "--kubeconfig", self.kubeconfig_path,
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
                    "--kubeconfig", self.kubeconfig_path,
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

    async def _copy_tools_to_pod_async(self) -> None:
        """
        Copy the tools directory to the pod by copying individual files asynchronously.
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

            # Create directory structure using async manager with timing
            if self.timeline_enabled and self._timeline:
                event_id = self._timeline.start_event(self.name, "tools_mkdir", {})
            try:
                await self.async_manager.execute_command(self.pod_name, "mkdir -p /workspace/src/tools/r2e")
                await self.async_manager.execute_command(self.pod_name, "mkdir -p /workspace/src/tools/mini_swe")
                await self.async_manager.execute_command(self.pod_name, "mkdir -p /workspace/pip_packages")
            finally:
                if self.timeline_enabled and self._timeline:
                    self._timeline.end_event(event_id)

            # Collect all files to copy
            copy_tasks = []

            # Copy all Python files from tools/r2e
            r2e_dir = os.path.join(tools_dir, "r2e")
            if os.path.exists(r2e_dir):
                for py_file in glob.glob(os.path.join(r2e_dir, "*.py")):
                    filename = os.path.basename(py_file)
                    pod_path = f"/workspace/src/tools/r2e/{filename}"
                    copy_tasks.append(self._copy_file_to_pod_async(py_file, pod_path))

            # Copy all Python files from tools/mini_swe
            mini_swe_dir = os.path.join(tools_dir, "mini_swe")
            if os.path.exists(mini_swe_dir):
                for py_file in glob.glob(os.path.join(mini_swe_dir, "*.py")):
                    filename = os.path.basename(py_file)
                    pod_path = f"/workspace/src/tools/mini_swe/{filename}"
                    copy_tasks.append(self._copy_file_to_pod_async(py_file, pod_path))

            # Copy base_tool.py if exists
            base_tool_path = os.path.join(tools_dir, "base_tool.py")
            if os.path.exists(base_tool_path):
                copy_tasks.append(self._copy_file_to_pod_async(base_tool_path, "/workspace/src/tools/base_tool.py"))

            # Copy __init__.py files
            for init_file in ["__init__.py", "r2e/__init__.py", "mini_swe/__init__.py"]:
                local_init = os.path.join(tools_dir, init_file)
                if os.path.exists(local_init):
                    pod_init = f"/workspace/src/tools/{init_file}"
                    copy_tasks.append(self._copy_file_to_pod_async(local_init, pod_init))
            
            # Copy pip packages if they exist (for offline installation)
            pip_packages_dir = os.path.join(os.path.dirname(current_dir), "pip_packages")
            if os.path.exists(pip_packages_dir):
                for wheel_file in glob.glob(os.path.join(pip_packages_dir, "*.whl")):
                    filename = os.path.basename(wheel_file)
                    pod_path = f"/workspace/pip_packages/{filename}"
                    copy_tasks.append(self._copy_file_to_pod_async(wheel_file, pod_path))
                self.logger.info(f"Found {len(glob.glob(os.path.join(pip_packages_dir, '*.whl')))} wheel files to copy")

            # Execute all copy operations concurrently with timing
            if self.timeline_enabled and self._timeline:
                event_id = self._timeline.start_event(self.name, "tools_copy_files", {"file_count": len(copy_tasks)})
            try:
                await asyncio.gather(*copy_tasks, return_exceptions=True)
            finally:
                if self.timeline_enabled and self._timeline:
                    self._timeline.end_event(event_id)

            self.logger.info("Tools copied to pod at /workspace/src/tools")

        except Exception as e:
            self.logger.warning(f"Failed to copy tools to pod: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())

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

    async def _copy_file_to_pod_async(self, local_path: str, pod_path: str) -> None:
        """
        Copy a single file to the pod using base64 encoding asynchronously.
        Supports both text and binary files.

        Args:
            local_path: Path to local file
            pod_path: Destination path in pod
        """
        try:
            # Read file content in binary mode (works for both text and binary files)
            import base64
            with open(local_path, 'rb') as f:
                content = f.read()

            # Use base64 encoding to avoid shell escaping issues
            content_b64 = base64.b64encode(content).decode('utf-8')

            # Create file in pod using base64 decoding with async manager
            cmd = f"echo '{content_b64}' | base64 -d > {pod_path}"
            await self.async_manager.execute_command(self.pod_name, cmd)

        except Exception as e:
            self.logger.warning(f"Failed to copy {local_path} to pod: {str(e)}")

    def _copy_file_to_pod(self, local_path: str, pod_path: str) -> None:
        """
        Copy a single file to the pod using base64 encoding.
        Supports both text and binary files.

        Args:
            local_path: Path to local file
            pod_path: Destination path in pod
        """
        try:
            # Read file content in binary mode (works for both text and binary files)
            import base64
            with open(local_path, 'rb') as f:
                content = f.read()

            # Use base64 encoding to avoid shell escaping issues
            content_b64 = base64.b64encode(content).decode('utf-8')

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
        Build command to execute tool in pod directly.

        All R2E tools are executable Python scripts that can be run via command line.
        We execute them directly and let kubectl capture stdout/stderr.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Command string to execute the tool
        """
        import json

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

        # Build command line arguments
        # Use conda environment's Python if available, otherwise fall back to python3
        cmd_parts = ['/opt/miniconda3/envs/testbed/bin/python', pod_script_path]

        # Handle 'command' as positional argument if present
        if 'command' in arguments:
            cmd_parts.append(str(arguments['command']))

        # Add other arguments as named parameters
        for key, value in arguments.items():
            if key == 'command':
                continue  # Already handled as positional

            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f'--{key}')
            elif isinstance(value, (list, dict)):
                cmd_parts.extend([f'--{key}', json.dumps(value)])
            else:
                cmd_parts.extend([f'--{key}', str(value)])

        # Join command parts with proper shell escaping
        import shlex
        return ' '.join(shlex.quote(part) for part in cmd_parts)

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
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup pod asynchronously."""
        await self.delete_pod_async()
        return False
    
    async def delete_pod_async(self) -> None:
        """
        Delete the pod asynchronously using async kodo manager.
        
        This ensures the pod is truly deleted, not just disconnected.
        """
        if self.pod is None and not self._pod_creation_started:
            self.logger.debug(f"No pod to delete: {self.pod_name}")
            return
        
        try:
            self.logger.info(f"Deleting pod: {self.pod_name}")
            
            # Ensure async manager is initialized
            await self._ensure_async_manager()
            
            # Delete pod using async kodo manager
            await self.async_manager.delete_pod(self.pod_name, grace_period=0)
            
            self.logger.info(f"Pod {self.pod_name} deleted successfully")
            
            self.pod = None
            self._pod_ready.clear()
            self._pod_creation_started = False
            
            # Close async manager
            if self.async_manager and self.async_manager.client:
                await self.async_manager.client.api_client.close()
                self.async_manager = None
            
        except Exception as e:
            self.logger.error(f"Error deleting pod {self.pod_name}: {str(e)}")
