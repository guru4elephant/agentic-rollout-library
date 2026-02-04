#!/usr/bin/env python3
"""
R2E Bash Executor tool - Function call description and entry point.
This file handles the tool schema and delegates execution to the executable.
"""

import os
import sys
import json
import subprocess
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

TOOL_NAME = "r2e_bash_executor"
EXECUTABLE_NAME = "r2e_bash_executor_exe.py"

# Commands that are blocked for security reasons (from R2E)
R2E_BLOCKED_COMMANDS = ["git", "ipython", "jupyter", "nohup"]

def get_tool_schema():
    """Return the OpenAI function tool schema for bash executor."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Execute a bash command in the terminal. R2E-style tool with security restrictions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "The bash command to execute. For example: 'python my_script.py'"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds (default: 30)",
                        "default": 30
                    }
                },
                "required": ["cmd"]
            }
        }
    }

def execute_tool(parameters: Dict[str, Any], execution_mode: str = "local", **kwargs) -> Dict[str, Any]:
    """
    Execute the bash command tool.
    
    Args:
        parameters: Tool parameters from function call
        execution_mode: Either "local" or "remote"
        **kwargs: Additional parameters for remote execution (pod_name, namespace, etc.)
    
    Returns:
        Dict with execution results
    """
    try:
        # Check for blocked commands
        cmd = parameters.get("cmd", "")
        first_word = cmd.split()[0] if cmd.split() else ""
        
        if first_word in R2E_BLOCKED_COMMANDS:
            return {
                "success": False,
                "error": f"Command '{first_word}' is blocked for security reasons",
                "blocked_command": first_word
            }
        
        if execution_mode == "remote":
            # Remote execution in K8s pod
            return execute_remote(parameters, **kwargs)
        else:
            # Local execution
            return execute_local(parameters)
    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def execute_local(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the tool locally by calling the executable."""
    try:
        # Find the executable
        current_dir = Path(__file__).parent
        exe_path = current_dir / EXECUTABLE_NAME
        
        if not exe_path.exists():
            return {
                "success": False,
                "error": f"Executable not found: {exe_path}"
            }
        
        # Prepare command
        cmd = ["python3", str(exe_path)]
        
        # Add the bash command as argument
        cmd.append(parameters["cmd"])
        
        # Add timeout if provided
        timeout = parameters.get("timeout", 30)
        cmd.append(str(timeout))
        
        # Execute
        logger.debug(f"Executing local command: python3 {EXECUTABLE_NAME} [command]")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5  # Add 5 seconds buffer for the executable to handle timeout
        )
        
        # Parse output
        if result.returncode == 0:
            # Try to parse JSON output
            try:
                output_data = json.loads(result.stdout)
                return output_data
            except json.JSONDecodeError:
                # Fallback to plain text output
                return {
                    "success": True,
                    "output": result.stdout,
                    "stderr": result.stderr
                }
        else:
            # Check for timeout
            if result.returncode == 124:  # Standard timeout exit code
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "timeout": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr or f"Command failed with exit code {result.returncode}",
                    "stdout": result.stdout,
                    "exit_code": result.returncode
                }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Execution timed out after {parameters.get('timeout', 30)} seconds",
            "timeout": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def execute_remote(parameters: Dict[str, Any], pod_name: str, namespace: str = "default", 
                  kubeconfig_path: Optional[str] = None, working_dir: str = "/testbed") -> Dict[str, Any]:
    """
    Execute the tool remotely in a K8s pod.
    
    Args:
        parameters: Tool parameters
        pod_name: Name of the K8s pod
        namespace: K8s namespace
        kubeconfig_path: Path to kubeconfig file
        working_dir: Working directory in the pod
    
    Returns:
        Dict with execution results
    """
    try:
        # Import K8s manager
        try:
            from kodo import KubernetesManager
        except ImportError:
            return {
                "success": False,
                "error": "kodo library is required for remote execution. Install from https://github.com/baidubce/kodo.git"
            }
        
        # Create K8s manager
        k8s_manager = KubernetesManager(
            namespace=namespace,
            kubeconfig_path=kubeconfig_path
        )
        
        # Build remote command
        # First cd to working directory, then execute the command via the executable
        timeout = parameters.get("timeout", 30)
        
        # Escape the command for shell
        cmd_escaped = parameters["cmd"].replace("'", "'\"'\"'")
        remote_cmd = f"cd {working_dir} && python3 /path/to/{EXECUTABLE_NAME} '{cmd_escaped}' {timeout}"
        
        # Execute in pod
        logger.debug(f"Executing in pod {pod_name}: {remote_cmd[:100]}...")
        output, exit_code = k8s_manager.execute_command(pod_name, remote_cmd)
        
        # Parse output
        if exit_code == 0:
            # Try to parse JSON output
            try:
                output_data = json.loads(output)
                return output_data
            except json.JSONDecodeError:
                # Fallback to plain text
                return {
                    "success": True,
                    "output": output,
                    "exit_code": exit_code
                }
        else:
            # Check for timeout
            if exit_code == 124:
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "timeout": True,
                    "output": output
                }
            else:
                return {
                    "success": False,
                    "error": f"Command failed with exit code {exit_code}",
                    "output": output,
                    "exit_code": exit_code
                }
            
    except Exception as e:
        logger.error(f"Remote execution failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

if __name__ == "__main__":
    # Test mode - can be used to test the tool directly
    import argparse
    parser = argparse.ArgumentParser(description="R2E Bash Executor Tool")
    parser.add_argument("cmd", help="Bash command to execute")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    parser.add_argument("--execution_mode", default="local", help="Execution mode: local or remote")
    parser.add_argument("--pod_name", help="K8s pod name for remote execution")
    parser.add_argument("--namespace", default="default", help="K8s namespace")
    parser.add_argument("--working_dir", default="/testbed", help="Working directory for remote execution")
    
    args = parser.parse_args()
    
    # Build parameters
    params = {
        "cmd": args.cmd,
        "timeout": args.timeout
    }
    
    # Execute
    if args.execution_mode == "remote":
        if not args.pod_name:
            print(json.dumps({"success": False, "error": "pod_name required for remote execution"}))
            sys.exit(1)
        
        result = execute_tool(
            params, 
            execution_mode="remote",
            pod_name=args.pod_name,
            namespace=args.namespace,
            working_dir=args.working_dir
        )
    else:
        result = execute_tool(params, execution_mode="local")
    
    print(json.dumps(result, indent=2))