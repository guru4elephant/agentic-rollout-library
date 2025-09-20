#!/usr/bin/env python3
"""
R2E String Replace Editor tool - Function call description and entry point.
A simplified version of the file editor focused on specific commands.
This file handles the tool schema and delegates execution to the file editor executable.
"""

import os
import sys
import json
import subprocess
import logging
from typing import Any, Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

TOOL_NAME = "r2e_str_replace_editor"
EXECUTABLE_NAME = "r2e_file_editor_exe.py"  # Reuses the file editor executable

# Control what commands are visible to agents (from R2E)
ALLOWED_STR_REPLACE_EDITOR_COMMANDS = ["view", "create", "str_replace", "insert"]

def get_tool_schema():
    """Return the OpenAI function tool schema for string replace editor."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "R2E String Replace Editor - a simplified file editing tool focused on string replacement operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run",
                        "enum": ALLOWED_STR_REPLACE_EDITOR_COMMANDS
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory"
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Required for 'create' command - content of the file to be created"
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Required for 'str_replace' - the exact string to replace (must be unique)"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "For 'str_replace' - replacement string; For 'insert' - string to insert"
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Required for 'insert' - line number after which to insert new_str"
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional for 'view' - line range [start, end], use -1 for end of file"
                    }
                },
                "required": ["command", "path"]
            }
        }
    }

def execute_tool(parameters: Dict[str, Any], execution_mode: str = "local", **kwargs) -> Dict[str, Any]:
    """
    Execute the string replace editor tool.
    
    Args:
        parameters: Tool parameters from function call
        execution_mode: Either "local" or "remote"
        **kwargs: Additional parameters for remote execution (pod_name, namespace, etc.)
    
    Returns:
        Dict with execution results
    """
    try:
        # Validate command is allowed
        command = parameters.get("command")
        if command not in ALLOWED_STR_REPLACE_EDITOR_COMMANDS:
            return {
                "success": False,
                "error": f"Command '{command}' is not allowed. Allowed commands: {', '.join(ALLOWED_STR_REPLACE_EDITOR_COMMANDS)}"
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
    """Execute the tool locally by calling the file editor executable."""
    try:
        # Find the executable (reuse file editor executable)
        current_dir = Path(__file__).parent
        exe_path = current_dir / EXECUTABLE_NAME
        
        if not exe_path.exists():
            return {
                "success": False,
                "error": f"Executable not found: {exe_path}"
            }
        
        # Prepare command (same as file editor)
        cmd = ["python3", str(exe_path)]
        
        # Add command and path as positional arguments
        cmd.append(parameters["command"])
        cmd.append(parameters["path"])
        
        # Add optional parameters as JSON-encoded argument
        optional_params = {}
        for key in ["file_text", "old_str", "new_str", "insert_line", "view_range"]:
            if key in parameters:
                optional_params[key] = parameters[key]
        
        if optional_params:
            cmd.append(json.dumps(optional_params))
        
        # Execute
        logger.debug(f"Executing local command: {' '.join(cmd[:3])}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout for file operations
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
            return {
                "success": False,
                "error": result.stderr or f"Command failed with exit code {result.returncode}",
                "stdout": result.stdout,
                "exit_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out after 60 seconds"
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
        
        # Build remote command (reuse file editor executable)
        remote_cmd = f"cd {working_dir} && python3 /path/to/{EXECUTABLE_NAME}"
        
        # Add command and path
        remote_cmd += f" {parameters['command']} '{parameters['path']}'"
        
        # Add optional parameters as JSON
        optional_params = {}
        for key in ["file_text", "old_str", "new_str", "insert_line", "view_range"]:
            if key in parameters:
                optional_params[key] = parameters[key]
        
        if optional_params:
            # Escape the JSON for shell
            json_str = json.dumps(optional_params).replace("'", "'\"'\"'")
            remote_cmd += f" '{json_str}'"
        
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
    parser = argparse.ArgumentParser(description="R2E String Replace Editor Tool")
    parser.add_argument("--command", required=True, 
                       choices=ALLOWED_STR_REPLACE_EDITOR_COMMANDS,
                       help="Command to execute")
    parser.add_argument("--path", required=True, help="File or directory path")
    parser.add_argument("--file_text", help="File content for create")
    parser.add_argument("--old_str", help="String to replace")
    parser.add_argument("--new_str", help="Replacement string")
    parser.add_argument("--insert_line", type=int, help="Line number for insert")
    parser.add_argument("--view_range", nargs=2, type=int, help="View range [start end]")
    parser.add_argument("--execution_mode", default="local", help="Execution mode: local or remote")
    
    args = parser.parse_args()
    
    # Build parameters
    params = {
        "command": args.command,
        "path": args.path
    }
    
    if args.file_text:
        params["file_text"] = args.file_text
    if args.old_str:
        params["old_str"] = args.old_str
    if args.new_str:
        params["new_str"] = args.new_str
    if args.insert_line:
        params["insert_line"] = args.insert_line
    if args.view_range:
        params["view_range"] = args.view_range
    
    # Execute
    result = execute_tool(params, execution_mode=args.execution_mode)
    print(json.dumps(result, indent=2))