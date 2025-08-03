#!/usr/bin/env python3
"""
R2E Search tool converted to official tool implementation.
Supports both local search and remote K8s pod search.
Based on the original R2E search.py.
"""

import os
import re
import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional K8s support
try:
    from kodo import KubernetesManager
    K8S_AVAILABLE = True
except ImportError:
    KubernetesManager = None
    K8S_AVAILABLE = False

from ...core.base_tool import AgenticBaseTool
from ...core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class R2ESearchTool(AgenticBaseTool):
    """R2E-style search tool for finding text in files and directories."""
    
    def __init__(self, config: Dict = None):
        """Initialize R2E search tool."""
        # Set execution mode and K8s config FIRST, before calling super().__init__
        config = config or {}
        self.execution_mode = config.get("execution_mode", "local")
        self.pod_name = config.get("pod_name")
        self.namespace = config.get("namespace", "default")
        self.kubeconfig_path = config.get("kubeconfig_path", None)
        
        super().__init__(config)
        
        # R2E-specific settings
        self.max_files_threshold = self.config.get("max_files_threshold", 100)
        self.python_only = self.config.get("python_only", True)
        self.exclude_hidden = self.config.get("exclude_hidden", True)
        self.k8s_manager = None
        
        # Validate K8s configuration if needed
        if self.execution_mode == "k8s":
            if not K8S_AVAILABLE:
                raise ImportError("kodo library is required for K8s execution mode. Please install it from https://github.com/baidubce/kodo.git")
            if not self.pod_name:
                raise ValueError("pod_name is required when execution_mode is 'k8s'")
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for R2E search tool."""
        execution_context = f" (executing in {self.execution_mode} mode)" if self.execution_mode != "local" else ""
        
        return create_openai_tool_schema(
            name="r2e_search",
            description=f"Search for a term in either a directory or a single file{execution_context}. R2E-style tool with Python file focus.",
            parameters={
                "search_term": {
                    "type": "string",
                    "description": "The term to search for in files"
                },
                "path": {
                    "type": "string",
                    "description": "The file or directory in which to search (defaults to current directory '.')"
                },
                "python_only": {
                    "type": "boolean",
                    "description": "If true, only search in .py files when searching a directory (default: true)"
                }
            },
            required=["search_term"]
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute search operation."""
        try:
            search_term = parameters["search_term"]
            path = parameters.get("path", ".")
            python_only = parameters.get("python_only", self.python_only)
            
            # Use appropriate method based on execution mode
            if self.execution_mode == "k8s":
                return await self._search_k8s(search_term, path, python_only)
            else:
                return await self._search_local(search_term, path, python_only)
                
        except Exception as e:
            logger.error(f"R2E search execution failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _search_local(self, search_term: str, path_str: str, python_only: bool) -> ToolResult:
        """Search locally (R2E-style)."""
        try:
            path = Path(path_str).resolve()
            
            # Check if path is a file or directory
            if path.is_file():
                return await self._search_in_file_local(search_term, path)
            elif path.is_dir():
                return await self._search_in_directory_local(search_term, path, python_only)
            else:
                return ToolResult(
                    success=False, 
                    error=f"Path '{path}' not found or not accessible."
                )
                
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _search_in_file_local(self, search_term: str, filepath: Path) -> ToolResult:
        """Search in a single file using grep (R2E-style)."""
        try:
            # Use grep -n for line numbers
            cmd = ["grep", "-n", search_term, str(filepath)]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )
            except TypeError:
                # Fallback for Python 3.5/3.6
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
            
            if result.returncode == 1:
                # No matches found
                output = f'No matches found for "{search_term}" in {filepath}'
                return ToolResult(
                    success=True,
                    result={"output": output, "matches": 0}
                )
            elif result.returncode != 0:
                # Error
                return ToolResult(
                    success=False,
                    error=f"Error executing grep: {result.stderr}"
                )
            
            # Format output
            output_lines = [f'Matches for "{search_term}" in {filepath}:']
            output_lines.append(result.stdout.strip())
            output = "\n".join(output_lines)
            
            # Count matches
            num_matches = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
            
            return ToolResult(
                success=True,
                result={
                    "output": output,
                    "matches": num_matches,
                    "file": str(filepath)
                }
            )
            
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="`grep` is not available on this system. Please install or use another method."
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _search_in_directory_local(self, search_term: str, directory: Path, python_only: bool) -> ToolResult:
        """Search in directory (R2E-style)."""
        try:
            matches = {}
            num_files_matched = 0
            
            for root, dirs, files in os.walk(directory):
                # Exclude hidden directories
                if self.exclude_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                
                for file in files:
                    # Skip hidden files
                    if self.exclude_hidden and file.startswith("."):
                        continue
                    
                    # If python_only is set, only search .py files
                    if python_only and not file.endswith(".py"):
                        continue
                    
                    filepath = Path(root) / file
                    try:
                        with open(filepath, "r", errors="ignore") as f:
                            file_matches = 0
                            for line_num, line in enumerate(f, 1):
                                if search_term in line:
                                    file_matches += 1
                            if file_matches > 0:
                                matches[str(filepath)] = file_matches
                                num_files_matched += 1
                    except (UnicodeDecodeError, PermissionError):
                        # Skip files that can't be read
                        continue
            
            if not matches:
                output = f'No matches found for "{search_term}" in {directory}'
                return ToolResult(
                    success=True,
                    result={"output": output, "matches": 0}
                )
            
            # Check threshold
            if num_files_matched > self.max_files_threshold:
                output = (
                    f'More than {num_files_matched} files matched for "{search_term}" in {directory}. '
                    "Please narrow your search."
                )
                return ToolResult(
                    success=True,
                    result={"output": output, "matches": sum(matches.values())}
                )
            
            # Format output
            num_matches = sum(matches.values())
            output_lines = [f'Found {num_matches} matches for "{search_term}" in {directory}:']
            
            # Print matched files
            for filepath, count in matches.items():
                relative_path = os.path.relpath(filepath, start=os.getcwd())
                if not relative_path.startswith("./"):
                    relative_path = "./" + relative_path
                output_lines.append(f"{relative_path} ({count} matches)")
            
            output_lines.append(f'End of matches for "{search_term}" in {directory}')
            output = "\n".join(output_lines)
            
            return ToolResult(
                success=True,
                result={
                    "output": output,
                    "matches": num_matches,
                    "files_matched": num_files_matched
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _get_k8s_manager(self):
        """Get or create K8s manager instance."""
        if self.k8s_manager is None:
            self.k8s_manager = KubernetesManager(
                namespace=self.namespace,
                kubeconfig_path=self.kubeconfig_path
            )
        return self.k8s_manager
    
    async def _exec_command(self, command: str) -> Dict[str, Any]:
        """Execute command in K8s pod and return results."""
        try:
            k8s_mgr = self._get_k8s_manager()
            
            # Execute command in pod using kodo API
            output, exit_code = k8s_mgr.execute_command(self.pod_name, command)
            
            # Convert exit_code to int if it's a string
            if isinstance(exit_code, str):
                # Handle "Error: Exit code X" format
                if "Exit code" in exit_code:
                    try:
                        # Extract number from "Error: Exit code 2"
                        exit_code_int = int(exit_code.split("Exit code")[-1].strip())
                    except:
                        exit_code_int = -1
                elif exit_code.isdigit():
                    exit_code_int = int(exit_code)
                else:
                    exit_code_int = -1
            else:
                exit_code_int = exit_code
                
            return {
                "success": exit_code_int == 0,
                "stdout": output,
                "stderr": "",  # kodo API doesn't separate stderr
                "return_code": exit_code_int
            }
            
        except Exception as e:
            logger.error(f"K8s command execution failed: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }
    
    async def _search_k8s(self, search_term: str, path_str: str, python_only: bool) -> ToolResult:
        """Search in K8s pod."""
        try:
            # Check if path exists and determine type
            # Note: This command always returns exit code 0 because of the || echo structure
            path_check_cmd = f"if [ -f '{path_str}' ]; then echo 'file'; elif [ -d '{path_str}' ]; then echo 'dir'; else echo 'none'; fi"
            path_check = await self._exec_command(path_check_cmd)
            
            # Check if command execution itself failed (not just the test)
            # Note: Return codes 0, 1, 2 are all valid for shell test commands
            # 0 = true/success, 1 = false/failure, 2 = usage error (but still valid response)
            # We should only fail if we get other error codes or no stdout
            if path_check["return_code"] not in [0, 1, 2] or not path_check.get("stdout"):
                error_msg = f"Failed to check path '{path_str}'. Command failed with return code: {path_check.get('return_code', 'N/A')}, stderr: {path_check.get('stderr', '')}"
                logger.error(error_msg)
                return ToolResult(success=False, error=error_msg)
            
            path_type = path_check["stdout"].strip()
            
            if path_type == "none":
                return ToolResult(
                    success=False,
                    error=f"Path '{path_str}' not found or not accessible."
                )
            elif path_type == "file":
                return await self._search_in_file_k8s(search_term, path_str)
            else:  # directory
                return await self._search_in_directory_k8s(search_term, path_str, python_only)
                
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _search_in_file_k8s(self, search_term: str, filepath: str) -> ToolResult:
        """Search in a single file in K8s pod."""
        # Escape single quotes in search term
        escaped_term = search_term.replace("'", "'\"'\"'")
        
        cmd = f"grep -n '{escaped_term}' '{filepath}'"
        result = await self._exec_command(cmd)
        
        if result["return_code"] == 1:
            # No matches found
            output = f'No matches found for "{search_term}" in {filepath}'
            return ToolResult(
                success=True,
                result={"output": output, "matches": 0}
            )
        elif not result["success"]:
            # Error
            return ToolResult(
                success=False,
                error=f"Error executing grep: {result['stderr']}"
            )
        
        # Format output
        output_lines = [f'Matches for "{search_term}" in {filepath}:']
        output_lines.append(result["stdout"].strip())
        output = "\n".join(output_lines)
        
        # Count matches
        num_matches = len(result["stdout"].strip().split("\n")) if result["stdout"].strip() else 0
        
        return ToolResult(
            success=True,
            result={
                "output": output,
                "matches": num_matches,
                "file": filepath
            }
        )
    
    async def _search_in_directory_k8s(self, search_term: str, directory: str, python_only: bool) -> ToolResult:
        """Search in directory in K8s pod."""
        # Build find command
        find_cmd = f"find '{directory}' -type f"
        
        if self.exclude_hidden:
            find_cmd += " -not -path '*/.*'"
        
        if python_only:
            find_cmd += " -name '*.py'"
        
        # Get list of files
        files_result = await self._exec_command(find_cmd)
        if not files_result["success"]:
            return ToolResult(success=False, error=f"Failed to list files: {files_result['stderr']}")
        
        files = [f.strip() for f in files_result["stdout"].strip().split("\n") if f.strip()]
        
        # Search in each file
        matches = {}
        num_files_matched = 0
        escaped_term = search_term.replace("'", "'\"'\"'")
        
        for filepath in files:
            # Count matches in file
            count_cmd = f"grep -c '{escaped_term}' '{filepath}' 2>/dev/null || echo 0"
            count_result = await self._exec_command(count_cmd)
            
            if count_result["success"]:
                try:
                    count = int(count_result["stdout"].strip())
                    if count > 0:
                        matches[filepath] = count
                        num_files_matched += 1
                except ValueError:
                    continue
        
        if not matches:
            output = f'No matches found for "{search_term}" in {directory}'
            return ToolResult(
                success=True,
                result={"output": output, "matches": 0}
            )
        
        # Check threshold
        if num_files_matched > self.max_files_threshold:
            output = (
                f'More than {num_files_matched} files matched for "{search_term}" in {directory}. '
                "Please narrow your search."
            )
            return ToolResult(
                success=True,
                result={"output": output, "matches": sum(matches.values())}
            )
        
        # Format output
        num_matches = sum(matches.values())
        output_lines = [f'Found {num_matches} matches for "{search_term}" in {directory}:']
        
        # Get current working directory for relative paths
        pwd_result = await self._exec_command("pwd")
        cwd = pwd_result["stdout"].strip() if pwd_result["success"] else directory
        
        # Print matched files
        for filepath, count in matches.items():
            # Try to make relative path
            if filepath.startswith(cwd):
                relative_path = filepath[len(cwd):].lstrip('/')
                relative_path = f"./{relative_path}" if relative_path else "."
            else:
                relative_path = filepath
            
            output_lines.append(f"{relative_path} ({count} matches)")
        
        output_lines.append(f'End of matches for "{search_term}" in {directory}')
        output = "\n".join(output_lines)
        
        return ToolResult(
            success=True,
            result={
                "output": output,
                "matches": num_matches,
                "files_matched": num_files_matched
            }
        )
    
    def get_execution_info(self) -> Dict[str, Any]:
        """Get information about the execution environment."""
        info = {
            "execution_mode": self.execution_mode,
            "max_files_threshold": self.max_files_threshold,
            "python_only": self.python_only,
            "exclude_hidden": self.exclude_hidden,
            "tool_style": "R2E"
        }
        
        if self.execution_mode == "k8s":
            info.update({
                "pod_name": self.pod_name,
                "namespace": self.namespace,
                "kubeconfig_path": self.kubeconfig_path or "default"
            })
        
        return info