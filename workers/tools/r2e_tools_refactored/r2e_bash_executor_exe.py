#!/usr/bin/env python3
"""
R2E Bash Executor executable - Actual implementation that runs as a standalone program.
This file executes bash commands and outputs results to stdout/stderr.
"""

import os
import sys
import json
import subprocess
import signal
from typing import Dict, Any

def main():
    """Main entry point for the executable."""
    try:
        # Parse command line arguments
        if len(sys.argv) < 2:
            error_output("Usage: r2e_bash_executor_exe.py <command> [timeout]")
            sys.exit(1)
        
        cmd = sys.argv[1]
        timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        
        # Execute command
        result = execute_bash_command(cmd, timeout)
        
        # Output result as JSON
        output_json(result)
        sys.exit(0 if result.get("success", False) else 1)
        
    except Exception as e:
        error_output(f"Unexpected error: {e}")
        sys.exit(1)

def output_json(data: Dict[str, Any]):
    """Output JSON data to stdout."""
    print(json.dumps(data))

def error_output(message: str):
    """Output error message to stderr."""
    sys.stderr.write(f"ERROR: {message}\n")

def execute_bash_command(cmd: str, timeout: int) -> Dict[str, Any]:
    """
    Execute a bash command with timeout.
    
    Args:
        cmd: The bash command to execute
        timeout: Timeout in seconds
    
    Returns:
        Dict with execution results
    """
    try:
        # Use shell=True to execute the command in bash
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Create new process group for timeout handling
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Format the output
            output_text = ""
            
            # Add stdout if present
            if stdout:
                output_text = stdout
            
            # Add stderr if present (R2E style: append stderr after stdout)
            if stderr:
                if output_text:
                    output_text += "\n" + stderr
                else:
                    output_text = stderr
            
            # Check exit code
            if process.returncode == 0:
                return {
                    "success": True,
                    "output": output_text,
                    "exit_code": 0
                }
            else:
                return {
                    "success": False,
                    "output": output_text,
                    "exit_code": process.returncode,
                    "error": f"Command failed with exit code {process.returncode}"
                }
                
        except subprocess.TimeoutExpired:
            # Kill the process group on timeout
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                # Wait a bit for graceful termination
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    stdout, stderr = process.communicate()
            except ProcessLookupError:
                # Process already terminated
                stdout, stderr = "", ""
            
            output_text = ""
            if stdout:
                output_text = stdout
            if stderr:
                if output_text:
                    output_text += "\n" + stderr
                else:
                    output_text = stderr
            
            return {
                "success": False,
                "output": output_text,
                "error": f"Command timed out after {timeout} seconds (interrupted with SIGINT)",
                "timeout": True,
                "exit_code": 124  # Standard timeout exit code
            }
            
    except FileNotFoundError:
        return {
            "success": False,
            "error": "Shell not found. Cannot execute command.",
            "exit_code": 127  # Command not found
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to execute command: {str(e)}",
            "error_type": type(e).__name__,
            "exit_code": 1
        }

def handle_ctrl_c():
    """
    Handle Ctrl+C interrupt request.
    This would be used when the command is "ctrl+c" to interrupt running processes.
    """
    # In the standalone executable context, this would need to track
    # and interrupt previously running processes
    # For now, we just return a message
    return {
        "success": True,
        "output": "Interrupt signal sent (Ctrl+C)",
        "interrupted": True
    }

if __name__ == "__main__":
    # Special handling for ctrl+c command
    if len(sys.argv) > 1 and sys.argv[1].lower() == "ctrl+c":
        result = handle_ctrl_c()
        output_json(result)
        sys.exit(0)
    
    main()