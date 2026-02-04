# R2E Tools Refactored

This directory contains a refactored implementation of R2E tools following a modular architecture that separates tool descriptions from executables.

## Architecture

Each tool consists of two files:

### 1. Tool Description File (`*_tool.py`)
- Contains the function call schema (OpenAI format)
- Handles parameter validation
- Routes execution to either local or remote mode
- Manages K8s integration when needed

### 2. Executable File (`*_exe.py`)
- Contains the actual implementation logic
- Can run as a standalone program
- Takes command-line arguments
- Outputs results as JSON to stdout
- Outputs errors to stderr

## Tools Available

### r2e_file_editor
- **Description file**: `r2e_file_editor_tool.py`
- **Executable**: `r2e_file_editor_exe.py`
- **Commands**: view, create, str_replace, insert, undo_edit
- **Purpose**: Full-featured file editing and viewing

### r2e_bash_executor
- **Description file**: `r2e_bash_executor_tool.py`
- **Executable**: `r2e_bash_executor_exe.py`
- **Purpose**: Execute bash commands with timeout and security restrictions
- **Blocked commands**: git, ipython, jupyter, nohup

### r2e_str_replace_editor
- **Description file**: `r2e_str_replace_editor_tool.py`
- **Executable**: `r2e_file_editor_exe.py` (reuses file editor executable)
- **Commands**: view, create, str_replace, insert (no undo_edit)
- **Purpose**: Simplified file editor focused on string replacement

## Usage

### Local Execution

```python
from r2e_tools_refactored import execute_file_editor

# View a file
result = execute_file_editor({
    "command": "view",
    "path": "/path/to/file.py"
})

# Create a file
result = execute_file_editor({
    "command": "create",
    "path": "/path/to/new_file.py",
    "file_text": "print('Hello, World!')"
})
```

### Remote K8s Execution

```python
from r2e_tools_refactored import execute_bash_executor

# Execute command in K8s pod
result = execute_bash_executor(
    parameters={"cmd": "ls -la"},
    execution_mode="remote",
    pod_name="my-pod",
    namespace="default",
    working_dir="/testbed"
)
```

### Command Line Usage

Each tool can be tested directly from command line:

```bash
# Test file editor
python3 r2e_file_editor_tool.py --command view --path /tmp/test.txt

# Test bash executor
python3 r2e_bash_executor_tool.py "ls -la" --timeout 30

# Test with remote execution
python3 r2e_bash_executor_tool.py "python test.py" \
  --execution_mode remote \
  --pod_name my-pod \
  --namespace default
```

### Direct Executable Usage

The executables can be called directly (useful for K8s deployment):

```bash
# File editor executable
python3 r2e_file_editor_exe.py view /tmp/test.txt

# With optional parameters as JSON
python3 r2e_file_editor_exe.py str_replace /tmp/test.txt \
  '{"old_str": "foo", "new_str": "bar"}'

# Bash executor executable
python3 r2e_bash_executor_exe.py "ls -la" 30
```

## K8s Deployment

For K8s deployment, copy only the executable files to the pod:

```bash
# Copy executables to pod
kubectl cp r2e_file_editor_exe.py my-pod:/path/to/r2e_file_editor_exe.py
kubectl cp r2e_bash_executor_exe.py my-pod:/path/to/r2e_bash_executor_exe.py
```

The tool description files remain on the host and execute commands remotely via `kubectl exec`.

## Output Format

All tools output JSON with consistent structure:

### Success Response
```json
{
    "success": true,
    "output": "Command output here",
    "type": "file|directory",
    "total_lines": 100
}
```

### Error Response
```json
{
    "success": false,
    "error": "Error message",
    "error_type": "ExceptionType",
    "exit_code": 1
}
```

## Security Notes

- Bash executor blocks potentially dangerous commands (git, ipython, jupyter, nohup)
- File operations validate paths and check permissions
- Remote execution requires proper K8s RBAC configuration
- Timeout protection prevents runaway processes