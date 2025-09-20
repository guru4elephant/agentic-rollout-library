"""
R2E Tools Refactored - Modular implementation of R2E tools.

Each tool consists of:
1. A tool file (e.g., r2e_file_editor_tool.py) - Handles function call description and routing
2. An executable file (e.g., r2e_file_editor_exe.py) - Contains the actual implementation

This separation allows:
- Remote execution in K8s pods by copying only the executable
- Clear separation of concerns between API and implementation
- Easier testing and maintenance
"""

from .r2e_file_editor_tool import get_tool_schema as get_file_editor_schema
from .r2e_file_editor_tool import execute_tool as execute_file_editor

from .r2e_bash_executor_tool import get_tool_schema as get_bash_executor_schema
from .r2e_bash_executor_tool import execute_tool as execute_bash_executor

from .r2e_str_replace_editor_tool import get_tool_schema as get_str_replace_editor_schema
from .r2e_str_replace_editor_tool import execute_tool as execute_str_replace_editor

__all__ = [
    'get_file_editor_schema',
    'execute_file_editor',
    'get_bash_executor_schema', 
    'execute_bash_executor',
    'get_str_replace_editor_schema',
    'execute_str_replace_editor'
]