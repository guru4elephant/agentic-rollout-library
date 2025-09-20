#!/usr/bin/env python3
"""
R2E File Editor executable - Actual implementation that runs as a standalone program.
This file contains the core logic and outputs results to stdout/stderr.
"""

import os
import sys
import json
import ast
import chardet
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# R2E constants
SNIPPET_LINES = 4
MAX_RESPONSE_LEN = 10000
TRUNCATED_MESSAGE = (
    "<response clipped><NOTE>To save on context only part of this file has been "
    "shown to you. You should retry this tool after you have searched inside the file "
    "with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
)

def main():
    """Main entry point for the executable."""
    try:
        # Parse command line arguments
        if len(sys.argv) < 3:
            error_output("Usage: r2e_file_editor_exe.py <command> <path> [optional_params_json]")
            sys.exit(1)
        
        command = sys.argv[1]
        path = sys.argv[2]
        
        # Parse optional parameters if provided
        optional_params = {}
        if len(sys.argv) > 3:
            try:
                optional_params = json.loads(sys.argv[3])
            except json.JSONDecodeError as e:
                error_output(f"Failed to parse optional parameters: {e}")
                sys.exit(1)
        
        # Execute command
        if command == "view":
            result = handle_view(path, optional_params.get("view_range"), optional_params.get("concise", False))
        elif command == "create":
            if "file_text" not in optional_params:
                error_output("file_text parameter required for create command")
                sys.exit(1)
            result = handle_create(path, optional_params["file_text"])
        elif command == "str_replace":
            if "old_str" not in optional_params:
                error_output("old_str parameter required for str_replace command")
                sys.exit(1)
            result = handle_str_replace(path, optional_params["old_str"], optional_params.get("new_str", ""))
        elif command == "insert":
            if "insert_line" not in optional_params or "new_str" not in optional_params:
                error_output("insert_line and new_str parameters required for insert command")
                sys.exit(1)
            result = handle_insert(path, optional_params["insert_line"], optional_params["new_str"])
        elif command == "undo_edit":
            result = handle_undo_edit(path)
        else:
            error_output(f"Unknown command: {command}")
            sys.exit(1)
        
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

def handle_view(path_str: str, view_range: Optional[List[int]], concise: bool) -> Dict[str, Any]:
    """Handle view command."""
    try:
        path = Path(path_str)
        
        if not path.exists():
            return {"success": False, "error": f"Path does not exist: {path}"}
        
        if path.is_dir():
            # List directory contents
            import subprocess
            cmd = ["find", str(path), "-maxdepth", "2", "-not", "-path", "*/.*"]
            
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if proc.stderr:
                return {"success": False, "error": proc.stderr.strip()}
            
            msg = (f"Here's the files and directories up to 2 levels deep in {path}, "
                   "excluding hidden:\n" + proc.stdout)
            msg = maybe_truncate(msg)
            
            return {"success": True, "output": msg, "type": "directory"}
        
        else:
            # View file
            # Auto-enable concise mode for large Python files
            if path.suffix == ".py" and not view_range and not concise:
                file_text = read_file(path)
                if len(file_text.splitlines()) > 110:
                    concise = True
            
            # Get file content
            if path.suffix == ".py" and concise:
                lines_with_numbers = get_elided_lines(path)
            else:
                file_text = read_file(path)
                lines_with_numbers = [(i, line) for i, line in enumerate(file_text.splitlines())]
            
            # Apply view range
            lines_with_numbers = apply_view_range(lines_with_numbers, view_range)
            
            # Format output
            if concise:
                output = f"Here is a condensed view for file: {path}; [Note: Useful for understanding file structure in a concise manner.]\n"
            else:
                output = f"Here's the result of running `cat -n` on the file: {path}:\n"
            
            for i, text in lines_with_numbers:
                output += f"{i+1:6d} {text}\n"
            
            output = maybe_truncate(output)
            
            return {
                "success": True,
                "output": output,
                "type": "file",
                "total_lines": len(lines_with_numbers)
            }
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def handle_create(path_str: str, file_text: str) -> Dict[str, Any]:
    """Handle create command."""
    try:
        path = Path(path_str)
        
        if path.exists():
            return {"success": False, "error": f"File already exists at: {path}. Cannot overwrite with 'create'."}
        
        # Lint check for Python files
        if path.suffix == ".py":
            lint_error = lint_check(file_text)
            if lint_error:
                return {"success": False, "error": f"Linting failed:\n{lint_error}"}
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        path.write_text(file_text, encoding="utf-8")
        
        # Save to history (if state management is needed)
        save_to_history(str(path), "")
        
        # Format output
        output = f"File created at {path}. "
        output += make_output(file_text, str(path))
        output += "Review the file and make sure that it is as expected. Edit the file if necessary."
        
        return {"success": True, "output": output, "path": str(path)}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def handle_str_replace(path_str: str, old_str: str, new_str: str) -> Dict[str, Any]:
    """Handle str_replace command."""
    try:
        path = Path(path_str)
        
        if not path.exists():
            return {"success": False, "error": f"File does not exist: {path}"}
        
        # Read file
        file_content = read_file(path).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs()
        
        # Check occurrences
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            return {"success": False, "error": f"No occurrences of '{old_str}' found in {path}"}
        if occurrences > 1:
            return {
                "success": False,
                "error": f"Multiple occurrences of '{old_str}' found in {path}. Please ensure it is unique."
            }
        
        # Replace
        updated_text = file_content.replace(old_str, new_str)
        
        # Lint check
        if path.suffix == ".py":
            lint_error = lint_check(updated_text)
            if lint_error:
                return {"success": False, "error": f"Linting failed:\n{lint_error}"}
        
        # Save history and write
        save_to_history(str(path), file_content)
        path.write_text(updated_text, encoding="utf-8")
        
        # Create snippet
        replacement_line = file_content.split(old_str)[0].count("\n")
        snippet = create_snippet(updated_text, replacement_line, new_str.count("\n"))
        
        output = f"The file {path} has been edited. "
        output += make_output(snippet, f"a snippet of {path}", replacement_line - SNIPPET_LINES + 1)
        output += "Review the changes and make sure they are as expected. Edit the file again if necessary."
        
        return {"success": True, "output": output}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def handle_insert(path_str: str, insert_line: int, new_str: str) -> Dict[str, Any]:
    """Handle insert command."""
    try:
        path = Path(path_str)
        
        if not path.exists():
            return {"success": False, "error": f"File does not exist: {path}"}
        
        # Read file
        old_text = read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_lines = old_text.split("\n")
        
        # Validate line number
        if insert_line < 0 or insert_line > len(file_lines):
            return {
                "success": False,
                "error": f"Invalid insert_line {insert_line}. Must be in [0, {len(file_lines)}]."
            }
        
        # Insert
        new_lines = new_str.split("\n")
        updated_lines = file_lines[:insert_line] + new_lines + file_lines[insert_line:]
        updated_text = "\n".join(updated_lines)
        
        # Lint check
        if path.suffix == ".py":
            lint_error = lint_check(updated_text)
            if lint_error:
                return {"success": False, "error": f"Linting failed:\n{lint_error}"}
        
        # Save history and write
        save_to_history(str(path), old_text)
        path.write_text(updated_text, encoding="utf-8")
        
        # Create snippet
        snippet_lines = (
            file_lines[max(0, insert_line - SNIPPET_LINES):insert_line] +
            new_lines +
            file_lines[insert_line:insert_line + SNIPPET_LINES]
        )
        snippet = "\n".join(snippet_lines)
        
        output = f"The file {path} has been edited. "
        output += make_output(snippet, "a snippet of the edited file", max(1, insert_line - SNIPPET_LINES + 1))
        output += "Review the changes and make sure they are as expected. Edit the file again if necessary."
        
        return {"success": True, "output": output}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def handle_undo_edit(path_str: str) -> Dict[str, Any]:
    """Handle undo_edit command."""
    try:
        path = Path(path_str)
        
        # Load history
        history = load_history()
        if str(path) not in history or not history[str(path)]:
            return {"success": False, "error": f"No previous edits found for {path} to undo."}
        
        # Restore previous content
        old_text = history[str(path)].pop()
        save_history_state(history)
        
        path.write_text(old_text, encoding="utf-8")
        
        output = f"Last edit to {path} undone successfully. "
        output += make_output(old_text, str(path))
        
        return {"success": True, "output": output}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# Helper functions

def read_file(path: Path) -> str:
    """Read file with encoding detection."""
    try:
        encoding = chardet.detect(path.read_bytes())["encoding"]
        if encoding is None:
            encoding = "utf-8"
        return path.read_text(encoding=encoding)
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")

def maybe_truncate(content: str) -> str:
    """Truncate content if too long."""
    if len(content) <= MAX_RESPONSE_LEN:
        return content
    return content[:MAX_RESPONSE_LEN] + TRUNCATED_MESSAGE

def make_output(file_content: str, file_descriptor: str, init_line: int = 1) -> str:
    """Format output R2E-style."""
    file_content = maybe_truncate(file_content)
    file_content = file_content.expandtabs()
    
    lines = file_content.split("\n")
    numbered = "\n".join(f"{i + init_line:6}\t{line}" for i, line in enumerate(lines))
    return f"Here's the result of running `cat -n` on {file_descriptor}:\n" + numbered + "\n"

def create_snippet(text: str, center_line: int, extra_lines: int = 0) -> str:
    """Create snippet around a line."""
    lines = text.split("\n")
    start = max(0, center_line - SNIPPET_LINES)
    end = center_line + SNIPPET_LINES + extra_lines + 1
    return "\n".join(lines[start:end])

def apply_view_range(lines_with_numbers: List[Tuple[int, str]], 
                     view_range: Optional[List[int]]) -> List[Tuple[int, str]]:
    """Apply view range to lines."""
    if not view_range or len(view_range) != 2:
        return lines_with_numbers
    
    start, end = view_range
    total_lines = len(lines_with_numbers)
    
    if not (1 <= start <= total_lines):
        return lines_with_numbers
    
    if end == -1:
        end = total_lines
    elif end < start or end > total_lines:
        return lines_with_numbers
    
    # Filter by 1-based index
    result = []
    for i, text in lines_with_numbers:
        one_based = i + 1
        if start <= one_based <= end:
            result.append((i, text))
    
    return result

def lint_check(content: str) -> Optional[str]:
    """Check Python syntax."""
    try:
        ast.parse(content)
        return None
    except SyntaxError as e:
        return str(e)

def get_elided_lines(path: Path) -> List[Tuple[int, str]]:
    """Get condensed view of Python file."""
    file_text = read_file(path)
    try:
        tree = ast.parse(file_text, filename=str(path))
    except SyntaxError as e:
        raise Exception(f"Syntax error in file {path}: {e}")
    
    def max_lineno_in_subtree(n: ast.AST) -> int:
        m = getattr(n, "lineno", 0)
        for child in ast.iter_child_nodes(n):
            m = max(m, max_lineno_in_subtree(child))
        return m
    
    # Gather line ranges for large function bodies
    elide_line_ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            last_stmt = node.body[-1]
            if hasattr(last_stmt, "end_lineno") and last_stmt.end_lineno:
                body_start = node.body[0].lineno - 1
                body_end = last_stmt.end_lineno - 1
            else:
                body_start = node.body[0].lineno - 1
                body_end = max_lineno_in_subtree(last_stmt) - 1
            
            if (body_end - body_start) >= 3:
                elide_line_ranges.append((body_start, body_end))
    
    # Build elided view
    elide_lines = {
        line for (start, end) in elide_line_ranges 
        for line in range(start, end + 1)
    }
    
    elide_messages = [
        (start, f"... eliding lines {start+1}-{end+1} ...")
        for (start, end) in elide_line_ranges
    ]
    
    all_lines = file_text.splitlines()
    keep_lines = [
        (i, line) for i, line in enumerate(all_lines) 
        if i not in elide_lines
    ]
    
    combined = elide_messages + keep_lines
    combined.sort(key=lambda x: x[0])
    
    return combined

# Simple history management (state file)

STATE_FILE = "/var/tmp/r2e_editor_state.json"

def load_history() -> Dict[str, List[str]]:
    """Load file edit history."""
    try:
        if Path(STATE_FILE).exists():
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_to_history(file_path: str, old_content: str):
    """Save file content to history."""
    try:
        history = load_history()
        if file_path not in history:
            history[file_path] = []
        history[file_path].append(old_content)
        save_history_state(history)
    except Exception:
        pass  # Ignore history errors

def save_history_state(history: Dict[str, List[str]]):
    """Save history state to file."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(history, f)
    except Exception:
        pass  # Ignore save errors

if __name__ == "__main__":
    main()
