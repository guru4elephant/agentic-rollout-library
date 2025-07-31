#!/usr/bin/env python3
"""
Search tool for finding text in files and directories.
Provides grep-like functionality and directory search capabilities.
"""

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.base_tool import AgenticBaseTool
from ..core.tool_schemas import OpenAIFunctionToolSchema, ToolResult, create_openai_tool_schema

logger = logging.getLogger(__name__)


class SearchTool(AgenticBaseTool):
    """Tool for searching text in files and directories."""
    
    def __init__(self, config: Dict = None):
        """Initialize search tool."""
        super().__init__(config)
        self.max_results = self.config.get("max_results", 100)
        self.max_file_size = self.config.get("max_file_size", 1024 * 1024)  # 1MB
        self.search_extensions = set(self.config.get("search_extensions", [
            ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".sh", ".js", ".ts",
            ".html", ".css", ".xml", ".csv", ".log", ".c", ".cpp", ".h", ".java"
        ]))
        self.exclude_dirs = set(self.config.get("exclude_dirs", [
            ".git", ".svn", "__pycache__", "node_modules", ".pytest_cache",
            ".mypy_cache", ".tox", "venv", ".venv", "env", ".env"
        ]))
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema for search tool."""
        return create_openai_tool_schema(
            name="search",
            description="Search for text patterns in files and directories. Supports regex patterns and various search modes.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "Search command type",
                    "enum": ["search_text", "search_files", "search_dir"]
                },
                "pattern": {
                    "type": "string",
                    "description": "Text pattern or regex to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path to search in"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search should be case sensitive (default: false)"
                },
                "regex": {
                    "type": "boolean",
                    "description": "Whether pattern is a regular expression (default: false)"
                },
                "whole_words": {
                    "type": "boolean",
                    "description": "Match whole words only (default: false)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return"
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File extensions to search (e.g., ['.py', '.txt'])"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show around matches (default: 2)"
                }
            },
            required=["command", "pattern", "path"]
        )
    
    async def execute_tool(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute search operation."""
        try:
            command = parameters["command"]
            pattern = parameters["pattern"]
            path_str = parameters["path"]
            
            # Validate path
            path = Path(path_str)
            if not path.exists():
                return ToolResult(success=False, error=f"Path does not exist: {path}")
            
            # Parse search options
            case_sensitive = parameters.get("case_sensitive", False)
            is_regex = parameters.get("regex", False)
            whole_words = parameters.get("whole_words", False)
            max_results = parameters.get("max_results", self.max_results)
            file_extensions = parameters.get("file_extensions", list(self.search_extensions))
            context_lines = parameters.get("context_lines", 2)
            
            if command == "search_text":
                return await self._search_text_in_files(
                    pattern, path, case_sensitive, is_regex, whole_words,
                    max_results, file_extensions, context_lines
                )
            elif command == "search_files":
                return await self._search_file_names(
                    pattern, path, case_sensitive, is_regex, max_results
                )
            elif command == "search_dir":
                return await self._search_directory_structure(
                    pattern, path, case_sensitive, is_regex, max_results
                )
            else:
                return ToolResult(success=False, error=f"Unknown search command: {command}")
                
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _search_text_in_files(self, pattern: str, path: Path, case_sensitive: bool,
                                   is_regex: bool, whole_words: bool, max_results: int,
                                   file_extensions: List[str], context_lines: int) -> ToolResult:
        """Search for text pattern in files."""
        try:
            # Compile search pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            
            if is_regex:
                search_pattern = re.compile(pattern, flags)
            elif whole_words:
                search_pattern = re.compile(r'\b' + re.escape(pattern) + r'\b', flags)
            else:
                search_pattern = re.compile(re.escape(pattern), flags)
            
            results = []
            files_searched = 0
            
            if path.is_file():
                # Search single file
                if self._should_search_file(path, file_extensions):
                    file_results = self._search_in_file(path, search_pattern, context_lines)
                    results.extend(file_results)
                    files_searched = 1
            else:
                # Search directory
                for file_path in self._walk_directory(path, file_extensions):
                    if len(results) >= max_results:
                        break
                    
                    file_results = self._search_in_file(file_path, search_pattern, context_lines)
                    results.extend(file_results[:max_results - len(results)])
                    files_searched += 1
            
            return ToolResult(
                success=True,
                result={
                    "matches": results[:max_results],
                    "total_matches": len(results),
                    "files_searched": files_searched,
                    "pattern": pattern,
                    "search_path": str(path)
                },
                metrics={
                    "matches_found": len(results),
                    "files_searched": files_searched
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Text search failed: {e}")
    
    async def _search_file_names(self, pattern: str, path: Path, case_sensitive: bool,
                                is_regex: bool, max_results: int) -> ToolResult:
        """Search for files matching name pattern."""
        try:
            # Compile pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            if is_regex:
                search_pattern = re.compile(pattern, flags)
            else:
                search_pattern = re.compile(re.escape(pattern), flags)
            
            results = []
            
            if path.is_file():
                # Check single file
                if search_pattern.search(path.name):
                    results.append({
                        "path": str(path),
                        "name": path.name,
                        "type": "file",
                        "size": path.stat().st_size
                    })
            else:
                # Search directory
                for item in self._walk_all_items(path):
                    if len(results) >= max_results:
                        break
                    
                    if search_pattern.search(item.name):
                        results.append({
                            "path": str(item),
                            "name": item.name,
                            "type": "directory" if item.is_dir() else "file",
                            "size": item.stat().st_size if item.is_file() else None
                        })
            
            return ToolResult(
                success=True,
                result={
                    "matches": results,
                    "total_matches": len(results),
                    "pattern": pattern,
                    "search_path": str(path)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"File name search failed: {e}")
    
    async def _search_directory_structure(self, pattern: str, path: Path, case_sensitive: bool,
                                         is_regex: bool, max_results: int) -> ToolResult:
        """Search directory structure and list matching paths."""
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            if is_regex:
                search_pattern = re.compile(pattern, flags)
            else:
                search_pattern = re.compile(re.escape(pattern), flags)
            
            results = []
            
            for item in self._walk_all_items(path):
                if len(results) >= max_results:
                    break
                
                # Search in full path
                if search_pattern.search(str(item)):
                    results.append({
                        "path": str(item),
                        "relative_path": str(item.relative_to(path)),
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "depth": len(item.relative_to(path).parts)
                    })
            
            return ToolResult(
                success=True,
                result={
                    "matches": results,
                    "total_matches": len(results),
                    "pattern": pattern,
                    "search_path": str(path)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Directory search failed: {e}")
    
    def _search_in_file(self, file_path: Path, pattern: re.Pattern, context_lines: int) -> List[Dict]:
        """Search for pattern in a single file."""
        results = []
        
        try:
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                return results
            
            # Read file content
            try:
                content = file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    content = file_path.read_text(encoding='latin-1')
                except:
                    return results  # Skip binary files
            
            lines = content.splitlines()
            
            # Search each line
            for line_num, line in enumerate(lines, 1):
                matches = list(pattern.finditer(line))
                
                for match in matches:
                    # Get context lines
                    context_start = max(0, line_num - context_lines - 1)
                    context_end = min(len(lines), line_num + context_lines)
                    
                    context = []
                    for i in range(context_start, context_end):
                        prefix = ">>> " if i == line_num - 1 else "    "
                        context.append(f"{prefix}{i + 1:4d}: {lines[i]}")
                    
                    results.append({
                        "file": str(file_path),
                        "line_number": line_num,
                        "column": match.start() + 1,
                        "match_text": match.group(),
                        "line_content": line,
                        "context": "\n".join(context)
                    })
            
        except Exception as e:
            logger.warning(f"Failed to search in file {file_path}: {e}")
        
        return results
    
    def _walk_directory(self, path: Path, file_extensions: List[str]):
        """Walk directory and yield files to search."""
        for root, dirs, files in os.walk(path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            root_path = Path(root)
            for file_name in files:
                file_path = root_path / file_name
                if self._should_search_file(file_path, file_extensions):
                    yield file_path
    
    def _walk_all_items(self, path: Path):
        """Walk directory and yield all items (files and directories)."""
        for root, dirs, files in os.walk(path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            root_path = Path(root)
            
            # Yield directories
            for dir_name in dirs:
                yield root_path / dir_name
            
            # Yield files
            for file_name in files:
                yield root_path / file_name
    
    def _should_search_file(self, file_path: Path, file_extensions: List[str]) -> bool:
        """Check if file should be searched based on extension and size."""
        if not file_extensions:
            return True
        
        if file_path.suffix.lower() in file_extensions:
            try:
                return file_path.stat().st_size <= self.max_file_size
            except:
                return False
        
        return False