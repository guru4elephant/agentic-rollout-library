# k8s_edit_tool.py
from workers.core.base_tool import AgenticBaseTool
from workers.core.tool_schemas import create_openai_tool_schema, ToolResult
from kodo import KubernetesManager

class K8sEditTool(AgenticBaseTool):
    def __init__(self, config=None):
        super().__init__(config)
        self.pod_name = config.get("pod_name") if config else None
        self.namespace = config.get("namespace", "default") if config else "default"
        self.k8s_manager = KubernetesManager(namespace=self.namespace)
    
    def get_openai_tool_schema(self):
        return create_openai_tool_schema(
            name="k8s_edit",
            description="This is a tool for editing files in Kubernetes pods. For moving or renaming files, use the Bash tool with the 'mv' command instead. For larger edits, use the Write tool to overwrite files. For Jupyter notebooks (.ipynb files), use the notebook edit tool instead.\n\nBefore using this tool:\n1. Use the View tool to understand the file's contents and context\n2. Verify the directory path is correct (only applicable when creating new files)\n\nTo make a file edit, provide:\n1. file_path: The absolute path to the file to modify (must be absolute, not relative)\n2. old_string: The text to replace (must be unique within the file, and must match exactly)\n3. new_string: The edited text to replace the old_string\n\nCRITICAL REQUIREMENTS:\n1. UNIQUENESS: The old_string MUST uniquely identify the specific instance with 3-5 lines of context before and after\n2. SINGLE INSTANCE: This tool can only change ONE instance at a time\n3. VERIFICATION: Check how many instances exist before using\n\nFor new files, use empty old_string and the file contents as new_string.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to modify"
                    },
                    "old_string": {
                        "type": "string", 
                        "description": "The text to replace (must be unique within the file, empty for new files)"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The edited text to replace the old_string"
                    }
                }
            },
            required=["file_path", "old_string", "new_string"]
        )
    
    async def execute_tool(self, instance_id, parameters, **kwargs):
        try:
            file_path = parameters.get("file_path")
            old_string = parameters.get("old_string")
            new_string = parameters.get("new_string")
            
            if not file_path:
                return ToolResult(success=False, error="file_path is required")
            
            # Check if file exists
            check_cmd = f"test -f '{file_path}' && echo 'exists' || echo 'not_exists'"
            check_output, check_exit_code = self.k8s_manager.execute_command(self.pod_name, check_cmd)
            
            file_exists = check_output.strip() == 'exists'
            
            if not file_exists and old_string:
                return ToolResult(success=False, error=f"File {file_path} does not exist")
            
            # Handle new file creation
            if not file_exists and not old_string:
                # Create directory if needed
                dir_path = "/".join(file_path.split("/")[:-1])
                if dir_path:
                    mkdir_cmd = f"mkdir -p '{dir_path}'"
                    self.k8s_manager.execute_command(self.pod_name, mkdir_cmd)
                
                # Create new file
                escaped_content = new_string.replace("'", "'\"'\"'")
                create_cmd = f"cat > '{file_path}' << 'EOF'\n{escaped_content}\nEOF"
                output, exit_code = self.k8s_manager.execute_command(self.pod_name, create_cmd)
                
                if exit_code != 0:
                    return ToolResult(success=False, error=f"Failed to create file: {output}")
                
                return ToolResult(
                    success=True,
                    result=f"Successfully created new file: {file_path}",
                    metrics={"operation": "create", "file_path": file_path}
                )
            
            # Handle file editing
            if not old_string:
                return ToolResult(success=False, error="old_string cannot be empty for existing files")
            
            # Read current file content
            read_cmd = f"cat '{file_path}'"
            content_output, read_exit_code = self.k8s_manager.execute_command(self.pod_name, read_cmd)
            
            if read_exit_code != 0:
                return ToolResult(success=False, error=f"Failed to read file: {content_output}")
            
            current_content = content_output
            
            # Check if old_string exists and is unique
            if old_string not in current_content:
                return ToolResult(success=False, error="old_string not found in file")
            
            occurrences = current_content.count(old_string)
            if occurrences > 1:
                return ToolResult(success=False, error=f"old_string appears {occurrences} times in file. It must be unique.")
            
            # Perform replacement
            new_content = current_content.replace(old_string, new_string, 1)
            
            # Write updated content back to file
            escaped_content = new_content.replace("'", "'\"'\"'")
            write_cmd = f"cat > '{file_path}' << 'EOF'\n{escaped_content}\nEOF"
            write_output, write_exit_code = self.k8s_manager.execute_command(self.pod_name, write_cmd)
            
            if write_exit_code != 0:
                return ToolResult(success=False, error=f"Failed to write file: {write_output}")
            
            return ToolResult(
                success=True,
                result=f"Successfully edited file: {file_path}",
                metrics={"operation": "edit", "file_path": file_path, "old_length": len(old_string), "new_length": len(new_string)}
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
