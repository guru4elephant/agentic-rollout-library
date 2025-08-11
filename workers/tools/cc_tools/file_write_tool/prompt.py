# k8s_write_file.py
from workers.core.base_tool import AgenticBaseTool
from workers.core.tool_schemas import create_openai_tool_schema, ToolResult
from kodo import KubernetesManager

class K8sWriteFile(AgenticBaseTool):
    def __init__(self, config=None):
        super().__init__(config)
        self.pod_name = config.get("pod_name")
        self.namespace = config.get("namespace", "default")
        self.k8s_manager = KubernetesManager(namespace=self.namespace)
    
    def get_openai_tool_schema(self):
        return create_openai_tool_schema(
            name="k8s_write_file",
            description="Write a file to the local filesystem. Overwrites the existing file if there is one. Before using this tool: 1. Use the ReadFile tool to understand the file's contents and context 2. Directory Verification (only applicable when creating new files): Use the LS tool to verify the parent directory exists and is the correct location",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path where the content should be written"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                }
            },
            required=["path", "content"]
        )
    
    async def execute_tool(self, instance_id, parameters, **kwargs):
        try:
            path = parameters.get("path")
            content = parameters.get("content")
            
            if not path:
                return ToolResult(success=False, error="Path parameter is required")
            
            if content is None:
                content = ""
            
            # Escape content for shell command
            escaped_content = content.replace("'", "'\"'\"'")
            
            # Write file using shell command
            write_command = f"echo '{escaped_content}' > '{path}'"
            output, exit_code = self.k8s_manager.execute_command(self.pod_name, write_command)
            
            if exit_code != 0:
                return ToolResult(success=False, error=f"Failed to write file: {output}")
            
            # Verify file was written by checking its existence
            verify_command = f"ls -la '{path}'"
            verify_output, verify_exit_code = self.k8s_manager.execute_command(self.pod_name, verify_command)
            
            if verify_exit_code != 0:
                return ToolResult(success=False, error=f"File write verification failed: {verify_output}")
            
            result_message = f"Successfully wrote file to {path}"
            
            return ToolResult(
                success=True,
                result=result_message,
                metrics={"file_path": path, "content_length": len(content)}
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
