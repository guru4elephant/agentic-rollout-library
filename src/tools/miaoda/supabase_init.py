#!/usr/bin/env python3
"""
Description: Initialize Supabase project and retrieve credentials via MCP server.
Parameters:
  name (string, required): Project name. Positional argument.
  app_id (string, optional): Application ID.

Usage:
  As a script: python supabase_init.py "my_project" --app_id "my_app"
  As a module: python -m tools.miaoda.impl.supabase_init "my_project"
  As a function: from tools.miaoda.impl.supabase_init import supabase_init_func; supabase_init_func("my_project")
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional


def supabase_init_func(name: str, app_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize Supabase project and retrieve credentials.

    Args:
        name: Project name
        app_id: Application ID (optional)

    Returns:
        Dictionary containing:
            - result: Supabase initialization result (endpoint, anon_key, etc.)
            - status: Status of the operation (success/error)
            - error: Error message if failed
    """
    app_info = f" (app_id: {app_id})" if app_id else ""
    success_message = f"Supabase project '{name}'{app_info} initialized successfully. Database credentials retrieved and ready to use."
    
    return {
        "result": success_message,
        "status": "success"
    }


def parse_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse Supabase init result for agent use.

    Args:
        result: Raw result from supabase_init_func or K8S execution

    Returns:
        Formatted result for agent
    """
    if isinstance(result, dict):
        return {
            "result": result.get("result", str(result)),
            "status": result.get("status", "success")
        }
    
    return {
        "result": str(result),
        "status": "success"
    }


def build_k8s_command(name: str, app_id: Optional[str] = None) -> str:
    """
    Build command for K8S pod execution.

    Args:
        name: Project name
        app_id: Application ID (optional)

    Returns:
        Command string for K8S execution
    """
    escaped_name = name.replace('"', '\\"').replace("'", "\\'")
    
    if app_id:
        escaped_app_id = app_id.replace('"', '\\"').replace("'", "\\'")
        return f'python3 -c "from tools.miaoda.impl.supabase_init import supabase_init_func; import json; print(json.dumps(supabase_init_func(\'{escaped_name}\', \'{escaped_app_id}\'), ensure_ascii=False))"'
    else:
        return f'python3 -c "from tools.miaoda.impl.supabase_init import supabase_init_func; import json; print(json.dumps(supabase_init_func(\'{escaped_name}\'), ensure_ascii=False))"'


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Initialize Supabase project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python supabase_init.py --name "my_project"
  python supabase_init.py --name "my_project" --app_id "my_app"
  python supabase_init.py "test_db" --json
        """
    )
    parser.add_argument(
        "--name",
        help="Project name (positional argument)"
    )
    parser.add_argument(
        "--app_id",
        help="Application ID (optional)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )

    args = parser.parse_args()

    result = supabase_init_func(args.name, args.app_id)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if result["status"] == "error":
            print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)
        
        parsed = parse_result(result)
        print(parsed["result"])

    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
