#!/usr/bin/env python3
"""
Test GeneralAgent with R2E tools for SWE-bench-verified processing.
This script reads JSONL files, creates pods using kodo, runs agents, and collects patches.
"""

import asyncio
import sys
import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    pass  # python-dotenv not installed, skip

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.agents.general_agent import GeneralAgent
from workers.core import create_tool
from workers.utils import create_llm_client
from workers.core.trajectory import TrajectoryStep, StepType
import logging
import re

# Import kodo for pod management
try:
    from kodo import ContainerRunner
except ImportError:
    print("ERROR: kodo package not found. Please install it with: pip install kodo")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM configuration
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")

# Import R2E configurations from test_r2e_general_agent.py
from test_r2e_general_agent import (
    CUSTOM_TOOL_DESCRIPTIONS,
    parse_xml_action_custom,
    CustomDescriptionWrapper,
    generate_custom_system_prompt
)


class SWEBenchRunner:
    """Runner for SWE-bench-verified instances."""
    
    def __init__(self, namespace: str = "default", kubeconfig_path: Optional[str] = None):
        """Initialize the SWE-bench runner.
        
        Args:
            namespace: Kubernetes namespace to use
            kubeconfig_path: Path to kubeconfig file (optional)
        """
        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path
        self.kodo_runner = None
        self.patches = {}  # Store patches by instance_id
        
    def _init_kodo(self):
        """Initialize kodo ContainerRunner."""
        if not self.kodo_runner:
            self.kodo_runner = ContainerRunner(
                backend="kubernetes",
                namespace=self.namespace,
                kubeconfig_path=self.kubeconfig_path
            )
            logger.info(f"Initialized kodo runner for namespace: {self.namespace}")
    
    async def process_instance(self, instance: Dict[str, Any]) -> Optional[str]:
        """Process a single SWE-bench instance.
        
        Args:
            instance: Dictionary with instance_id, issue, image fields
            
        Returns:
            The patch string if successful, None otherwise
        """
        instance_id = instance.get("instance_id", "unknown")
        issue = instance.get("issue", "")
        image = instance.get("image", "")
        
        if not all([instance_id, issue, image]):
            logger.error(f"Missing required fields for instance {instance_id}")
            return None
            
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing instance: {instance_id}")
        logger.info(f"Image: {image}")
        logger.info(f"Issue: {issue[:100]}..." if len(issue) > 100 else f"Issue: {issue}")
        logger.info(f"{'='*80}")
        
        # Initialize kodo if needed
        self._init_kodo()
        
        # Create pod name from instance_id
        pod_name = f"swe-{instance_id.lower().replace('/', '-').replace('_', '-')[:40]}"
        pod = None
        
        try:
            # 1. Start pod using kodo
            logger.info(f"Starting pod: {pod_name}")
            pod = self.kodo_runner.start_container(
                image=image,
                name=pod_name,
                environment={
                    "PYTHONPATH": "/testbed",
                    "SWE_INSTANCE_ID": instance_id
                }
            )
            logger.info(f"Pod {pod_name} started successfully")
            
            # Wait for pod to be ready
            await asyncio.sleep(5)
            
            # 2. Initialize git repo to track changes
            logger.info("Initializing git repository...")
            output, exit_code = self.kodo_runner.execute_command(
                pod, 
                "cd /testbed && git init && git config user.email 'agent@swe.bench' && git config user.name 'SWE Agent' && git add -A && git commit -m 'Initial commit' || true"
            )
            if exit_code != 0:
                logger.warning(f"Git init had issues: {output}")
            
            # 3. Create agent with R2E tools configured for this pod
            k8s_config = {
                "execution_mode": "k8s",
                "pod_name": pod_name,
                "namespace": self.namespace,
                "kubeconfig_path": self.kubeconfig_path
            }
            
            # Create R2E tools
            base_tools = {
                "r2e_bash_executor": create_tool("R2EBashExecutor", k8s_config.copy()),
                "r2e_file_editor": create_tool("R2EFileEditor", k8s_config.copy()),
                "r2e_search": create_tool("R2ESearch", k8s_config.copy()),
                "r2e_submit": create_tool("R2ESubmit", {})
            }
            
            # Wrap tools with custom descriptions
            tools = {}
            for tool_name, tool in base_tools.items():
                if tool_name in CUSTOM_TOOL_DESCRIPTIONS:
                    tools[tool_name] = CustomDescriptionWrapper(tool, CUSTOM_TOOL_DESCRIPTIONS[tool_name])
                else:
                    tools[tool_name] = tool
            
            # Generate custom system prompt
            custom_system_prompt = generate_custom_system_prompt(
                tools,
                task_description="analyze and fix the reported issue in the repository",
                working_directory="/testbed",
                additional_instructions="\n- Focus on the specific issue described\n- Make minimal changes to fix the issue\n- Ensure your changes don't break existing functionality"
            )
            
            # Create agent
            agent = GeneralAgent(
                max_rounds=30,  # More rounds for complex issues
                debug=False,
                termination_tool_names=["r2e_submit"],
                action_parser=parse_xml_action_custom,
                system_prompt=custom_system_prompt
            )
            agent.set_tools(tools)
            
            # 4. Run agent with the issue
            logger.info("Running agent to solve the issue...")
            llm_client = create_llm_client(
                api_key=API_KEY,
                base_url=BASE_URL,
                model=MODEL_NAME,
                debug=False
            )
            
            # Prepare the prompt with issue details
            prompt = f"""
Please analyze and fix the following issue in the repository at /testbed:

{issue}

First explore the repository structure, understand the codebase, locate the relevant files, and then make the necessary changes to fix the issue. When you're done, call the finish function to submit your solution.
"""
            
            result = await agent.run_trajectory(
                prompt=prompt,
                llm_generate_func=llm_client.generate,
                request_id=f"swe_{instance_id}"
            )
            
            logger.info(f"Agent completed: {result.is_completed}")
            logger.info(f"Total steps: {len(result.steps)}")
            
            # 5. Get the patch using git diff
            logger.info("Generating patch...")
            
            # First add all changes
            output, exit_code = self.kodo_runner.execute_command(
                pod,
                "cd /testbed && git add -A"
            )
            
            # Get the diff
            output, exit_code = self.kodo_runner.execute_command(
                pod,
                "cd /testbed && git diff --cached"
            )
            
            if exit_code == 0 and output.strip():
                patch = output.strip()
                logger.info(f"Generated patch ({len(patch)} chars)")
                logger.debug(f"Patch preview: {patch[:500]}..." if len(patch) > 500 else f"Patch: {patch}")
                return patch
            else:
                logger.warning(f"No patch generated or git diff failed: {output}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing instance {instance_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Clean up pod
            if pod and self.kodo_runner:
                try:
                    logger.info(f"Cleaning up pod: {pod_name}")
                    # For kodo, cleanup is handled by the runner
                    # The pod variable is just the pod name when using kubernetes backend
                    pass
                except Exception as e:
                    logger.error(f"Error cleaning up pod: {e}")
    
    async def process_jsonl_file(self, jsonl_path: str, output_dir: str = "./swe_patches"):
        """Process all instances in a JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file
            output_dir: Directory to save patches
        """
        if not os.path.exists(jsonl_path):
            logger.error(f"JSONL file not found: {jsonl_path}")
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read instances
        instances = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        instances.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line: {e}")
        
        logger.info(f"Loaded {len(instances)} instances from {jsonl_path}")
        
        # Process each instance
        for i, instance in enumerate(instances):
            logger.info(f"\nProcessing instance {i+1}/{len(instances)}")
            
            patch = await self.process_instance(instance)
            
            if patch:
                instance_id = instance.get("instance_id", f"unknown_{i}")
                self.patches[instance_id] = patch
                
                # Save patch to file
                patch_file = os.path.join(output_dir, f"{instance_id.replace('/', '_')}.patch")
                with open(patch_file, 'w') as f:
                    f.write(patch)
                logger.info(f"Saved patch to: {patch_file}")
            
            # Add delay between instances to avoid overwhelming the system
            if i < len(instances) - 1:
                await asyncio.sleep(5)
        
        # Save summary
        summary_file = os.path.join(output_dir, "summary.json")
        summary = {
            "total_instances": len(instances),
            "successful_patches": len(self.patches),
            "instance_ids": list(self.patches.keys()),
            "timestamp": datetime.now().isoformat()
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to: {summary_file}")
        
        # Cleanup kodo runner
        if self.kodo_runner:
            try:
                self.kodo_runner.cleanup()
            except Exception as e:
                logger.error(f"Error during final cleanup: {e}")
    
    def get_patches(self) -> Dict[str, str]:
        """Get all collected patches."""
        return self.patches.copy()


async def main():
    """Main function to run SWE-bench processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run R2E agent on SWE-bench-verified instances")
    parser.add_argument("jsonl_file", help="Path to JSONL file with SWE-bench instances")
    parser.add_argument("--output-dir", default="./swe_patches", help="Directory to save patches")
    parser.add_argument("--namespace", default=os.getenv("K8S_NAMESPACE", "default"), 
                       help="Kubernetes namespace (default: from K8S_NAMESPACE env or 'default')")
    parser.add_argument("--kubeconfig", default=os.getenv("KUBECONFIG", None),
                       help="Path to kubeconfig file (default: from KUBECONFIG env)")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting SWE-bench R2E Agent Runner")
    logger.info(f"JSONL file: {args.jsonl_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Namespace: {args.namespace}")
    
    # Create runner
    runner = SWEBenchRunner(
        namespace=args.namespace,
        kubeconfig_path=args.kubeconfig
    )
    
    # Process JSONL file
    await runner.process_jsonl_file(args.jsonl_file, args.output_dir)
    
    # Print summary
    patches = runner.get_patches()
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Processing complete!")
    logger.info(f"Total patches generated: {len(patches)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    # If no arguments provided, show usage example
    if len(sys.argv) == 1:
        print("Usage: python test_r2e_general_agent_on_swe.py <jsonl_file> [options]")
        print("\nExample:")
        print("  python test_r2e_general_agent_on_swe.py swe_bench_verified.jsonl --output-dir ./patches")
        print("\nFor testing with a sample JSONL:")
        print('  echo \'{"instance_id": "test-001", "issue": "Fix import error", "image": "ubuntu:20.04"}\' > test.jsonl')
        print("  python test_r2e_general_agent_on_swe.py test.jsonl")
        sys.exit(1)
    
    asyncio.run(main())