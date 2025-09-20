#!/usr/bin/env python3
"""
Simplified test for GeneralAgent with R2E tools for SWE-bench processing using subprocess.
Synchronous version to avoid asyncio issues.
"""

import sys
import os
import json
import time
import subprocess
import pickle
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import tempfile
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LLM configuration from environment
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")


class ToolLogger:
    """Simple wrapper to log tool executions (synchronous version)."""
    
    def __init__(self, tool, tool_name):
        self.tool = tool
        self.tool_name = tool_name
        self.execution_count = 0
    
    def execute_tool(self, instance_id, tool_args):
        """Synchronous tool execution with logging."""
        self.execution_count += 1
        start_time = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TOOL #{self.execution_count}: {self.tool_name}")
        logger.info(f"Instance: {instance_id}")
        
        # Log arguments (truncated for readability)
        for key, value in tool_args.items():
            if isinstance(value, str) and len(value) > 1000:
                logger.info(f"  {key}: [String with {len(value)} chars]")
            else:
                logger.info(f"  {key}: {value}")
        
        try:
            # Convert async to sync if needed
            import asyncio
            import inspect
            
            if inspect.iscoroutinefunction(self.tool.execute_tool):
                # If tool is async, run it in new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.tool.execute_tool(instance_id, tool_args)
                    )
                finally:
                    loop.close()
            else:
                # Tool is already sync
                result = self.tool.execute_tool(instance_id, tool_args)
            
            execution_time = time.time() - start_time
            logger.info(f"Status: SUCCESS ({execution_time:.2f}s)")
            logger.info(f"{'='*60}\n")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Status: FAILED ({execution_time:.2f}s)")
            logger.error(f"Error: {e}")
            logger.error(f"{'='*60}\n")
            raise
    
    def __getattr__(self, name):
        return getattr(self.tool, name)


class PodManager:
    """Manages Kubernetes pod lifecycle (synchronous version)."""
    
    def __init__(self, kodo_runner, namespace, kubeconfig_path):
        self.kodo_runner = kodo_runner
        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path
        self.pod = None
        self.pod_name = None
    
    def start_pod(self, instance_id: str, image: str) -> Any:
        """Start a pod with retry logic (synchronous)."""
        import uuid
        
        # Generate unique pod name
        unique_suffix = str(uuid.uuid4())[:8]
        instance_part = instance_id.lower().replace('/', '-').replace('_', '-')[:35]
        self.pod_name = f"swe-{instance_part}-{unique_suffix}"
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting pod: {self.pod_name} (attempt {attempt + 1}/{max_retries})")
                
                self.pod = self.kodo_runner.start_container(
                    image,
                    name=self.pod_name,
                    environment={
                        "PYTHONPATH": "/testbed",
                        "SWE_INSTANCE_ID": instance_id,
                        "PYTHONIOENCODING": "utf-8",
                        "LANG": "C.UTF-8",
                        "LC_ALL": "C.UTF-8",
                        "http_proxy": "http://agent.baidu.com:8891",
                        "https_proxy": "http://agent.baidu.com:8891",
                        "PIP_INDEX_URL": "http://pip.baidu.com/pypi/simple",
                        "PIP_TRUSTED_HOST": "pip.baidu.com"
                    }
                )
                
                logger.info(f"Pod {self.pod_name} started successfully")
                
                # Setup environment
                time.sleep(3)  # Wait for pod to be ready
                self.kodo_runner.execute_command(self.pod, "ln -s /opt/miniconda3/envs/testbed /root/.venv")
                
                return self.pod
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
    
    def cleanup(self):
        """Clean up the pod."""
        if self.pod and self.kodo_runner:
            try:
                logger.info(f"Stopping pod: {self.pod_name}")
                self.kodo_runner.stop_container(self.pod)
                logger.info(f"Pod {self.pod_name} stopped")
            except Exception as e:
                logger.error(f"Error stopping pod: {e}")


def create_llm_function(model_name: str, base_url: str, api_key: str, max_tokens: int = 16000):
    """Create a synchronous LLM function wrapper."""
    from workers.utils.llm_helper import call_llm
    
    llm_call_count = 0
    
    # Set environment variables for the LLM
    os.environ['LLM_BASE_URL'] = base_url
    os.environ['LLM_API_KEY'] = api_key
    
    # Save proxy settings
    saved_http_proxy = os.environ.get('http_proxy')
    saved_https_proxy = os.environ.get('https_proxy')
    
    def llm_generate_func(messages, **kwargs):
        nonlocal llm_call_count
        llm_call_count += 1
        
        # Restore proxy settings
        if saved_http_proxy:
            os.environ['http_proxy'] = saved_http_proxy
        if saved_https_proxy:
            os.environ['https_proxy'] = saved_https_proxy
        
        # Set defaults
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = max_tokens
        
        logger.info(f"\nLLM CALL #{llm_call_count}")
        logger.info(f"Model: {model_name}, Max Tokens: {kwargs.get('max_tokens')}")
        
        try:
            # Synchronous call_llm
            response = call_llm(
                messages=messages,
                model=model_name,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.95),
                max_tokens=kwargs.get('max_tokens', max_tokens),
                timeout=60
            )
            
            logger.info(f"LLM Response: SUCCESS")
            return response
            
        except Exception as e:
            logger.error(f"LLM Response: FAILED - {e}")
            raise
    
    return llm_generate_func


def generate_patch(kodo_runner, pod, base_commit: Optional[str] = None) -> Optional[str]:
    """Generate git patch from changes."""
    logger.info("\nGenerating patch...")
    
    # Add all changes
    kodo_runner.execute_command(pod, "cd /testbed && git add -A")
    
    # Generate patch
    if base_commit:
        logger.info(f"Generating patch against base_commit: {base_commit}")
        output, exit_code = kodo_runner.execute_command(
            pod, f"cd /testbed && git diff {base_commit}"
        )
    else:
        logger.info("Generating patch against staged changes")
        output, exit_code = kodo_runner.execute_command(
            pod, "cd /testbed && git diff --cached"
        )
    
    if int(exit_code) == 0:
        patch = output.strip()
        if patch:
            logger.info(f"Patch generated: {len(patch)} characters")
            return patch
    
    logger.warning("No patch generated")
    return None


def process_single_instance(instance_data_file: str, output_file: str, 
                           model_name: str = None, log_file: str = None, 
                           base_url: str = None) -> None:
    """
    Process a single SWE-bench instance (synchronous version).
    
    This function is called by subprocess.run() for concurrent execution.
    """
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    # Import required modules
    from workers.agents.general_agent_sync import GeneralAgentSync, dump_trajectory
    from workers.core import create_tool
    from workers.core.safe_profiler import SafeProfiler
    
    try:
        from kodo import ContainerRunner
    except ImportError:
        print("ERROR: kodo package not found. Please install it with: pip install kodo")
        sys.exit(1)
    
    from workers.tools.r2e_configs import (
        CUSTOM_TOOL_DESCRIPTIONS,
        parse_xml_action_custom,
        CustomDescriptionWrapper,
        generate_custom_system_prompt
    )
    
    # Configure logging for subprocess
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode='w')
            ],
            force=True
        )
    
    logger = logging.getLogger(__name__)
    
    # Load instance data
    with open(instance_data_file, 'rb') as f:
        data = pickle.load(f)
    
    instance = data['instance']
    namespace = data['namespace']
    kubeconfig_path = data['kubeconfig_path']
    output_dir = data['output_dir']
    enable_profiling = data['enable_profiling']
    max_tokens = data.get('max_tokens', 16000)
    provided_base_url = data.get('base_url', None)
    
    instance_id = instance.get("instance_id", "unknown")
    issue = instance.get("problem_statement", "")
    image = instance.get("image", "")
    base_commit = instance.get("base_commit", None)
    
    # Configure model and base URL
    actual_model_name = model_name or MODEL_NAME
    actual_base_url = base_url or provided_base_url or BASE_URL
    
    logger.info(f"Processing {instance_id} with {actual_model_name}")
    
    # Set environment variables for LLM
    os.environ['LLM_API_KEY'] = API_KEY
    os.environ['LLM_BASE_URL'] = actual_base_url
    os.environ['LLM_MODEL_NAME'] = actual_model_name
    
    def run_instance():
        """Synchronous function to process the instance."""
        
        # Initialize Kubernetes runner
        kodo_runner = ContainerRunner(
            backend="kubernetes",
            namespace=namespace,
            kubeconfig_path=kubeconfig_path
        )
        
        # Create pod manager
        pod_manager = PodManager(kodo_runner, namespace, kubeconfig_path)
        
        try:
            # Start pod
            pod = pod_manager.start_pod(instance_id, image)
            
            # Create R2E tools
            working_dir = "/testbed"
            k8s_config = {
                "execution_mode": "k8s",
                "pod_name": pod_manager.pod_name,
                "namespace": namespace,
                "kubeconfig_path": kubeconfig_path,
                "working_dir": working_dir
            }
            
            base_tools = {
                "r2e_bash_executor": create_tool("R2EBashExecutor", k8s_config.copy()),
                "r2e_file_editor": create_tool("R2EFileEditor", k8s_config.copy()),
                "r2e_search": create_tool("R2ESearch", k8s_config.copy()),
                "r2e_submit": create_tool("R2ESubmit", {})
            }
            
            # Wrap tools with logging and custom descriptions
            tools = {}
            for tool_name, tool in base_tools.items():
                logged_tool = ToolLogger(tool, tool_name)
                if tool_name in CUSTOM_TOOL_DESCRIPTIONS:
                    tools[tool_name] = CustomDescriptionWrapper(
                        logged_tool, CUSTOM_TOOL_DESCRIPTIONS[tool_name]
                    )
                else:
                    tools[tool_name] = logged_tool
            
            # Generate system prompt
            custom_system_prompt = generate_custom_system_prompt(
                tools,
                task_description="analyze and fix the reported issue in the repository",
                working_directory="/testbed",
                additional_instructions=(
                    "\n- Focus on the specific issue described"
                    "\n- Make minimal changes to fix the issue"
                    "\n- Ensure your changes don't break existing functionality"
                )
            )
            
            # Create profiler if enabled
            profiler = SafeProfiler(enabled=enable_profiling) if enable_profiling else None
            
            # Create agent (using sync version)
            agent = GeneralAgentSync(
                max_rounds=50,
                debug=True,
                termination_tool_names=["r2e_submit"],
                action_parser=parse_xml_action_custom,
                system_prompt=custom_system_prompt,
                profiler=profiler
            )
            agent.set_tools(tools)
            
            # Create LLM function (synchronous)
            llm_func = create_llm_function(
                actual_model_name, actual_base_url, API_KEY, max_tokens
            )
            
            # Test LLM connection before starting
            logger.info("Testing LLM connection...")
            try:
                test_response = llm_func([
                    {"role": "user", "content": "给我讲个笑话"}
                ])
                logger.info(f"LLM connection test successful. Response preview: {test_response[:100]}...")
            except Exception as e:
                logger.error(f"LLM connection test failed: {e}")
                raise RuntimeError(f"Failed to connect to LLM service: {e}")
            
            # Create prompt
            prompt = f"""Consider the following github issue:
<github_issue>            
{issue}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I've already taken care of all changes to any of the test files described in the <github_issue>. 
This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.

Follow these steps:
1. Explore the repo to familiarize yourself with its structure
2. Create a script to reproduce the error and confirm it
3. Edit the sourcecode to resolve the issue
4. Rerun your reproduce script to confirm the fix
5. Think about edge cases and ensure your fix handles them
6. When viewing large files, use specific line-ranges (50-100 lines)
7. The repository is at '/testbed' and that's the current working directory
"""
            
            # Run agent (now fully synchronous)
            logger.info("Running agent...")
            
            result = agent.run_trajectory(
                prompt=prompt,
                llm_generate_func=llm_func,
                request_id=f"swe_{instance_id}"
            )
            
            logger.info(f"Agent completed: {result.is_completed}")
            logger.info(f"Total steps: {len(result.steps)}")
            
            # Save trajectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trajectory_dir = os.path.join(output_dir, "trajectories")
            os.makedirs(trajectory_dir, exist_ok=True)
            
            file_safe_id = instance_id.replace('/', '__')
            trajectory_file = os.path.join(
                trajectory_dir, f"{file_safe_id}_{timestamp}.jsonl"
            )
            dump_trajectory(result, trajectory_file, format="jsonl")
            logger.info(f"Saved trajectory to: {trajectory_file}")
            
            # Save profiler data if enabled
            profiler_file = None
            if profiler and profiler.enabled:
                try:
                    profiler_dir = os.path.join(output_dir, "profiles")
                    os.makedirs(profiler_dir, exist_ok=True)
                    profiler_file = os.path.join(
                        profiler_dir, f"{file_safe_id}_{timestamp}_profile.json"
                    )
                    profiler.export_events(profiler_file)
                    logger.info(f"Saved profiler data to: {profiler_file}")
                except Exception as e:
                    logger.error(f"Failed to save profiler data: {e}")
            
            # Generate patch
            patch = generate_patch(kodo_runner, pod, base_commit)
            
            if patch:
                # Save patch
                patches_dir = os.path.join(output_dir, "patches")
                os.makedirs(patches_dir, exist_ok=True)
                patch_file = os.path.join(
                    patches_dir, f"{file_safe_id}_{timestamp}.patch"
                )
                
                with open(patch_file, 'w', encoding='utf-8') as f:
                    f.write(patch)
                logger.info(f"Saved patch to: {patch_file}")
                
                # Save result
                result_data = {
                    'success': True,
                    'instance_id': instance_id,
                    'patch': patch,
                    'patch_file': patch_file,
                    'trajectory_file': trajectory_file
                }
                
                if profiler_file:
                    result_data['profiler_file'] = profiler_file
                
            else:
                result_data = {
                    'success': False,
                    'instance_id': instance_id,
                    'error': 'No patch generated',
                    'trajectory_file': trajectory_file
                }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f)
            
            return patch
            
        except Exception as e:
            logger.error(f"Error processing {instance_id}: {e}")
            logger.error(traceback.format_exc())
            
            result_data = {
                'success': False,
                'instance_id': instance_id,
                'error': str(e),
                'error_details': {
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f)
            
            return None
            
        finally:
            pod_manager.cleanup()
    
    # Run the synchronous function
    run_instance()


class SubprocessSWEBenchRunner:
    """Runner for SWE-bench instances using subprocess for concurrency."""
    
    def __init__(self, namespace: str = "default", kubeconfig_path: Optional[str] = None,
                 output_dir: str = "./swe_patches", max_concurrent: int = 1,
                 enable_profiling: bool = False, model_name: str = None,
                 model_index_range: Optional[Tuple[int, int]] = None,
                 timeout_seconds: int = 600, max_tokens: int = 16000,
                 base_url: str = None, local_mode: bool = False):
        """Initialize the runner."""
        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        self.enable_profiling = enable_profiling
        self.model_name = model_name or MODEL_NAME
        self.model_index_range = model_index_range
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.local_mode = local_mode
        self.patches = {}
    
    def process_instances(self, instances: List[Dict[str, Any]]):
        """Process multiple instances concurrently using ThreadPoolExecutor."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not self.local_mode:
            logs_dir = os.path.join(self.output_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            logger.info(f"Logs directory: {logs_dir}")
        
        temp_dir = tempfile.mkdtemp(prefix="swe_bench_")
        
        logger.info(f"Processing {len(instances)} instances")
        logger.info(f"Max concurrency: {self.max_concurrent}")
        
        if self.model_index_range:
            logger.info(f"Model range: {self.model_name}-{self.model_index_range[0]} "
                       f"to {self.model_name}-{self.model_index_range[1]}")
        
        # Prepare all tasks
        tasks = []
        for i, instance in enumerate(instances):
            # Determine model name for load balancing
            if self.model_index_range:
                start_idx, end_idx = self.model_index_range
                model_range = end_idx - start_idx + 1
                model_index = (i % model_range) + start_idx
                actual_model_name = f"{self.model_name}-{model_index}"
            else:
                actual_model_name = self.model_name
            
            # Prepare instance data
            instance_data = {
                'instance': instance,
                'namespace': self.namespace,
                'kubeconfig_path': self.kubeconfig_path,
                'output_dir': self.output_dir,
                'enable_profiling': self.enable_profiling,
                'max_tokens': self.max_tokens,
                'base_url': self.base_url
            }
            
            # Save to temp file
            instance_data_file = os.path.join(temp_dir, f"instance_{i}.pkl")
            with open(instance_data_file, 'wb') as f:
                pickle.dump(instance_data, f)
            
            output_file = os.path.join(temp_dir, f"result_{i}.json")
            
            # Prepare task parameters
            instance_id = instance.get('instance_id', f'instance_{i}')
            if self.local_mode:
                log_file = None
            else:
                log_file = os.path.join(
                    logs_dir, 
                    f"{instance_id.replace('/', '__')}.log"
                )
            
            tasks.append({
                'index': i,
                'total': len(instances),
                'instance_id': instance_id,
                'instance_data_file': instance_data_file,
                'output_file': output_file,
                'model_name': actual_model_name,
                'log_file': log_file
            })
        
        # Execute with ThreadPoolExecutor
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all tasks
            if self.local_mode:
                futures = {
                    executor.submit(
                        self._run_local_instance,
                        task['index'], task['total'], task['instance_data_file'],
                        task['output_file'], task['model_name']
                    ): task for task in tasks
                }
            else:
                futures = {
                    executor.submit(
                        self._run_subprocess,
                        task['index'], task['total'], task['instance_data_file'],
                        task['output_file'], task['model_name'], task['log_file']
                    ): task for task in tasks
                }
            
            # Process completed futures
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task for {task['instance_id']} failed: {e}")
                    results.append({
                        'success': False,
                        'instance_id': task['instance_id'],
                        'error': str(e)
                    })
        
        elapsed_time = time.time() - start_time
        
        # Process results
        successful = sum(1 for r in results 
                        if isinstance(r, dict) and r.get('success'))
        failed = len(results) - successful
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total: {len(instances)} instances")
        logger.info(f"Success: {successful} ({successful/len(instances)*100:.1f}%)")
        logger.info(f"Failed: {failed} ({failed/len(instances)*100:.1f}%)")
        logger.info(f"Time: {elapsed_time:.2f}s")
        logger.info(f"Throughput: {len(instances)/elapsed_time:.3f} instances/sec")
        
        # Save summary
        summary = {
            "total_instances": len(instances),
            "successful_patches": successful,
            "failed_instances": failed,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = os.path.join(self.output_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def _run_subprocess(self, index: int, total: int, instance_data_file: str,
                       output_file: str, model_name: str, log_file: str) -> Optional[Dict]:
        """Run instance in subprocess."""
        with open(instance_data_file, 'rb') as f:
            data = pickle.load(f)
        instance_id = data['instance'].get('instance_id', f'instance_{index}')
        
        logger.info(f"[{index+1}/{total}] Starting {instance_id} with {model_name}")
        
        # Prepare command
        cmd = [
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, '{os.path.dirname(os.path.dirname(__file__))}')
from tests.test_r2e_general_agent_on_swe_subprocess_simple import process_single_instance
process_single_instance('{instance_data_file}', '{output_file}', '{model_name}', '{log_file}', '{self.base_url or ""}')
"""
        ]
        
        # Run subprocess with timeout
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            
            if result.returncode != 0:
                logger.error(f"[{index+1}/{total}] {instance_id} failed: return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr[:1000]}")
                return {
                    'success': False,
                    'instance_id': instance_id,
                    'error': f'Process failed with return code {result.returncode}'
                }
            
        except subprocess.TimeoutExpired:
            logger.error(f"[{index+1}/{total}] {instance_id} timeout after {self.timeout_seconds}s")
            return {
                'success': False,
                'instance_id': instance_id,
                'error': f'Timeout after {self.timeout_seconds} seconds',
                'timeout': True
            }
        
        # Read result
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                result = json.load(f)
            logger.info(f"[{index+1}/{total}] {instance_id} completed: {result.get('success')}")
            return result
        
        return {
            'success': False,
            'instance_id': instance_id,
            'error': 'No output file generated'
        }
    
    def _run_local_instance(self, index: int, total: int, instance_data_file: str,
                           output_file: str, model_name: str) -> Optional[Dict]:
        """Run instance locally for debugging."""
        with open(instance_data_file, 'rb') as f:
            data = pickle.load(f)
        instance_id = data['instance'].get('instance_id', f'instance_{index}')
        
        logger.info(f"[{index+1}/{total}] Local execution: {instance_id}")
        
        try:
            process_single_instance(
                instance_data_file, output_file, model_name, None, self.base_url
            )
            
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    return json.load(f)
            
            return {
                'success': False,
                'instance_id': instance_id,
                'error': 'No output file generated'
            }
            
        except Exception as e:
            logger.error(f"[{index+1}/{total}] {instance_id} failed: {e}")
            return {
                'success': False,
                'instance_id': instance_id,
                'error': str(e)
            }


def main():
    """Main entry point (synchronous)."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run R2E agent on SWE-bench instances"
    )
    parser.add_argument("jsonl_file", help="Path to JSONL file")
    parser.add_argument("--output-dir", default="./swe_patches")
    parser.add_argument("--namespace", default=os.getenv("K8S_NAMESPACE", "default"))
    parser.add_argument("--kubeconfig", default=os.getenv("KUBECONFIG"))
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-index-range", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--local-mode", action="store_true")
    
    args = parser.parse_args()
    
    # Parse model index range
    model_index_range = None
    if args.model_index_range:
        parts = args.model_index_range.split(',')
        if len(parts) == 2:
            model_index_range = (int(parts[0]), int(parts[1]))
    
    # Load instances
    instances = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    
    logger.info(f"Loaded {len(instances)} instances")
    
    # Create and run
    runner = SubprocessSWEBenchRunner(
        namespace=args.namespace,
        kubeconfig_path=args.kubeconfig,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        enable_profiling=args.enable_profiling,
        model_name=args.model_name,
        model_index_range=model_index_range,
        timeout_seconds=args.timeout,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        local_mode=args.local_mode
    )
    
    runner.process_instances(instances)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python script.py <jsonl_file> [options]")
        print("\nExample:")
        print("  python script.py swe_bench.jsonl --max-concurrent 20")
        sys.exit(1)
    
    main()
