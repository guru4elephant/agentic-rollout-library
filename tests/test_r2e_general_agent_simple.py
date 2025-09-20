#!/usr/bin/env python3
"""
简化版的 SWE-bench 测试脚本 - 使用同步代码，支持多进程并发
"""

import os
import sys
import json
import time
import pickle
import logging
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(process)d] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LLM 配置
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "http://211.23.3.237:27544")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-sonnet-4-20250514")


def process_instance(instance_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个 SWE-bench 实例（在子进程中运行）
    
    Args:
        instance_data: 包含实例信息的字典
        
    Returns:
        处理结果字典
    """
    instance_id = instance_data.get('instance_id', 'unknown')
    issue = instance_data.get('issue', '')
    
    # 设置日志
    logger.info(f"Processing instance: {instance_id}")
    start_time = time.time()
    
    # 添加项目路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    try:
        # 导入必要的模块
        from workers.agents.general_agent import GeneralAgent
        from workers.core import create_tool
        from workers.utils.llm_helper import call_llm
        
        # 创建工作目录
        working_dir = f"/tmp/swe_{instance_id}_{int(time.time())}"
        os.makedirs(working_dir, exist_ok=True)
        logger.info(f"Working directory: {working_dir}")
        
        # 创建 R2E 工具
        tools = {}
        execution_mode = instance_data.get('execution_mode', 'local')
        
        if execution_mode == 'k8s':
            # K8s 模式
            pod_name = instance_data.get('pod_name')
            logger.info(f"Using K8s mode with pod: {pod_name}")
            
            tools['bash'] = create_tool(
                "R2EBashExecutor",
                {"execution_mode": "k8s", "pod_name": pod_name}
            )
            tools['editor'] = create_tool(
                "R2EEditor", 
                {"execution_mode": "k8s", "pod_name": pod_name}
            )
            tools['search'] = create_tool(
                "R2ESearcher",
                {"execution_mode": "k8s", "pod_name": pod_name}
            )
        else:
            # 本地模式
            logger.info("Using local execution mode")
            
            tools['bash'] = create_tool(
                "R2EBashExecutor",
                {"execution_mode": "local", "working_dir": working_dir}
            )
            tools['editor'] = create_tool(
                "R2EEditor",
                {"execution_mode": "local", "working_dir": working_dir}
            )
            tools['search'] = create_tool(
                "R2ESearcher",
                {"execution_mode": "local", "working_dir": working_dir}
            )
        
        # 添加提交工具
        tools['submit'] = create_tool("R2ESubmit", {})
        
        # 创建 Agent
        agent = GeneralAgent(
            name=f"SWE_Agent_{instance_id}",
            description=f"Agent for solving {instance_id}",
            system_prompt="You are a software engineer fixing GitHub issues.",
            max_rounds=30,
            max_steps=60,
            use_xml_format=True
        )
        agent.set_tools(tools)
        
        # 准备 LLM 调用函数（同步版本）
        def llm_generate(messages, **kwargs):
            """同步的 LLM 生成函数"""
            # 设置环境变量
            os.environ['LLM_API_KEY'] = API_KEY
            os.environ['LLM_BASE_URL'] = BASE_URL
            os.environ['LLM_MODEL_NAME'] = MODEL_NAME
            
            # 调用 LLM
            return call_llm(
                messages=messages,
                model=MODEL_NAME,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 4000),
                timeout=60
            )
        
        # 构建提示词
        prompt = f"""
Please analyze and fix the following GitHub issue:

<issue>
{issue}
</issue>

The repository is located at /testbed. 
Use the provided tools to explore the code, understand the issue, and create a fix.
When you have a working solution, use the submit tool to provide the final patch.
"""
        
        # 运行 Agent（同步版本 - 需要修改 GeneralAgent 支持同步）
        # 这里我们简化处理，直接调用 LLM 并执行工具
        logger.info(f"Starting to solve issue for {instance_id}")
        
        # 简化的执行流程
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        patch = None
        for round_num in range(5):  # 简化为5轮
            logger.info(f"Round {round_num + 1}/5")
            
            # 调用 LLM
            try:
                response = llm_generate(messages, max_tokens=2000)
                messages.append({"role": "assistant", "content": response})
                
                # 检查是否有提交
                if '<submit>' in response.lower():
                    # 提取 patch
                    import re
                    patch_match = re.search(r'<submit>(.*?)</submit>', response, re.DOTALL)
                    if patch_match:
                        patch = patch_match.group(1).strip()
                        logger.info("Found patch submission")
                        break
                
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                break
        
        # 处理结果
        elapsed_time = time.time() - start_time
        
        result = {
            'instance_id': instance_id,
            'success': patch is not None,
            'patch': patch or '',
            'elapsed_time': elapsed_time,
            'error': None
        }
        
        logger.info(f"Completed {instance_id} in {elapsed_time:.2f}s - Success: {result['success']}")
        
    except Exception as e:
        logger.error(f"Failed to process {instance_id}: {e}")
        result = {
            'instance_id': instance_id,
            'success': False,
            'patch': '',
            'elapsed_time': time.time() - start_time,
            'error': str(e)
        }
    
    return result


def run_parallel_test(
    instances_file: str,
    output_dir: str,
    max_workers: int = 3,
    max_instances: int = None,
    execution_mode: str = 'local'
):
    """
    并行运行多个实例的测试
    
    Args:
        instances_file: JSONL 文件路径
        output_dir: 输出目录
        max_workers: 最大并发数
        max_instances: 最大处理实例数
        execution_mode: 执行模式 ('local' 或 'k8s')
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 读取实例
    instances = []
    with open(instances_file, 'r') as f:
        for line in f:
            if line.strip():
                instance = json.loads(line)
                instance['execution_mode'] = execution_mode
                instances.append(instance)
                
                if max_instances and len(instances) >= max_instances:
                    break
    
    logger.info(f"Loaded {len(instances)} instances")
    
    # 使用进程池并行处理
    results = []
    failed_count = 0
    success_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_instance = {
            executor.submit(process_instance, instance): instance
            for instance in instances
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_instance):
            instance = future_to_instance[future]
            instance_id = instance.get('instance_id', 'unknown')
            
            try:
                result = future.result(timeout=600)  # 10分钟超时
                results.append(result)
                
                if result['success']:
                    success_count += 1
                    # 保存 patch
                    patch_file = os.path.join(
                        output_dir,
                        f"{instance_id}_{timestamp}.patch"
                    )
                    with open(patch_file, 'w') as f:
                        f.write(result['patch'])
                    logger.info(f"✅ {instance_id}: Saved patch to {patch_file}")
                else:
                    failed_count += 1
                    logger.warning(f"❌ {instance_id}: No patch generated")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ {instance_id}: Process failed - {e}")
                results.append({
                    'instance_id': instance_id,
                    'success': False,
                    'error': str(e)
                })
    
    # 保存结果汇总
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    summary = {
        'timestamp': timestamp,
        'total': len(instances),
        'success': success_count,
        'failed': failed_count,
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"总实例数: {len(instances)}")
    print(f"成功: {success_count} ({success_count/len(instances)*100:.1f}%)")
    print(f"失败: {failed_count} ({failed_count/len(instances)*100:.1f}%)")
    print(f"结果保存在: {output_dir}")
    print(f"汇总文件: {summary_file}")
    
    return summary


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='简化版 SWE-bench 测试')
    parser.add_argument('instances_file', help='JSONL 实例文件')
    parser.add_argument('--output-dir', default='outputs', help='输出目录')
    parser.add_argument('--max-workers', type=int, default=3, help='最大并发数')
    parser.add_argument('--max-instances', type=int, help='最大处理实例数')
    parser.add_argument('--execution-mode', choices=['local', 'k8s'], 
                       default='local', help='执行模式')
    
    args = parser.parse_args()
    
    # 运行测试
    run_parallel_test(
        instances_file=args.instances_file,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        max_instances=args.max_instances,
        execution_mode=args.execution_mode
    )


if __name__ == "__main__":
    main()
