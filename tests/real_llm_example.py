#!/usr/bin/env python3
"""
使用真实LLM API的GeneralAgent示例
获取pod环境内的CPU利用率和内存使用情况

示例展示如何使用LLMAPIClient与各种LLM提供商的API进行交互
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.core import create_tool, create_agent
from workers.agents.general_agent import dump_trajectory
from workers.utils import create_llm_client

# Check if kodo is available for K8S execution
try:
    from kodo import KubernetesManager
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    print("❌ K8S工具不可用，请确保安装了kodo依赖")
    print("   pip install git+https://github.com/baidubce/kodo.git")
    exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API配置
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")

# K8S配置
K8S_POD_NAME = "swebench-xarray-pod"




async def run_k8s_pod_monitoring_task():
    """运行K8S Pod监控任务"""
    print("=== K8S Pod监控任务 - 使用真实LLM API ===")
    
    # 1. 创建LLM客户端
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=True,
        max_retries=3
    )
    
    try:
        # 2. 创建K8S工具（使用统一的执行模式配置）
        print("\n1. 创建K8S工具...")
        k8s_config = {
            "execution_mode": "k8s",
            "pod_name": K8S_POD_NAME,
            "namespace": "default",
            "timeout": 30
        }
        
        tools = {
            "bash_executor": create_tool("BashExecutor", k8s_config.copy()),
            "finish": create_tool("Finish")
        }
        print(f"   创建了 {len(tools)} 个K8S工具: {list(tools.keys())}")
        print(f"   目标Pod: {K8S_POD_NAME}")
        
        # 3. 创建GeneralAgent
        print("\n2. 创建GeneralAgent...")
        agent = create_agent("General", {
            "max_rounds": 8,
            "system_prompt": f"""你是一个Kubernetes系统监控专家。你的任务是获取指定Pod ({K8S_POD_NAME}) 环境内的CPU利用率和内存使用情况。

使用ReAct框架：
1. Thought: 分析需要执行什么命令来获取系统信息
2. Action: 使用K8S工具在Pod内执行相关命令
3. 重复直到获取到完整的系统信息

你可以使用的工具：
- bash_executor: 在Pod内执行bash命令获取系统信息（通过K8S execution_mode）
- finish: 完成任务并提供最终报告

请获取以下信息：
1. CPU使用率 (可以使用top, ps, /proc/stat等)
2. 内存使用情况 (可以使用free, /proc/meminfo等)  
3. 系统负载 (可以使用uptime, /proc/loadavg等)
4. 磁盘使用情况 (可以使用df命令)
5. Pod的基本信息和运行状态

最后提供一个完整的Pod系统状态报告。""",
            "termination_tool_names": ["finish"]
        })
        
        # 4. 配置工具
        agent.set_tools(tools)
        print(f"   GeneralAgent已配置 {len(agent.tools)} 个工具")
        
        # 5. 执行K8S Pod监控任务
        print(f"\n3. 开始执行K8S Pod监控任务...")
        print(f"   目标Pod: {K8S_POD_NAME}")
        print("   (这可能需要几分钟时间，LLM正在分析和执行K8S命令)")
        
        trajectory = await agent.run_trajectory(
            prompt=f"请获取Kubernetes Pod '{K8S_POD_NAME}' 环境内的CPU利用率、内存使用情况、系统负载、磁盘使用情况和Pod基本信息，并生成一个完整的Pod系统状态报告。",
            llm_generate_func=llm_client.generate,
            request_id="k8s_pod_monitoring_001"
        )
        
        # 6. 显示结果
        print(f"\n4. 任务完成!")
        print(f"   轨迹状态: {'完成' if trajectory.is_completed else '未完成'}")
        print(f"   总步数: {len(trajectory.steps)}")
        print(f"   总tokens: {trajectory.total_tokens}")
        
        # 7. 显示执行过程和结果
        print("\n=== 执行过程 ===")
        for i, step in enumerate(trajectory.steps, 1):
            print(f"\nStep {i}: {step.step_type.value.upper()}")
            if len(step.content) > 200:
                print(f"Content: {step.content[:200]}...")
            else:
                print(f"Content: {step.content}")
            
            if step.tool_name:
                print(f"Tool: {step.tool_name}")
                if step.tool_result and isinstance(step.tool_result, dict):
                    if "result" in step.tool_result:
                        result = step.tool_result["result"]
                        if isinstance(result, str) and len(result) > 300:
                            print(f"Result: {result[:300]}...")
                        else:
                            print(f"Result: {result}")
        
        # 8. 获取最终报告
        final_answer = None
        final_reasoning = None
        
        for step in reversed(trajectory.steps):
            if step.tool_name == "finish" and step.tool_result:
                final_answer = step.tool_result.get("answer")
                final_reasoning = step.tool_result.get("reasoning")
                break
        
        if final_answer:
            print(f"\n=== 最终系统状态报告 ===")
            print(final_answer)
            if final_reasoning:
                print(f"\n=== 分析过程 ===")
                print(final_reasoning)
        
        # 9. 保存轨迹
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"tests/k8s_pod_monitoring_trajectory_{timestamp}.json"
        txt_filename = f"tests/k8s_pod_monitoring_trajectory_{timestamp}.txt"
        
        dump_trajectory(trajectory, json_filename, "json")
        dump_trajectory(trajectory, txt_filename, "txt")
        print(f"\n轨迹已保存到 {json_filename} 和 {txt_filename}")
        
        return trajectory
        
    except Exception as e:
        logger.error(f"任务执行失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # 清理资源
        await llm_client.close()


async def test_api_connection():
    """测试API连接"""
    print("=== 测试API连接 ===")
    
    client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=False,
        max_retries=1
    )
    
    try:
        # 简单的测试消息
        test_messages = [
            {"role": "user", "content": "Hello, can you respond with 'API connection successful'?"}
        ]
        
        response = await client.generate(test_messages, max_tokens=50)
        print(f"API测试响应: {response}")
        
        if "successful" in response.lower() or "connection" in response.lower():
            print("✅ API连接测试成功!")
            return True
        else:
            print("⚠️ API连接可能有问题，但收到了响应")
            return True
            
    except Exception as e:
        print(f"❌ API连接测试失败: {e}")
        return False
    
    finally:
        await client.close()


async def main():
    """主函数"""
    print("K8S Pod监控任务 - 使用真实LLM API")
    print("=" * 50)
    
    try:
        # 首先测试API连接
        if not await test_api_connection():
            print("API连接测试失败，无法继续执行任务")
            return
        
        print("\n" + "=" * 50)
        
        # 执行K8S Pod监控任务
        await run_k8s_pod_monitoring_task()
        
        print("\n✅ 所有任务完成!")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("""
    🐳 K8S Pod监控任务说明:
    
    本示例将使用GeneralAgent和真实的LLM API来：
    1. 连接到您提供的LLM API端点
    2. 创建配置了K8S工具的GeneralAgent
    3. 让Agent自动获取指定Pod环境的系统信息：
       - CPU利用率
       - 内存使用情况  
       - 系统负载
       - 磁盘使用情况
       - Pod基本信息和运行状态
    4. 生成完整的Pod系统状态报告
    
    Agent将使用ReAct框架，通过K8S工具在Pod内执行命令获取信息。
    
    API配置:
    - Model: claude-sonnet-4-20250514
    - Base URL: http://211.23.3.237:27544/
    - 超时: 60秒
    
    K8S配置:
    - Pod Name: swebench-xarray-pod
    """)
    
    # 检查required的openai库
    try:
        import openai
    except ImportError:
        print("❌ 需要安装openai库: pip install openai")
        sys.exit(1)
    
    asyncio.run(main())