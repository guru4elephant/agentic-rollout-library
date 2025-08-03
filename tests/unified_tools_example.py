#!/usr/bin/env python3
"""
统一工具接口示例
展示如何通过execution_mode配置在本地和K8S环境之间切换，
而不需要修改Agent代码或工具名称。
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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API配置
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
#MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")

# K8S配置
K8S_POD_NAME = "swebench-xarray-pod"


def create_tools_with_mode(execution_mode: str = "local", pod_name: str = None):
    """
    创建工具集合，通过execution_mode参数控制执行环境
    
    Args:
        execution_mode: "local" 或 "k8s"
        pod_name: K8S pod名称（当execution_mode="k8s"时需要）
    
    Returns:
        工具字典
    """
    print(f"🔧 创建工具 (执行模式: {execution_mode})")
    
    # 基础工具配置
    base_configs = {
        "timeout": 30,
        "execution_mode": execution_mode
    }
    
    # 如果是K8S模式，添加pod配置
    if execution_mode == "k8s":
        if not pod_name:
            raise ValueError("K8S模式需要提供pod_name")
        base_configs.update({
            "pod_name": pod_name,
            "namespace": "default"
        })
        print(f"   K8S目标Pod: {pod_name}")
    
    # 创建工具（工具名称保持不变）
    tools = {
        "bash_executor": create_tool("BashExecutor", base_configs.copy()),
        # "file_editor": create_tool("FileEditor", base_configs.copy()),  # 暂时注释，等待重构完成
        # "search": create_tool("Search", base_configs.copy()),  # 暂时注释，等待重构完成
        "finish": create_tool("Finish")
    }
    
    print(f"   创建了 {len(tools)} 个工具: {list(tools.keys())}")
    return tools


async def run_local_task():
    """运行本地任务示例"""
    print("\n=== 本地执行模式示例 ===")
    
    # 1. 创建LLM客户端
    llm_client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True)
    
    try:
        # 2. 创建本地工具
        tools = create_tools_with_mode(execution_mode="local")
        
        # 3. 创建GeneralAgent
        agent = create_agent("General", {
            "max_rounds": 20,
            "system_prompt": """你是一个系统信息收集专家。请获取本地系统的基本信息：

使用ReAct框架：
1. Thought: 分析需要执行什么命令
2. Action: 使用bash_executor工具执行命令
3. 重复直到获取完整信息

请获取：
1. 当前目录内容 (ls -la)
2. 系统信息 (uname -a)
3. 当前用户 (whoami)
4. Python版本 (python3 --version)

最后使用finish工具提供总结报告。""",
            "termination_tool_names": ["finish"]
        })
        
        # 4. 配置工具并执行
        agent.set_tools(tools)
        print(f"   GeneralAgent已配置 {len(agent.tools)} 个工具")
        
        trajectory = await agent.run_trajectory(
            prompt="请获取本地系统的基本信息并生成报告。",
            llm_generate_func=llm_client.generate,
            request_id="local_system_info_001"
        )
        
        # 5. 显示结果
        print(f"\n✅ 本地任务完成!")
        print(f"   轨迹状态: {'完成' if trajectory.is_completed else '未完成'}")
        print(f"   总步数: {len(trajectory.steps)}")
        
        # 获取最终报告
        final_answer = None
        for step in reversed(trajectory.steps):
            if step.tool_name == "finish" and step.tool_result:
                final_answer = step.tool_result.get("answer")
                break
        
        if final_answer:
            print(f"\n=== 本地系统信息报告 ===")
            print(final_answer)
        
        # 6. 保存轨迹文件
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"tests/local_system_info_trajectory_{timestamp}.json"
        txt_filename = f"tests/local_system_info_trajectory_{timestamp}.txt"
        
        dump_trajectory(trajectory, json_filename, "json")
        dump_trajectory(trajectory, txt_filename, "txt")
        print(f"\n💾 轨迹已保存到 {json_filename} 和 {txt_filename}")
        
        return trajectory
        
    finally:
        await llm_client.close()


async def run_k8s_task():
    """运行K8S任务示例"""
    print("\n=== K8S执行模式示例 ===")
    
    # 1. 创建LLM客户端
    llm_client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True)
    
    try:
        # 2. 创建K8S工具
        tools = create_tools_with_mode(execution_mode="k8s", pod_name=K8S_POD_NAME)
        
        # 3. 创建GeneralAgent（相同的Agent代码）
        agent = create_agent("General", {
            "max_rounds": 5,
            "system_prompt": f"""你是一个Kubernetes系统监控专家。请获取Pod {K8S_POD_NAME} 的基本信息：

使用ReAct框架：
1. Thought: 分析需要执行什么命令
2. Action: 使用bash_executor工具在Pod内执行命令
3. 重复直到获取完整信息

请获取：
1. Pod内当前目录内容 (ls -la)
2. Pod系统信息 (uname -a)
3. Pod内当前用户 (whoami)
4. Pod内Python版本 (python3 --version)
5. Pod内存信息 (free -h)

最后使用finish工具提供Pod状态总结报告。""",
            "termination_tool_names": ["finish"]
        })
        
        # 4. 配置工具并执行（相同的接口）
        agent.set_tools(tools)
        print(f"   GeneralAgent已配置 {len(agent.tools)} 个工具")
        
        trajectory = await agent.run_trajectory(
            prompt=f"请获取本地系统的基本信息并生成报告。",
            llm_generate_func=llm_client.generate,
            request_id="k8s_system_info_001"
        )
        
        # 5. 显示结果
        print(f"\n✅ K8S任务完成!")
        print(f"   轨迹状态: {'完成' if trajectory.is_completed else '未完成'}")
        print(f"   总步数: {len(trajectory.steps)}")
        
        # 获取最终报告
        final_answer = None
        for step in reversed(trajectory.steps):
            if step.tool_name == "finish" and step.tool_result:
                final_answer = step.tool_result.get("answer")
                break
        
        if final_answer:
            print(f"\n=== K8S Pod信息报告 ===")
            print(final_answer)
        
        # 6. 保存轨迹文件
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"tests/k8s_system_info_trajectory_{timestamp}.json"
        txt_filename = f"tests/k8s_system_info_trajectory_{timestamp}.txt"
        
        dump_trajectory(trajectory, json_filename, "json")
        dump_trajectory(trajectory, txt_filename, "txt")
        print(f"\n💾 轨迹已保存到 {json_filename} 和 {txt_filename}")
        
        return trajectory
        
    finally:
        await llm_client.close()


async def main():
    """主函数 - 演示统一工具接口的强大之处"""
    print("🚀 统一工具接口示例")
    print("=" * 60)
    print("""
📋 本示例展示：
1. 相同的Agent代码可以在本地和K8S环境执行
2. 通过execution_mode配置切换执行环境
3. Agent无需知道工具的底层实现细节
4. 工具名称和接口保持一致

API配置:
- Model: claude-sonnet-4-20250514
- Base URL: http://211.23.3.237:27544/

执行环境:
- 本地模式: execution_mode="local"
- K8S模式: execution_mode="k8s", pod_name="swebench-xarray-pod"
    """)
    
    try:
        # 运行本地任务
        local_trajectory = await run_local_task()
        
        print("\n" + "=" * 60)
        
        # 运行K8S任务
        k8s_trajectory = await run_k8s_task()
        
        print("\n" + "=" * 60)
        print("🎉 演示完成！")
        print(f"✅ 本地任务: {len(local_trajectory.steps)} 步")
        print(f"✅ K8S任务: {len(k8s_trajectory.steps)} 步")
        print(f"💾 轨迹文件: 每个任务都保存了完整的JSON和TXT格式轨迹文件")
        print("""
💡 关键优势：
- Agent代码完全相同
- 工具接口统一
- 配置驱动的执行环境切换
- 对Agent透明的底层实现
- 自动保存完整的执行轨迹（不截断）
        """)
        
    except Exception as e:
        logger.error(f"演示执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
🔧 统一工具接口架构说明:

传统方式:
- BashExecutorTool (本地)
- K8sBashExecutorTool (K8S)
- Agent需要知道使用哪个工具

新的统一方式:
- BashExecutorTool (支持execution_mode配置)
  - execution_mode="local" -> 本地执行
  - execution_mode="k8s" -> K8S执行
- Agent使用相同的工具名称和接口
- 底层实现对Agent透明
    """)
    
    # 检查依赖
    try:
        import openai
    except ImportError:
        print("❌ 需要安装openai库: pip install openai")
        sys.exit(1)
    
    asyncio.run(main())
