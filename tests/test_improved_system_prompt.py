#!/usr/bin/env python3
"""
测试改进后的GeneralAgent系统提示词
验证工具描述和ReAct格式规范
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
#MODEL_NAME = "claude-sonnet-4-20250514"
MODEL_NAME = "gpt-4.1"


async def test_system_prompt_generation():
    """测试系统提示词生成功能"""
    print("🔍 测试系统提示词生成")
    print("=" * 60)
    
    # 1. 创建工具
    tools = {
        "bash_executor": create_tool("BashExecutor", {
            "execution_mode": "local",
            "timeout": 30
        }),
        "finish": create_tool("Finish")
    }
    
    # 2. 创建Agent
    agent = create_agent("General", {
        "max_rounds": 3,
        "termination_tool_names": ["finish"]
    })
    
    # 3. 设置工具
    agent.set_tools(tools)
    
    # 4. 生成系统提示词
    system_prompt = agent.create_system_prompt()
    
    print("📋 生成的系统提示词:")
    print("-" * 60)
    print(system_prompt)
    print("-" * 60)
    
    # 5. 验证提示词内容
    checks = [
        ("ReAct框架说明", "ReAct" in system_prompt),
        ("Thought格式要求", "Thought:" in system_prompt),
        ("Action格式要求", "Action:" in system_prompt),
        ("工具列表", "Available Tools" in system_prompt),
        ("bash_executor工具", "bash_executor" in system_prompt),
        ("finish工具", "finish" in system_prompt),
        ("参数格式说明", "param=value" in system_prompt),
        ("示例流程", "Example ReAct Flow" in system_prompt),
    ]
    
    print("\n✅ 系统提示词内容验证:")
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}: {'通过' if result else '失败'}")
    
    # 6. 统计信息
    prompt_length = len(system_prompt)
    prompt_lines = len(system_prompt.split('\n'))
    
    print(f"\n📊 提示词统计:")
    print(f"   总长度: {prompt_length} 字符")
    print(f"   总行数: {prompt_lines} 行")
    
    return system_prompt


async def test_agent_with_improved_prompt():
    """测试使用改进提示词的Agent执行效果"""
    print("\n🚀 测试改进后的Agent执行")
    print("=" * 60)
    
    # 1. 创建LLM客户端
    llm_client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True)
    
    try:
        # 2. 创建工具
        tools = {
            "bash_executor": create_tool("BashExecutor", {
                "execution_mode": "local",
                "timeout": 15
            }),
            "finish": create_tool("Finish")
        }
        
        # 3. 创建Agent（不使用自定义提示词，让它使用改进后的默认提示词）
        agent = create_agent("General", {
            "max_rounds": 5,
            "termination_tool_names": ["finish"]
        })
        
        # 4. 设置工具
        agent.set_tools(tools)
        print(f"✅ 配置了 {len(agent.tools)} 个工具")
        
        # 5. 执行任务
        print("\n📤 开始执行任务...")
        trajectory = await agent.run_trajectory(
            prompt="请使用ReAct格式获取当前系统的用户名和Python版本，然后提供总结报告。",
            llm_generate_func=llm_client.generate,
            request_id="improved_prompt_test_001"
        )
        
        # 6. 分析结果
        print(f"\n✅ 任务完成!")
        print(f"   轨迹状态: {'完成' if trajectory.is_completed else '未完成'}")
        print(f"   总步数: {len(trajectory.steps)}")
        print(f"   总tokens: {trajectory.total_tokens}")
        
        # 7. 分析ReAct格式使用情况
        thought_steps = [s for s in trajectory.steps if s.step_type.value == "thought"]
        action_steps = [s for s in trajectory.steps if s.step_type.value == "action"]
        
        print(f"\n📋 ReAct格式分析:")
        print(f"   Thought步数: {len(thought_steps)}")
        print(f"   Action步数: {len(action_steps)}")
        
        # 检查格式规范性
        proper_format_count = 0
        for step in trajectory.steps:
            if step.content and (step.content.startswith("Thought:") or step.content.startswith("Action:")):
                proper_format_count += 1
        
        print(f"   正确格式步数: {proper_format_count}")
        print(f"   格式规范率: {proper_format_count/len(trajectory.steps)*100:.1f}%")
        
        # 8. 显示最终答案
        final_answer = None
        for step in reversed(trajectory.steps):
            if step.tool_name == "finish" and step.tool_result:
                final_answer = step.tool_result.get("answer")
                break
        
        if final_answer:
            print(f"\n=== 最终报告 ===")
            print(final_answer)
        
        # 9. 保存轨迹
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"tests/improved_prompt_test_{timestamp}.json"
        txt_filename = f"tests/improved_prompt_test_{timestamp}.txt"
        
        dump_trajectory(trajectory, json_filename, "json")
        dump_trajectory(trajectory, txt_filename, "txt")
        print(f"\n💾 轨迹已保存到 {json_filename} 和 {txt_filename}")
        
        return trajectory
        
    finally:
        await llm_client.close()


async def main():
    """主函数"""
    print("🔧 改进后的GeneralAgent系统提示词测试")
    print("=" * 80)
    print("""
📋 测试内容:
1. ✅ 验证系统提示词包含完整的工具描述
2. ✅ 验证ReAct格式规范和要求
3. ✅ 验证工具调用格式说明
4. ✅ 测试Agent使用改进提示词的执行效果
5. ✅ 分析ReAct格式的使用情况

改进要点:
- 详细的工具参数说明
- 明确的ReAct格式要求
- 具体的工具调用示例
- 完整的流程说明
    """)
    
    try:
        # 测试系统提示词生成
        system_prompt = await test_system_prompt_generation()
        
        # 测试Agent执行效果
        trajectory = await test_agent_with_improved_prompt()
        
        print("\n" + "=" * 80)
        print("🎉 改进后的系统提示词测试完成！")
        print(f"✅ 系统提示词长度: {len(system_prompt)} 字符")
        print(f"✅ Agent执行步数: {len(trajectory.steps)}")
        print("""
💡 主要改进:
- 📚 详细的工具功能和参数说明
- 📝 明确的ReAct格式要求 (Thought: / Action:)
- 🔧 具体的工具调用格式 (tool_name(param=value))
- 📖 完整的使用示例和流程说明
- 🎯 更好的格式规范和错误处理指导

这些改进将帮助LLM更好地理解和使用ReAct框架！
        """)
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔧 改进后的GeneralAgent系统提示词测试")
    print("验证工具描述、ReAct格式规范和执行效果")
    
    # 检查依赖
    try:
        import openai
    except ImportError:
        print("❌ 需要安装openai库: pip install openai")
        sys.exit(1) 
    
    asyncio.run(main())
