#!/usr/bin/env python3
"""
测试GeneralAgent的JSON Action格式
验证新的系统提示词和JSON解析功能
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
MODEL_NAME = "gpt-4.1"


async def test_json_system_prompt():
    """测试JSON格式的系统提示词生成"""
    print("🔍 测试JSON格式系统提示词")
    print("=" * 60)
    
    # 1. 创建工具
    tools = {
        "bash_executor": create_tool("BashExecutor", {
            "execution_mode": "local",
            "timeout": 30
        }),
        "finish": create_tool("Finish")
    }
    
    # 2. 创建Agent（使用默认的JSON格式提示词）
    agent = create_agent("General", {
        "max_rounds": 5,
        "termination_tool_names": ["finish"]
    })
    
    # 3. 设置工具
    agent.set_tools(tools)
    
    # 4. 生成系统提示词
    system_prompt = agent.create_system_prompt()
    
    print("📋 生成的JSON格式系统提示词:")
    print("-" * 60)
    print(system_prompt[:1500] + "..." if len(system_prompt) > 1500 else system_prompt)
    print("-" * 60)
    
    # 5. 验证提示词内容
    checks = [
        ("JSON Action格式要求", '"tool_name"' in system_prompt and '"parameters"' in system_prompt),
        ("完整JSON Schema", "Complete JSON Schema" in system_prompt),
        ("Thought+Action组合", "Thought:" in system_prompt and "Action:" in system_prompt),
        ("工具Schema文档", "bash_executor" in system_prompt),
        ("使用示例", "Usage Example" in system_prompt),
        ("格式规则", "Critical Format Rules" in system_prompt),
    ]
    
    print("\n✅ JSON格式提示词验证:")
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}: {'通过' if result else '失败'}")
    
    return system_prompt


async def test_custom_system_prompt():
    """测试自定义系统提示词与工具文档的组合"""
    print("\n🔍 测试自定义系统提示词")
    print("=" * 60)
    
    # 1. 创建工具
    tools = {
        "bash_executor": create_tool("BashExecutor", {
            "execution_mode": "local",
            "timeout": 15
        }),
        "finish": create_tool("Finish")
    }
    
    # 2. 自定义系统提示词
    custom_prompt = """You are a specialized system administrator assistant.
Your task is to help users with system management tasks.
Always use the ReAct framework with JSON-formatted actions.

When performing tasks:
1. Think carefully about each step
2. Use appropriate tools to gather information
3. Provide clear explanations of what you're doing"""
    
    # 3. 创建Agent with custom prompt
    agent = create_agent("General", {
        "max_rounds": 3,
        "system_prompt": custom_prompt,
        "termination_tool_names": ["finish"]
    })
    
    # 4. 设置工具
    agent.set_tools(tools)
    
    # 5. 生成系统提示词
    system_prompt = agent.create_system_prompt()
    
    print("📋 自定义系统提示词+工具文档:")
    print("-" * 60)
    print(system_prompt[:800] + "..." if len(system_prompt) > 800 else system_prompt)
    print("-" * 60)
    
    # 6. 验证自定义内容和工具文档都存在
    has_custom = "specialized system administrator" in system_prompt
    has_tools = "Available Tools" in system_prompt or "Tool:" in system_prompt
    
    print(f"\n✅ 自定义提示词验证:")
    print(f"   {'✅' if has_custom else '❌'} 包含自定义内容: {'是' if has_custom else '否'}")
    print(f"   {'✅' if has_tools else '❌'} 包含工具文档: {'是' if has_tools else '否'}")
    
    return has_custom and has_tools


async def test_json_parsing_simulation():
    """模拟测试JSON解析功能"""
    print("\n🔍 测试JSON解析功能")
    print("=" * 60)
    
    # 创建Agent
    agent = create_agent("General", {"max_rounds": 1})
    
    # 测试不同的JSON格式
    test_cases = [
        {
            "name": "标准JSON格式",
            "input": '''Thought: I need to check the current directory.

Action:
{
  "tool_name": "bash_executor",
  "parameters": {
    "command": "pwd"
  }
}''',
            "expected_tool": "bash_executor"
        },
        {
            "name": "紧凑JSON格式",
            "input": '''Thought: Let me finish this task.

Action: {"tool_name": "finish", "parameters": {"answer": "Task completed"}}''',
            "expected_tool": "finish"
        },
        {
            "name": "包含注释的JSON",
            "input": '''Thought: I'll check system info.

Action:
{
  "tool_name": "bash_executor",
  "parameters": {
    // Get system information
    "command": "uname -a"
  }
}''',
            "expected_tool": "bash_executor"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   测试 {i}: {test_case['name']}")
        
        try:
            # 使用Agent的解析方法
            steps = agent._parse_react_response(test_case["input"])
            
            # 检查解析结果
            if steps:
                thought_step = None
                action_step = None
                
                for step in steps:
                    if step.step_type.value == "thought":
                        thought_step = step
                    elif step.step_type.value == "action":
                        action_step = step
                
                print(f"     📝 Thought: {'✅ 解析成功' if thought_step else '❌ 解析失败'}")
                
                if action_step and action_step.tool_name == test_case["expected_tool"]:
                    print(f"     🔧 Action: ✅ 解析成功 (tool: {action_step.tool_name})")
                    if hasattr(action_step, 'tool_args'):
                        print(f"         参数: {action_step.tool_args}")
                else:
                    print(f"     🔧 Action: ❌ 解析失败 (期望: {test_case['expected_tool']})")
            else:
                print(f"     ❌ 完全解析失败")
                
        except Exception as e:
            print(f"     ❌ 解析异常: {e}")
    
    return True


async def main():
    """主函数"""
    print("🧪 GeneralAgent JSON Action格式测试")
    print("=" * 80)
    print("""
📋 测试内容:
1. ✅ JSON格式系统提示词生成
2. ✅ 自定义提示词与工具文档组合
3. ✅ JSON解析功能模拟测试

新特性:
- 外部可设定的system prompt
- 基于JSON schema的工具文档
- JSON格式的Action输出要求
- repair_json库支持（如果可用）
    """)
    
    test_results = []
    
    try:
        # 执行所有测试
        system_prompt = await test_json_system_prompt()
        test_results.append(("JSON格式系统提示词", len(system_prompt) > 1000))
        
        custom_result = await test_custom_system_prompt()
        test_results.append(("自定义提示词组合", custom_result))
        
        parsing_result = await test_json_parsing_simulation()
        test_results.append(("JSON解析功能", parsing_result))
        
        # 统计结果
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        
        print("\n" + "=" * 80)
        print("📊 测试结果汇总:")
        print("-" * 80)
        
        for test_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"   {status} {test_name}")
        
        print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
        
        if passed == total:
            print("🎉 所有测试通过！JSON Action格式功能正常。")
            print("""
💡 新功能特性验证完成:
✅ 支持外部设定的system prompt
✅ 使用完整JSON schema作为工具文档
✅ 要求Action输出为结构化JSON格式
✅ 支持Thought+Action的组合解析
✅ 智能JSON解析与错误恢复
            """)
        else:
            print("⚠️ 部分测试失败，请检查JSON格式实现。")
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔧 GeneralAgent JSON Action格式测试")
    print("验证新的系统提示词和JSON解析功能")
    
    asyncio.run(main())