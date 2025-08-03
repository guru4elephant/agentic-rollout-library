#!/usr/bin/env python3
"""
演示改进后的GeneralAgent系统提示词
展示工具描述和ReAct格式规范
"""

import sys
import os

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.core import create_tool, create_agent


def demo_system_prompt():
    """演示改进后的系统提示词"""
    print("🔧 改进后的GeneralAgent系统提示词演示")
    print("=" * 80)
    
    # 1. 创建工具
    print("📋 步骤1: 创建工具")
    tools = {
        "bash_executor": create_tool("BashExecutor", {
            "execution_mode": "local",
            "timeout": 30
        }),
        "finish": create_tool("Finish")
    }
    print(f"   ✅ 创建了 {len(tools)} 个工具: {list(tools.keys())}")
    
    # 2. 创建Agent
    print("\n📋 步骤2: 创建GeneralAgent")
    agent = create_agent("General", {
        "max_rounds": 5,
        "termination_tool_names": ["finish"]
    })
    
    # 3. 设置工具
    print("\n📋 步骤3: 配置工具")
    agent.set_tools(tools)
    print(f"   ✅ Agent已配置 {len(agent.tools)} 个工具")
    
    # 4. 生成系统提示词
    print("\n📋 步骤4: 生成系统提示词")
    system_prompt = agent.create_system_prompt()
    
    # 5. 显示系统提示词
    print("\n" + "="*80)
    print("📝 生成的系统提示词:")
    print("="*80)
    print(system_prompt)
    print("="*80)
    
    # 6. 分析提示词内容
    print("\n📊 系统提示词分析:")
    
    # 统计信息
    total_chars = len(system_prompt)
    total_lines = len(system_prompt.split('\n'))
    
    print(f"   📏 总长度: {total_chars:,} 字符")
    print(f"   📄 总行数: {total_lines} 行")
    
    # 内容检查
    content_checks = [
        ("ReAct框架说明", "ReAct" in system_prompt and "Reasoning + Acting" in system_prompt),
        ("格式要求", "Thought:" in system_prompt and "Action:" in system_prompt),
        ("工具文档", "Available Tools" in system_prompt),
        ("bash_executor工具", "bash_executor" in system_prompt),
        ("参数说明", "Parameters:" in system_prompt),
        ("使用示例", "Usage:" in system_prompt),
        ("finish工具", "finish" in system_prompt),
        ("格式示例", "tool_name(param=value)" in system_prompt),
        ("流程说明", "Process Flow" in system_prompt),
        ("示例流程", "Example ReAct Flow" in system_prompt),
    ]
    
    print("\n   ✅ 内容完整性检查:")
    for check_name, passed in content_checks:
        status = "✅" if passed else "❌"
        print(f"      {status} {check_name}")
    
    # 工具信息检查
    print("\n   🔧 工具信息检查:")
    for tool_name in tools.keys():
        if tool_name in system_prompt:
            print(f"      ✅ {tool_name}: 已包含在提示词中")
            
            # 检查是否有参数说明
            if "Parameters:" in system_prompt:
                tool_section_start = system_prompt.find(f"**{tool_name}**")
                if tool_section_start != -1:
                    tool_section = system_prompt[tool_section_start:tool_section_start+500]
                    if "Parameters:" in tool_section:
                        print(f"         📋 包含参数说明")
                    if "Usage:" in tool_section:
                        print(f"         📖 包含使用示例")
        else:
            print(f"      ❌ {tool_name}: 未包含在提示词中")
    
    # 关键改进点检查
    print("\n   🚀 关键改进点验证:")
    improvements = [
        ("明确的ReAct格式要求", "MUST follow this exact format" in system_prompt),
        ("详细的格式规范", "Critical Format Requirements" in system_prompt),
        ("处理流程说明", "Process Flow" in system_prompt),
        ("使用指南", "Guidelines" in system_prompt),
        ("终止条件说明", "Termination" in system_prompt),
        ("完整示例流程", "Example ReAct Flow" in system_prompt),
        ("错误处理指导", "Handle errors gracefully" in system_prompt),
    ]
    
    for improvement_name, condition in improvements:
        status = "✅" if condition else "❌"
        print(f"      {status} {improvement_name}")
    
    print("\n" + "="*80)
    print("🎉 系统提示词演示完成！")
    print("""
💡 主要改进特性:
✅ 详细的工具功能描述和参数说明
✅ 明确的ReAct格式要求 (Thought: / Action:)  
✅ 具体的工具调用格式 (tool_name(param=value))
✅ 完整的使用示例和流程说明
✅ 错误处理和最佳实践指导
✅ 结构化的文档格式，易于理解

🎯 这些改进将帮助LLM:
- 更准确地理解工具功能和参数
- 严格遵循ReAct格式规范
- 正确使用工具调用语法
- 更好地处理错误和异常情况
    """)
    
    return system_prompt


if __name__ == "__main__":
    demo_system_prompt()