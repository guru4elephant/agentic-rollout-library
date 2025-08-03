#!/usr/bin/env python3
"""
工厂模式使用示例
演示如何使用ToolFactory和AgentFactory基于类名和配置创建实例
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.core import create_tool, create_tools, create_agent, create_agents

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def example_tool_factory():
    """工具工厂使用示例"""
    print("=== 工具工厂使用示例 ===")
    
    # 1. 基于类名创建单个工具
    print("\n1. 创建单个工具:")
    
    # 创建计算器工具，带配置
    calculator = create_tool("Calculator", {
        "debug": True,
        "precision": 6
    })
    print(f"   创建了 {type(calculator).__name__} 工具")
    
    # 创建搜索工具，带配置
    search = create_tool("Search", {
        "max_results": 100,
        "max_file_size": 2048000
    })
    print(f"   创建了 {type(search).__name__} 工具")
    
    # 创建文件编辑工具
    file_editor = create_tool("FileEditor", {
        "encoding": "utf-8",
        "backup": True
    })
    print(f"   创建了 {type(file_editor).__name__} 工具")
    
    # 2. 批量创建多个工具
    print("\n2. 批量创建工具:")
    
    tool_configs = {
        "Calculator": {
            "debug": False,
            "precision": 10
        },
        "Search": {
            "max_results": 50,
            "search_extensions": [".py", ".txt", ".md"]
        },
        "Finish": {
            # Finish工具通常不需要特殊配置
        }
    }
    
    tools = create_tools(tool_configs)
    print(f"   批量创建了 {len(tools)} 个工具:")
    for name, tool in tools.items():
        print(f"   - {name}: {type(tool).__name__}")
    
    # 3. 测试工具执行
    print("\n3. 测试工具执行:")
    
    try:
        import uuid
        
        # 测试计算器
        calc_result = await calculator.execute_tool(
            str(uuid.uuid4()), 
            {"expression": "sqrt(16) + 2^3"}
        )
        print(f"   计算结果: {calc_result.result}")
        
        # 测试文件搜索
        search_result = await search.execute_tool(
            str(uuid.uuid4()),
            {
                "command": "search_files",
                "pattern": "*.py",
                "path": ".",
                "max_results": 5
            }
        )
        print(f"   搜索找到 {len(search_result.result.get('matches', []))} 个匹配项")
        
    except Exception as e:
        print(f"   工具执行出错: {e}")


async def example_agent_factory():
    """智能体工厂使用示例"""
    print("\n=== 智能体工厂使用示例 ===")
    
    # 1. 基于类名创建单个智能体
    print("\n1. 创建单个智能体:")
    
    # 创建通用智能体，带配置
    general_agent = create_agent("General", {
        "max_rounds": 5,
        "system_prompt": "你是一个数学助手，专门帮助解决数学问题。",
        "termination_tool_names": ["finish"]
    })
    print(f"   创建了 {type(general_agent).__name__}")
    print(f"   最大轮数: {general_agent.max_rounds}")
    
    # 创建ReAct智能体
    react_agent = create_agent("React", {
        "max_steps": 10,
        "temperature": 0.7,
        "require_thought_before_action": True
    })
    print(f"   创建了 {type(react_agent).__name__}")
    print(f"   最大步数: {react_agent.max_steps}")
    
    # 2. 批量创建多个智能体
    print("\n2. 批量创建智能体:")
    
    agent_configs = {
        "General": {
            "max_rounds": 3,
            "system_prompt": "你是一个编程助手。",
            "termination_tool_names": ["finish"]
        },
        "React": {
            "max_steps": 8,
            "temperature": 0.5
        }
    }
    
    agents = create_agents(agent_configs)
    print(f"   批量创建了 {len(agents)} 个智能体:")
    for name, agent in agents.items():
        print(f"   - {name}: {type(agent).__name__}")
    
    return general_agent


async def example_complete_workflow():
    """完整工作流示例"""
    print("\n=== 完整工作流示例 ===")
    
    # 1. 使用工厂创建工具
    print("\n1. 创建所需工具:")
    
    tools = create_tools({
        "Calculator": {"debug": False},
        "Search": {"max_results": 20},
        "Finish": {}
    })
    
    print(f"   创建了工具: {list(tools.keys())}")
    
    # 2. 使用工厂创建智能体
    print("\n2. 创建智能体:")
    
    agent = create_agent("General", {
        "max_rounds": 4,
        "system_prompt": """你是一个helpful的数学助手。使用ReAct框架解决问题：
        1. Thought: 分析问题
        2. Action: 使用工具
        3. 重复直到完成
        
        可用工具:
        - calculator: 数学计算
        - search: 文件搜索  
        - finish: 完成任务""",
        "termination_tool_names": ["finish"]
    })
    
    print(f"   创建了 {type(agent).__name__}")
    
    # 3. 配置智能体工具
    agent.set_tools(tools)
    print(f"   为智能体配置了 {len(agent.tools)} 个工具")
    
    # 4. 模拟LLM响应
    class MockLLM:
        def __init__(self):
            self.responses = [
                "Thought: 我需要计算这个数学表达式。",
                "Action: calculator(expression=25*4+10/2)",
                "Thought: 计算完成，结果是105。现在我需要完成任务。", 
                "Action: finish(answer=计算结果是105, reasoning=25*4=100, 10/2=5, 100+5=105)"
            ]
            self.call_count = 0
        
        async def __call__(self, messages, **kwargs):
            if self.call_count < len(self.responses):
                response = self.responses[self.call_count]
                self.call_count += 1
                return response
            return "Action: finish(answer=任务完成)"
    
    # 5. 执行完整工作流
    print("\n3. 执行工作流:")
    
    try:
        trajectory = await agent.run_trajectory(
            prompt="计算表达式: 25 * 4 + 10 / 2",
            llm_generate_func=MockLLM(),
            request_id="factory_workflow_001"
        )
        
        print(f"   轨迹完成: {trajectory.is_completed}")
        print(f"   总步数: {len(trajectory.steps)}")
        
        # 显示最后的结果
        for step in trajectory.steps:
            if step.step_type.value == "action_result" and step.tool_name == "finish":
                print(f"   最终答案: {step.tool_result.get('answer', 'N/A')}")
                break
        
    except Exception as e:
        print(f"   工作流执行出错: {e}")


def example_configuration_patterns():
    """配置模式示例"""
    print("\n=== 配置模式示例 ===")
    
    print("\n1. 工具配置模式:")
    
    # 不同场景的工具配置
    scenarios = {
        "数学计算场景": {
            "Calculator": {"debug": True, "precision": 10},
            "Finish": {}
        },
        "文件处理场景": {
            "FileEditor": {"encoding": "utf-8", "backup": True},
            "Search": {"max_results": 100, "search_extensions": [".py", ".txt"]},
            "Finish": {}
        },
        "系统管理场景": {
            "BashExecutor": {"timeout": 30, "shell": "/bin/bash"},
            "FileEditor": {"encoding": "utf-8"},
            "Search": {"max_file_size": 5242880},  # 5MB
            "Finish": {}
        }
    }
    
    for scenario_name, tool_config in scenarios.items():
        print(f"   {scenario_name}:")
        for tool_name, config in tool_config.items():
            print(f"     - {tool_name}: {config}")
    
    print("\n2. 智能体配置模式:")
    
    # 不同角色的智能体配置
    roles = {
        "数学导师": {
            "max_rounds": 5,
            "system_prompt": "你是一个耐心的数学导师，善于解释数学概念。",
            "termination_tool_names": ["finish"]
        },
        "代码审查员": {
            "max_rounds": 8,
            "system_prompt": "你是一个经验丰富的代码审查员，专注于代码质量和最佳实践。",
            "termination_tool_names": ["finish"]
        },
        "问题解决者": {
            "max_rounds": 10,
            "system_prompt": "你是一个系统的问题解决者，使用逻辑思维分析问题。",
            "termination_tool_names": ["finish"]
        }
    }
    
    for role_name, agent_config in roles.items():
        print(f"   {role_name}:")
        for key, value in agent_config.items():
            if isinstance(value, str) and len(value) > 50:
                print(f"     - {key}: {value[:50]}...")
            else:
                print(f"     - {key}: {value}")


async def main():
    """主函数"""
    print("工厂模式使用示例")
    print("================")
    
    try:
        await example_tool_factory()
        await example_agent_factory()
        await example_complete_workflow()
        example_configuration_patterns()
        
        print("\n✅ 所有示例执行完成!")
        
    except Exception as e:
        logger.error(f"示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
    🏭 工厂模式特性总结:
    
    ToolFactory:
    ✓ 基于类名创建工具: create_tool("Calculator", config)
    ✓ 支持配置参数传递给构造函数
    ✓ 自动模块加载和类发现
    ✓ 内置工具注册 (Calculator, Search, FileEditor, etc.)
    ✓ 批量创建: create_tools(tool_configs)
    
    AgentFactory:
    ✓ 基于类名创建智能体: create_agent("General", config)
    ✓ 支持配置参数传递给构造函数
    ✓ 自动模块加载和类发现
    ✓ 内置智能体注册 (General, React, Tool, Coding)
    ✓ 批量创建: create_agents(agent_configs)
    
    使用优势:
    📝 配置化创建，无需手动导入类
    🔧 支持复杂配置参数传递
    🚀 自动缓存和优化加载
    📊 统一的创建接口
    🔍 内置信息查询功能
    """)
    
    asyncio.run(main())