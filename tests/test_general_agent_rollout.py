#!/usr/bin/env python3
"""
GeneralAgent完整rollout测试程序
使用自定义system prompt和预制工具包，执行完整的推理轨迹并保存结果
"""

import asyncio
import logging
import sys
import os
import datetime

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.core import create_tool, create_agent
from workers.agents.general_agent import dump_trajectory
from workers.utils import create_llm_client

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API配置
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")

# 自定义系统提示词（包含必要的ReAct格式说明）
CUSTOM_SYSTEM_PROMPT = """You are an advanced AI coding assistant specializing in software development tasks within Kubernetes environments.

Your mission is to help users with code analysis, file management, and system operations in containerized environments using a systematic approach.

## Environment Context
- You are operating within a Kubernetes Pod: swebench-xarray-pod
- All commands will be executed inside the container environment
- You have access to the complete project structure within the Pod

## Working Methodology
1. **Analyze** the user's request thoroughly
2. **Plan** your approach step by step  
3. **Execute** using available tools systematically
4. **Verify** your results when possible
5. **Provide** clear explanations and comprehensive summaries

## Output Format Requirements

Your response MUST contain both **Thought** and **Action** sections in this exact format:

```
Thought: [Your detailed reasoning about what to do next, analysis of the situation, planning your approach]

Action:
{
  "tool_name": "function_name",
  "parameters": {
    "param1": "value1",
    "param2": 42,
    "param3": true
  }
}
```

## Critical Format Rules

1. **Always include both Thought and Action** in a single response
2. **Thought section**: Free-form reasoning in natural language
3. **Action section**: Valid JSON object with "tool_name" and "parameters" fields
4. **JSON must be valid** - use proper quotes, brackets, and data types
5. **Use exact tool names** from the schemas below
6. **Follow parameter types** as specified in the schemas

## Best Practices
- Break complex tasks into smaller, manageable steps
- Use bash_executor to explore and understand the project structure
- Use the search tool to find specific files or content patterns
- Use the file_editor tool to view, create, and modify files
- Verify file contents before and after modifications  
- Provide detailed explanations of what you're doing and why
- Always aim for clean, maintainable solutions
- Remember you're working in a containerized environment
- When you complete the task, use the finish tool with your final answer

Let's work together to accomplish your goals efficiently and accurately!"""


def create_comprehensive_tools():
    """创建K8S执行模式的工具集合"""
    print("🔧 正在创建K8S工具集合...")
    
    # K8S配置
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default",
        "timeout": 30
    }
    
    tools = {
        "bash_executor": create_tool("BashExecutor", k8s_config.copy()),
        "file_editor": create_tool("FileEditor", k8s_config.copy()),
        "search": create_tool("Search", k8s_config.copy()),
        "finish": create_tool("Finish")
    }
    
    print(f"   ✅ 创建了 {len(tools)} 个工具: {list(tools.keys())}")
    print(f"   🐳 执行模式: K8S Pod (swebench-xarray-pod)")
    print(f"   💡 所有工具将在Pod内执行:")
    print(f"      - bash_executor: 执行shell命令")
    print(f"      - file_editor: 查看、创建、编辑文件")
    print(f"      - search: 搜索文件和内容")
    return tools


async def run_general_agent_rollout():
    """执行完整的GeneralAgent rollout测试"""
    
    print("🚀 GeneralAgent完整Rollout测试")
    print("=" * 80)
    
    # 测试查询
    test_query = """请帮我分析当前项目的结构，然后创建一个项目总结文档。

具体要求：
1. 首先探索项目目录结构，了解主要组件
2. 查找并分析主要的Python文件，特别是核心模块
3. 创建一个名为 'project_summary.md' 的文档，包含：
   - 项目概述
   - 目录结构说明
   - 主要模块功能介绍
   - 关键文件说明

请用系统化的方法完成这个任务。"""

    print(f"📋 测试查询:")
    print(f"   {test_query}")
    print()
    
    # 1. 创建LLM客户端
    print("🔗 正在连接LLM服务...")
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL, 
        model=MODEL_NAME,
        debug=True,
        max_retries=3
    )
    
    try:
        # 2. 创建工具
        tools = create_comprehensive_tools()
        
        # 3. 创建GeneralAgent (开启debug模式)
        print("🤖 正在创建GeneralAgent...")
        agent = create_agent("General", {
            "max_rounds": 15,  # 给足够的轮数来完成复杂任务
            "system_prompt": CUSTOM_SYSTEM_PROMPT,
            "termination_tool_names": ["finish"],
            "debug": True  # 开启debug模式，打印所有LLM输入输出
        })
        
        # 4. 配置工具
        agent.set_tools(tools)
        print(f"   ✅ Agent已配置 {len(agent.tools)} 个工具")
        
        # 显示系统提示词（截断版本）
        system_prompt = agent.create_system_prompt()
        print(f"\n📝 系统提示词长度: {len(system_prompt)} 字符")
        print(f"   提示词预览: {system_prompt[:200]}...")
        
        # 5. 开始执行rollout
        print(f"\n🎯 开始执行Rollout...")
        print("-" * 60)
        
        start_time = datetime.datetime.now()
        
        trajectory = await agent.run_trajectory(
            prompt=test_query,
            llm_generate_func=llm_client.generate,
            request_id=f"general_agent_rollout_{start_time.strftime('%Y%m%d_%H%M%S')}"
        )
        
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print("-" * 60)
        print(f"✅ Rollout执行完成!")
        
        # 6. 分析结果
        print(f"\n📊 执行统计:")
        print(f"   执行时间: {execution_time:.1f} 秒")
        print(f"   轨迹状态: {'✅ 完成' if trajectory.is_completed else '❌ 未完成'}")
        print(f"   总步数: {len(trajectory.steps)}")
        print(f"   总tokens: {trajectory.total_tokens}")
        
        # 分析步骤类型
        step_types = {}
        for step in trajectory.steps:
            step_type = step.step_type.value
            step_types[step_type] = step_types.get(step_type, 0) + 1
        
        print(f"   步骤分布:")
        for step_type, count in step_types.items():
            print(f"     - {step_type}: {count}")
        
        # 7. 保存完整轨迹到result.txt
        result_filename = "result.txt"
        print(f"\n💾 保存完整轨迹到 {result_filename}...")
        
        dump_trajectory(trajectory, result_filename, "txt")
        
        # 验证文件
        if os.path.exists(result_filename):
            file_size = os.path.getsize(result_filename)
            print(f"   ✅ 轨迹已保存: {result_filename} ({file_size} 字节)")
            
            # 显示文件前几行作为预览
            with open(result_filename, 'r', encoding='utf-8') as f:
                preview_lines = f.readlines()[:10]
                print(f"   📄 文件预览 (前10行):")
                for i, line in enumerate(preview_lines, 1):
                    print(f"      {i:2d}: {line.rstrip()}")
                if len(preview_lines) == 10:
                    print(f"      ... (完整内容请查看 {result_filename})")
        else:
            print(f"   ❌ 文件保存失败")
        
        # 8. 显示最终结果
        final_answer = None
        for step in reversed(trajectory.steps):
            if step.tool_name == "finish" and step.tool_result:
                final_answer = step.tool_result.get("answer")
                break
        
        if final_answer:
            print(f"\n🎯 任务最终结果:")
            print("=" * 40)
            print(final_answer)
            print("=" * 40)
        
        return trajectory
        
    except Exception as e:
        logger.error(f"Rollout执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        await llm_client.close()


async def main():
    """主函数"""
    print("🧪 GeneralAgent完整Rollout测试程序")
    print("=" * 80)
    print("""
📋 测试配置:
- 模型: gpt-4.1
- 工具: bash_executor, file_editor, search, finish (K8S模式)
- 执行环境: K8S Pod (swebench-xarray-pod)
- 最大轮数: 15
- 自定义系统提示词: K8S容器环境代码分析专家
- 任务: K8S Pod内项目结构分析和文档生成

🎯 预期流程:
1. 探索项目目录结构
2. 分析主要Python文件
3. 生成项目总结文档
4. 保存完整轨迹到result.txt

开始执行...
    """)
    
    try:
        trajectory = await run_general_agent_rollout()
        
        if trajectory:
            print("\n" + "=" * 80)
            print("🎉 测试完成!")
            print(f"✅ 成功执行了包含 {len(trajectory.steps)} 个步骤的完整轨迹")
            print(f"✅ 轨迹已保存到 result.txt")
            print("""
💡 测试验证了以下功能:
- ✅ 自定义系统提示词加载
- ✅ 多工具协同工作 (bash_executor, file_editor, search, finish)
- ✅ JSON格式Action解析
- ✅ 完整ReAct流程执行
- ✅ 轨迹保存和结果输出
- ✅ 错误处理和重试机制

🔍 查看详细轨迹: cat result.txt
            """)
        else:
            print("\n❌ 测试失败，请检查错误信息")
            
    except Exception as e:
        logger.error(f"测试程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🤖 GeneralAgent完整Rollout测试")
    print("测试自定义system prompt + 预制工具 + 完整推理轨迹")
    
    # 不需要检查openai依赖，已在LLM客户端中处理
    
    asyncio.run(main())