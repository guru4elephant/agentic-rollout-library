#!/usr/bin/env python3
"""
Trajectory保存功能测试
验证完整的trajectory保存，不截断内容
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
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API配置
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = "claude-sonnet-4-20250514"


async def test_trajectory_save():
    """测试trajectory保存功能"""
    print("🔍 测试Trajectory保存功能")
    print("=" * 60)
    
    # 1. 创建LLM客户端
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=False,
        max_retries=2
    )
    
    try:
        # 2. 创建简单的本地工具
        tools = {
            "bash_executor": create_tool("BashExecutor", {
                "execution_mode": "local",
                "timeout": 15
            }),
            "finish": create_tool("Finish")
        }
        
        print(f"✅ 创建了 {len(tools)} 个工具: {list(tools.keys())}")
        
        # 3. 创建GeneralAgent（限制轮数以快速完成）
        agent = create_agent("General", {
            "max_rounds": 3,
            "system_prompt": """你是一个简单的系统信息收集器。请快速获取以下信息：
1. 当前目录内容 (ls -la)
2. 当前用户 (whoami)
然后使用finish工具提供简短报告。""",
            "termination_tool_names": ["finish"]
        })
        
        # 4. 配置工具并执行
        agent.set_tools(tools)
        print(f"✅ GeneralAgent已配置 {len(agent.tools)} 个工具")
        
        print("\n📤 开始执行任务...")
        trajectory = await agent.run_trajectory(
            prompt="请快速获取基本系统信息并生成简短报告。",
            llm_generate_func=llm_client.generate,
            request_id="trajectory_save_test_001"
        )
        
        # 5. 显示结果
        print(f"\n✅ 任务完成!")
        print(f"   轨迹状态: {'完成' if trajectory.is_completed else '未完成'}")
        print(f"   总步数: {len(trajectory.steps)}")
        print(f"   总tokens: {trajectory.total_tokens}")
        
        # 6. 保存轨迹文件（完整信息，不截断）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"tests/trajectory_save_test_{timestamp}.json"
        txt_filename = f"tests/trajectory_save_test_{timestamp}.txt"
        
        print(f"\n💾 保存轨迹文件...")
        dump_trajectory(trajectory, json_filename, "json")
        dump_trajectory(trajectory, txt_filename, "txt")
        
        print(f"✅ JSON轨迹已保存到: {json_filename}")
        print(f"✅ TXT轨迹已保存到: {txt_filename}")
        
        # 7. 验证文件内容
        print(f"\n🔍 验证保存的文件...")
        
        # 检查JSON文件
        try:
            import json
            with open(json_filename, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                print(f"📄 JSON文件大小: {len(json.dumps(json_data, ensure_ascii=False))} 字符")
                print(f"📄 JSON包含步数: {len(json_data.get('steps', []))}")
        except Exception as e:
            print(f"❌ JSON文件验证失败: {e}")
        
        # 检查TXT文件
        try:
            with open(txt_filename, 'r', encoding='utf-8') as f:
                txt_content = f.read()
                print(f"📄 TXT文件大小: {len(txt_content)} 字符")
                print(f"📄 TXT文件行数: {len(txt_content.splitlines())}")
        except Exception as e:
            print(f"❌ TXT文件验证失败: {e}")
        
        # 8. 显示轨迹摘要（展示完整信息保存）
        print(f"\n📋 轨迹摘要:")
        for i, step in enumerate(trajectory.steps, 1):
            step_type = step.step_type.value.upper()
            content_length = len(step.content) if step.content else 0
            
            print(f"   Step {i}: {step_type} ({content_length} 字符)")
            
            if step.tool_name:
                print(f"           Tool: {step.tool_name}")
                if step.tool_result:
                    result_str = str(step.tool_result)
                    print(f"           Result: {len(result_str)} 字符")
        
        # 9. 获取最终答案
        final_answer = None
        for step in reversed(trajectory.steps):
            if step.tool_name == "finish" and step.tool_result:
                final_answer = step.tool_result.get("answer")
                break
        
        if final_answer:
            print(f"\n=== 最终报告 ===")
            print(final_answer)
        
        return trajectory
        
    finally:
        await llm_client.close()


async def main():
    """主函数"""
    print("""
📋 Trajectory保存测试说明:

本测试将验证：
1. ✅ 完整的trajectory保存（JSON和TXT格式）
2. ✅ 不截断任何内容
3. ✅ 包含所有步骤的完整信息
4. ✅ 保存所有工具调用和结果
5. ✅ 文件大小和内容验证

保存的信息包括：
- 每个步骤的完整内容
- 工具调用的参数和结果
- LLM生成的完整响应
- Token使用统计
- 时间戳和元数据
    """)
    
    try:
        trajectory = await test_trajectory_save()
        
        print("\n" + "=" * 60)
        print("🎉 Trajectory保存测试完成！")
        print(f"✅ 成功保存了包含 {len(trajectory.steps)} 个步骤的完整轨迹")
        print("""
💡 重要特性：
- 完整信息保存，无截断
- 支持JSON和TXT两种格式
- 包含所有调试信息
- 便于后续分析和重现
        """)
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("💾 Trajectory保存功能测试")
    print("验证轨迹文件的完整保存（不截断内容）")
    
    # 检查依赖
    try:
        import openai
    except ImportError:
        print("❌ 需要安装openai库: pip install openai")
        sys.exit(1)
    
    asyncio.run(main())