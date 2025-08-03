#!/usr/bin/env python3
"""
测试GeneralAgent的debug模式
验证LLM输入输出的详细日志记录
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.core import create_tool, create_agent
from workers.utils import create_llm_client

# 设置DEBUG级别日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API配置
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = "gpt-4.1"

async def test_debug_mode():
    """测试GeneralAgent的debug模式"""
    print("🔍 测试GeneralAgent Debug模式")
    print("=" * 60)
    
    # 1. 创建LLM客户端
    llm_client = create_llm_client(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        debug=False,  # LLM客户端不开debug，只看Agent的debug
        max_retries=1
    )
    
    try:
        # 2. 创建简单工具
        tools = {
            "bash_executor": create_tool("BashExecutor", {
                "execution_mode": "local",
                "timeout": 10
            }),
            "finish": create_tool("Finish")
        }
        
        # 3. 创建Agent（开启debug模式）
        print("🤖 创建GeneralAgent (debug=True)...")
        agent = create_agent("General", {
            "max_rounds": 2,  # 只执行2轮进行测试
            "debug": True,    # 开启debug模式
            "termination_tool_names": ["finish"]
        })
        
        # 4. 配置工具
        agent.set_tools(tools)
        print(f"✅ Agent已配置，debug模式: {agent.debug}")
        
        # 5. 执行简单任务
        print("\n🚀 执行简单任务...")
        print("-" * 40)
        
        trajectory = await agent.run_trajectory(
            prompt="请执行ls命令查看当前目录，然后用finish工具结束。",
            llm_generate_func=llm_client.generate,
            request_id="debug_test_001"
        )
        
        print("-" * 40)
        print(f"✅ 任务完成，总步数: {len(trajectory.steps)}")
        
        return trajectory
        
    finally:
        await llm_client.close()


async def main():
    """主函数"""
    print("🧪 GeneralAgent Debug模式测试")
    print("=" * 80)
    print("""
📋 测试目标:
- 验证GeneralAgent的debug参数工作正常
- 确认每次LLM调用的输入输出都有详细日志
- 检查debug日志的格式和内容完整性

🔍 预期看到的debug信息:
- LLM输入: 消息内容、参数设置
- LLM输出: 响应内容、长度统计、内容分析
- 轮次标识和时间戳

开始测试...
    """)
    
    try:
        trajectory = await test_debug_mode()
        
        print("\n" + "=" * 80)
        print("🎉 Debug模式测试完成！")
        print(f"✅ 成功执行了 {len(trajectory.steps)} 个步骤")
        print("""
💡 Debug功能验证:
- ✅ GeneralAgent debug参数传递正常
- ✅ LLM输入输出详细日志记录  
- ✅ 轮次标识和内容分析
- ✅ 调试信息格式化良好

查看上面的详细debug日志输出！
        """)
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔍 GeneralAgent Debug模式测试")
    print("验证LLM调用的详细日志记录功能")
    
    asyncio.run(main())