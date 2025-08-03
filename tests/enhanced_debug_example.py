#!/usr/bin/env python3
"""
Enhanced Debug模式示例
展示改进后的debug模式如何完整显示输入和输出信息
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.utils import create_llm_client

# API配置
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")


def setup_debug_logging():
    """设置debug级别的日志"""
    # 清除现有的处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 设置debug级别日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )
    
    # 确保LLM客户端的日志也是DEBUG级别
    logging.getLogger('workers.utils.llm_client').setLevel(logging.DEBUG)


async def test_short_input():
    """测试短输入的debug显示"""
    print("\n🔍 === 测试短输入的Debug显示 ===")
    
    client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True)
    
    try:
        messages = [
            {"role": "user", "content": "Hello, world!"}
        ]
        
        print("📤 发送短消息...")
        response = await client.generate(messages, max_tokens=50, temperature=0.5)
        print(f"✅ 收到响应: {response}")
        
    finally:
        await client.close()


async def test_long_input():
    """测试长输入的debug显示"""
    print("\n🔍 === 测试长输入的Debug显示 ===")
    
    client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True)
    
    try:
        # 构造一个很长的输入
        long_content = """
这是一个非常长的输入消息，用来测试debug模式如何处理长文本。
我们在这里包含了大量的信息，包括：
1. 详细的任务描述
2. 复杂的上下文信息
3. 多个示例和说明
4. 技术细节和规范
5. 期望的输出格式

在实际使用中，用户可能会发送非常长的prompt，包含：
- 完整的文档内容
- 代码片段和示例
- 详细的指令和要求
- 历史对话记录
- 复杂的数据结构

Debug模式应该能够完整地显示这些信息，同时保持日志的可读性。
长文本应该既显示预览版本（前几百个字符），也显示完整版本，
这样开发者可以根据需要查看所需的详细程度。

这个测试用例就是为了验证这种长文本的处理能力。
""".strip()
        
        messages = [
            {"role": "system", "content": "你是一个有用的AI助手，专门处理长文本分析任务。"},
            {"role": "user", "content": long_content},
            {"role": "user", "content": "请总结上述内容的要点。"}
        ]
        
        print("📤 发送长消息...")
        response = await client.generate(messages, max_tokens=200, temperature=0.3)
        print(f"✅ 收到响应: {response}")
        
    finally:
        await client.close()


async def test_multi_turn_conversation():
    """测试多轮对话的debug显示"""
    print("\n🔍 === 测试多轮对话的Debug显示 ===")
    
    client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True)
    
    try:
        # 构建一个复杂的多轮对话
        conversation = [
            {"role": "system", "content": "你是一个Python编程专家和教学助手。"},
            {"role": "user", "content": "我想学习Python的装饰器，能给我解释一下吗？"},
            {"role": "assistant", "content": "当然！装饰器是Python中的一个强大特性。简单来说，装饰器是一个函数，它接受另一个函数作为参数，并扩展或修改其行为，而不显式修改函数本身。"},
            {"role": "user", "content": "能给我一个具体的例子吗？比如计时装饰器？"},
            {"role": "assistant", "content": "好的！这是一个计时装饰器的例子：\n\n```python\nimport time\nimport functools\n\ndef timer(func):\n    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f'{func.__name__} took {end - start:.4f} seconds')\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    time.sleep(1)\n    return 'Done'\n```"},
            {"role": "user", "content": "这个例子很好！但是我还想了解带参数的装饰器，能再给我一个例子吗？"}
        ]
        
        print("📤 发送多轮对话...")
        response = await client.generate(conversation, max_tokens=300, temperature=0.4)
        print(f"✅ 收到响应: {response}")
        
    finally:
        await client.close()


async def test_different_parameters():
    """测试不同参数的debug显示"""
    print("\n🔍 === 测试不同参数的Debug显示 ===")
    
    client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True)
    
    try:
        messages = [{"role": "user", "content": "请生成一个创意的故事开头。"}]
        
        # 测试不同参数组合
        test_cases = [
            {"max_tokens": 30, "temperature": 0.1, "description": "低创造性，短输出"},
            {"max_tokens": 100, "temperature": 0.8, "description": "高创造性，中等输出"},
            {"max_tokens": 200, "temperature": 1.0, "description": "最高创造性，长输出"},
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 测试 {i}: {test_case['description']}")
            print(f"   参数: max_tokens={test_case['max_tokens']}, temperature={test_case['temperature']}")
            
            response = await client.generate(
                messages, 
                max_tokens=test_case['max_tokens'], 
                temperature=test_case['temperature']
            )
            print(f"📝 结果: {response}")
            
    finally:
        await client.close()


async def main():
    """主函数"""
    print("🔍 Enhanced Debug模式测试")
    print("=" * 80)
    print("""
📋 本测试将验证增强的debug模式功能：

1. ✅ 完整显示输入信息
   - 消息的role和content
   - 内容长度统计
   - 长文本的预览和完整显示

2. ✅ 详细显示API参数
   - 模型名称
   - max_tokens和temperature
   - 请求元数据

3. ✅ 完整显示输出信息
   - Token使用统计
   - 响应内容长度
   - 完整响应内容

4. ✅ 清晰的格式化显示
   - 使用分隔线和emoji
   - 结构化的信息布局
   - 易于阅读的格式
    """)
    
    # 设置debug日志
    setup_debug_logging()
    
    try:
        # 运行所有测试
        await test_short_input()
        await test_long_input() 
        await test_multi_turn_conversation()
        await test_different_parameters()
        
        print("\n" + "=" * 80)
        print("🎉 Enhanced Debug模式测试完成！")
        print("""
💡 Debug模式的改进：
✅ 输入信息完整显示 - 包括每条消息的role、content和长度
✅ 长文本智能处理 - 既显示预览也显示完整内容
✅ 结构化日志格式 - 使用分隔线和emoji便于阅读
✅ 详细的Token统计 - 精确的使用情况分析
✅ 响应内容完整记录 - 支持长响应的完整显示

🔧 适用场景：
- API调用问题排查
- Token使用优化分析
- 长文本处理验证
- 多轮对话调试
- 参数效果对比
        """)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 检查依赖
    try:
        import openai
    except ImportError:
        print("❌ 需要安装openai库: pip install openai")
        sys.exit(1)
    
    print("🔍 Enhanced Debug模式 - 完整输入输出显示测试")
    print("该示例将展示改进后的debug模式如何完整显示所有输入和输出信息")
    
    asyncio.run(main())