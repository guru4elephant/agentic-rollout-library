#!/usr/bin/env python3
"""
LLMAPIClient Debug模式示例
展示如何使用debug模式查看API调用的详细输入和输出
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


def setup_logging(debug: bool = False):
    """设置日志级别"""
    # 重新配置日志，强制刷新
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            force=True
        )
        # 确保workers.utils.llm_client的日志级别也是DEBUG
        logging.getLogger('workers.utils.llm_client').setLevel(logging.DEBUG)
        print("🐛 Debug模式已启用 - 将显示详细的API调用日志")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s - %(message)s',
            force=True
        )
        print("ℹ️  标准模式 - 只显示基本信息")


async def compare_debug_modes():
    """对比debug模式开启和关闭的效果"""
    
    print("=" * 80)
    print("🔍 LLMAPIClient Debug模式对比示例")
    print("=" * 80)
    
    # 测试消息
    test_messages = [
        {"role": "system", "content": "你是一个有用的AI助手。请简洁地回答问题。"},
        {"role": "user", "content": "请用一句话解释什么是人工智能？"}
    ]
    
    print("\n🚫 === 标准模式（debug=False）===")
    setup_logging(debug=False)
    
    # 创建标准模式客户端
    normal_client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=False)
    
    try:
        print("\n📤 调用API...")
        response1 = await normal_client.generate(test_messages, max_tokens=100, temperature=0.5)
        print(f"\n✅ 响应: {response1}")
        
    finally:
        await normal_client.close()
    
    print("\n" + "="*80)
    print("\n🐛 === Debug模式（debug=True）===")
    setup_logging(debug=True)
    
    # 创建debug模式客户端
    debug_client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True)
    
    try:
        print("\n📤 调用API...")
        response2 = await debug_client.generate(test_messages, max_tokens=100, temperature=0.5)
        print(f"\n✅ 响应: {response2}")
        
    finally:
        await debug_client.close()


async def debug_with_long_conversation():
    """使用debug模式处理长对话"""
    
    print("\n" + "="*80)
    print("📚 Debug模式 - 长对话示例")
    print("="*80)
    
    setup_logging(debug=True)
    
    # 创建debug客户端
    client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True)
    
    try:
        # 构建一个多轮对话
        conversation = [
            {"role": "system", "content": "你是一个Python编程专家。"},
            {"role": "user", "content": "我想学习Python列表推导式。"},
            {"role": "assistant", "content": "列表推导式是Python中创建列表的简洁方式。基本语法是：[expression for item in iterable if condition]"},
            {"role": "user", "content": "能给我一个具体的例子吗？"},
        ]
        
        print("\n🎯 发送多轮对话...")
        response = await client.generate(conversation, max_tokens=150, temperature=0.3)
        print(f"\n✅ 最终响应: {response}")
        
    finally:
        await client.close()


async def debug_with_different_parameters():
    """使用debug模式测试不同参数"""
    
    print("\n" + "="*80)
    print("⚙️  Debug模式 - 不同参数测试")
    print("="*80)
    
    setup_logging(debug=True)
    
    client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True)
    
    try:
        messages = [{"role": "user", "content": "用三个词描述春天"}]
        
        # 测试不同的参数组合
        params_list = [
            {"max_tokens": 20, "temperature": 0.1},
            {"max_tokens": 50, "temperature": 0.7},
            {"max_tokens": 80, "temperature": 1.0},
        ]
        
        for i, params in enumerate(params_list, 1):
            print(f"\n🧪 测试 {i}: max_tokens={params['max_tokens']}, temperature={params['temperature']}")
            response = await client.generate(messages, **params)
            print(f"📝 结果: {response}")
            
    finally:
        await client.close()


async def main():
    """主函数"""
    print("""
🔍 LLMAPIClient Debug模式使用说明:

Debug模式的作用：
1. 📋 记录每次API请求的详细参数
2. 📤 显示发送给LLM的完整消息内容  
3. 📥 记录LLM返回的完整响应内容
4. 📊 显示token使用情况和响应元数据
5. 🔧 帮助调试API调用问题

使用方法：
```python
# 启用debug模式
client = create_llm_client(api_key, base_url, model, debug=True)

# 或使用工厂函数
client = create_llm_client(api_key, base_url, model, debug=True)
```

注意：debug模式会产生大量日志，建议只在开发和调试时使用。
    """)
    
    try:
        # 运行所有示例
        await compare_debug_modes()
        await debug_with_long_conversation()
        await debug_with_different_parameters()
        
        print("\n" + "="*80)
        print("🎉 Debug模式演示完成！")
        print("""
💡 Debug模式的优势：
- 完整记录API交互过程
- 便于排查API调用问题
- 监控token使用情况
- 分析响应质量和延迟

⚠️  生产环境建议关闭debug模式以减少日志量。
        """)
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 检查依赖
    try:
        import openai
    except ImportError:
        print("❌ 需要安装openai库: pip install openai")
        sys.exit(1)
    
    print("🔍 LLMAPIClient Debug模式演示")
    print("该示例将展示debug模式下的详细API调用日志")
    
    asyncio.run(main())
