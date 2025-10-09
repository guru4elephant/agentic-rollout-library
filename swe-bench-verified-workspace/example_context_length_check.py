#!/usr/bin/env python3
"""
上下文长度检查使用示例

演示如何使用 context_length_checker 工具来处理长上下文
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import create_openai_api_handle_async
from utils.context_length_checker import (
    is_context_length_error_simple,
    parse_context_length_error,
    check_and_raise_if_context_error,
    ContextLengthExceededError,
    truncate_messages_to_fit,
    estimate_token_count,
    get_context_length_advice
)


async def example_1_simple_check():
    """示例1: 简单检查是否为长度错误"""
    print("\n" + "=" * 80)
    print("示例 1: 简单检查长度错误")
    print("=" * 80)
    
    llm_handle = create_openai_api_handle_async(
        base_url="http://211.23.3.237:27544/v1",
        api_key="sk-qq7xJtnAdB1Gv6IkHTQhDAPuUAT700vF3CMmGinILsmP2HuY",
        model="deepseek-v3-1-terminus"
    )
    
    # 创建超长消息
    long_text = "测试文本" * 50000  # 约 100k tokens
    messages = [{"role": "user", "content": long_text}]
    
    try:
        response = await llm_handle(messages=messages, max_tokens=100)
        print("✓ 请求成功")
    except Exception as e:
        if is_context_length_error_simple(e):
            print("✗ 检测到长度超限错误")
            print(f"  错误信息: {str(e)[:200]}...")
        else:
            print(f"✗ 其他错误: {type(e).__name__}")
            raise


async def example_2_detailed_parse():
    """示例2: 详细解析错误信息"""
    print("\n" + "=" * 80)
    print("示例 2: 详细解析错误信息")
    print("=" * 80)
    
    llm_handle = create_openai_api_handle_async(
        base_url="http://211.23.3.237:27544/v1",
        api_key="sk-qq7xJtnAdB1Gv6IkHTQhDAPuUAT700vF3CMmGinILsmP2HuY",
        model="deepseek-v3-1-terminus"
    )
    
    # 创建超长消息
    long_text = "这是一段测试文本。" * 30000  # 约 150k tokens
    messages = [{"role": "user", "content": long_text}]
    
    try:
        response = await llm_handle(messages=messages, max_tokens=100)
        print("✓ 请求成功")
    except Exception as e:
        is_length_err, details = parse_context_length_error(str(e))
        
        if is_length_err:
            print("✗ 长度超限错误详情:")
            print(f"  实际长度: {details.get('actual_length', 'N/A'):,} tokens")
            print(f"  最大限制: {details.get('max_length', 'N/A'):,} tokens")
            print(f"  超出长度: {details.get('overflow', 'N/A'):,} tokens")
            print(f"  超出比例: {details.get('overflow_ratio', 0):.1%}")
            
            # 提供建议
            if 'overflow_ratio' in details:
                advice = get_context_length_advice(details['overflow_ratio'])
                print(f"\n  {advice}")
        else:
            print(f"✗ 其他错误: {type(e).__name__}")


async def example_3_custom_exception():
    """示例3: 使用自定义异常"""
    print("\n" + "=" * 80)
    print("示例 3: 使用自定义异常")
    print("=" * 80)
    
    llm_handle = create_openai_api_handle_async(
        base_url="http://211.23.3.237:27544/v1",
        api_key="sk-qq7xJtnAdB1Gv6IkHTQhDAPuUAT700vF3CMmGinILsmP2HuY",
        model="deepseek-v3-1-terminus"
    )
    
    long_text = "测试" * 60000
    messages = [{"role": "user", "content": long_text}]
    
    try:
        response = await llm_handle(messages=messages, max_tokens=100)
        print("✓ 请求成功")
    except Exception as e:
        try:
            check_and_raise_if_context_error(e)
        except ContextLengthExceededError as length_err:
            print("✗ 捕获到自定义长度异常:")
            print(f"  类型: {type(length_err).__name__}")
            print(f"  实际长度: {length_err.actual_length:,} tokens")
            print(f"  最大限制: {length_err.max_length:,} tokens")
            print(f"  超出: {length_err.overflow:,} tokens")
            print(f"\n  异常信息: {str(length_err)}")


async def example_4_auto_truncate():
    """示例4: 自动截断消息"""
    print("\n" + "=" * 80)
    print("示例 4: 自动截断并重试")
    print("=" * 80)
    
    llm_handle = create_openai_api_handle_async(
        base_url="http://211.23.3.237:27544/v1",
        api_key="sk-qq7xJtnAdB1Gv6IkHTQhDAPuUAT700vF3CMmGinILsmP2HuY",
        model="deepseek-v3-1-terminus"
    )
    
    # 创建多条消息（模拟对话历史）
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "第一个问题" * 1000},
        {"role": "assistant", "content": "第一个回答" * 1000},
        {"role": "user", "content": "第二个问题" * 1000},
        {"role": "assistant", "content": "第二个回答" * 1000},
        {"role": "user", "content": "第三个问题" * 20000},  # 这条很长
    ]
    
    # 估算总 token 数
    total_tokens = sum(estimate_token_count(msg["content"]) for msg in messages)
    print(f"原始消息数: {len(messages)}")
    print(f"估算总 tokens: {total_tokens:,}")
    
    try:
        # 第一次尝试
        response = await llm_handle(messages=messages, max_tokens=100)
        print("✓ 第一次请求成功")
        
    except Exception as e:
        is_length_err, details = parse_context_length_error(str(e))
        
        if is_length_err:
            print("\n✗ 第一次请求失败（长度超限）")
            print(f"  需要减少约 {details.get('overflow', 0):,} tokens")
            
            # 自动截断
            max_allowed = details.get('max_length', 50000)
            truncated_messages = truncate_messages_to_fit(
                messages, 
                max_tokens=int(max_allowed * 0.9),  # 留 10% 余量
                truncate_strategy="from_middle"
            )
            
            truncated_tokens = sum(estimate_token_count(msg["content"]) for msg in truncated_messages)
            
            print(f"\n自动截断后:")
            print(f"  消息数: {len(messages)} → {len(truncated_messages)}")
            print(f"  估算 tokens: {total_tokens:,} → {truncated_tokens:,}")
            
            # 重试
            try:
                print("\n重试请求...")
                response = await llm_handle(messages=truncated_messages, max_tokens=100)
                print("✓ 第二次请求成功！")
                print(f"  响应: {response.get('content', '')[:100]}...")
            except Exception as retry_err:
                print(f"✗ 第二次请求仍然失败: {str(retry_err)[:100]}...")
        else:
            print(f"✗ 其他错误: {type(e).__name__}")


async def example_5_estimate_tokens():
    """示例5: 估算 token 数量"""
    print("\n" + "=" * 80)
    print("示例 5: 估算 token 数量")
    print("=" * 80)
    
    texts = [
        ("短文本", "你好，世界！"),
        ("中文段落", "这是一段中文测试文本。" * 100),
        ("英文段落", "This is an English test paragraph. " * 100),
        ("超长文本", "测试" * 10000),
    ]
    
    for name, text in texts:
        tokens = estimate_token_count(text)
        print(f"{name:10s}: {len(text):8,} 字符 → 约 {tokens:8,} tokens")


async def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("上下文长度检查工具使用示例")
    print("=" * 80)
    
    # 运行各个示例
    await example_5_estimate_tokens()
    await asyncio.sleep(1)
    
    await example_1_simple_check()
    await asyncio.sleep(1)
    
    await example_2_detailed_parse()
    await asyncio.sleep(1)
    
    await example_3_custom_exception()
    await asyncio.sleep(1)
    
    await example_4_auto_truncate()
    
    print("\n" + "=" * 80)
    print("所有示例运行完成")
    print("=" * 80)
    print("""
总结：
1. 使用 is_context_length_error_simple() 快速判断
2. 使用 parse_context_length_error() 获取详细信息
3. 使用 check_and_raise_if_context_error() 转换为自定义异常
4. 使用 truncate_messages_to_fit() 自动截断
5. 使用 estimate_token_count() 预估长度

更多文档请查看: LLM_LONG_CONTEXT_TEST_RESULTS.md
""")


if __name__ == "__main__":
    asyncio.run(main())
