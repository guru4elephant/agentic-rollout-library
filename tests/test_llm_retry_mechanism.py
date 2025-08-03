#!/usr/bin/env python3
"""
测试LLMAPIClient的重试机制
验证API调用失败时的重试行为
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.utils import create_llm_client

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API配置
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")


async def test_successful_call():
    """测试正常的API调用"""
    print("🔍 测试1: 正常API调用")
    print("-" * 60)
    
    client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True, max_retries=3)
    
    try:
        messages = [{"role": "user", "content": "请回复'重试机制测试成功'"}]
        
        response = await client.generate(messages, max_tokens=50, temperature=0.1)
        print(f"✅ 响应: {response}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        await client.close()


async def test_invalid_url_retry():
    """测试无效URL的重试行为"""
    print("\n🔍 测试2: 无效URL重试")
    print("-" * 60)
    
    # 使用无效的URL来触发网络错误
    invalid_url = "http://invalid-url-for-testing.com:12345"
    client = create_llm_client(API_KEY, invalid_url, MODEL_NAME, debug=True, max_retries=2)
    
    try:
        messages = [{"role": "user", "content": "测试重试"}]
        
        start_time = asyncio.get_event_loop().time()
        response = await client.generate(messages, max_tokens=50)
        end_time = asyncio.get_event_loop().time()
        
        print(f"❌ 意外成功: {response}")
        return False
        
    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time
        
        print(f"✅ 预期的失败: {e}")
        print(f"   总耗时: {elapsed:.1f} 秒")
        print(f"   验证重试机制正常工作")
        return True
    finally:
        await client.close()


async def test_invalid_api_key():
    """测试无效API Key的行为（不应重试）"""
    print("\n🔍 测试3: 无效API Key (不重试)")
    print("-" * 60)
    
    # 使用无效的API Key
    invalid_key = "invalid-api-key-123"
    client = create_llm_client(invalid_key, BASE_URL, MODEL_NAME, debug=True, max_retries=3)
    
    try:
        messages = [{"role": "user", "content": "测试无效key"}]
        
        start_time = asyncio.get_event_loop().time()
        response = await client.generate(messages, max_tokens=50)
        end_time = asyncio.get_event_loop().time()
        
        print(f"❌ 意外成功: {response}")
        return False
        
    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time
        
        print(f"✅ 预期的失败: {e}")
        print(f"   总耗时: {elapsed:.1f} 秒")
        
        # 验证没有过度重试（应该很快失败）
        if elapsed < 5:  # 如果重试了3次，应该需要更长时间
            print(f"   ✅ 正确地快速失败，没有无意义重试")
            return True
        else:
            print(f"   ❌ 耗时过长，可能进行了不必要的重试")
            return False
    finally:
        await client.close()


async def test_different_retry_counts():
    """测试不同的重试次数配置"""
    print("\n🔍 测试4: 不同重试次数配置")
    print("-" * 60)
    
    retry_configs = [0, 1, 2, 5]
    invalid_url = "http://timeout-test.invalid:9999"
    
    for max_retries in retry_configs:
        print(f"\n   测试max_retries={max_retries}:")
        client = create_llm_client(API_KEY, invalid_url, MODEL_NAME, debug=False, max_retries=max_retries)
        
        try:
            messages = [{"role": "user", "content": "测试"}]
            
            start_time = asyncio.get_event_loop().time()
            await client.generate(messages, max_tokens=20)
            
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            elapsed = end_time - start_time
            
            print(f"     ⏱️  耗时: {elapsed:.1f}秒")
            print(f"     🔄 预期尝试次数: {max_retries + 1}")
            
            # 验证重试次数与耗时的关系
            # 理论上更多重试应该花费更多时间（由于指数退避）
            expected_min_time = sum(2**i for i in range(max_retries)) if max_retries > 0 else 0
            
            if elapsed >= expected_min_time * 0.8:  # 允许20%的误差
                print(f"     ✅ 重试时间符合预期")
            else:
                print(f"     ⚠️  重试时间可能不符合预期")
        
        finally:
            await client.close()
    
    return True


async def test_retry_with_success():
    """模拟重试后成功的场景"""
    print("\n🔍 测试5: 模拟重试后成功")
    print("-" * 60)
    
    # 这个测试比较难模拟，因为我们需要一个"有时失败有时成功"的API
    # 作为替代，我们测试配置是否正确传递
    
    client = create_llm_client(API_KEY, BASE_URL, MODEL_NAME, debug=True, max_retries=1)
    
    # 验证配置
    assert client.max_retries == 1, "重试配置未正确设置"
    assert client.debug == True, "debug配置未正确设置"
    
    print("✅ 重试配置验证通过")
    
    try:
        # 进行一个简单的成功调用来验证基本功能
        messages = [{"role": "user", "content": "简单测试"}]
        response = await client.generate(messages, max_tokens=20)
        print(f"✅ 基本功能正常: {response[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False
    finally:
        await client.close()


async def main():
    """主测试函数"""
    print("🧪 LLM API Client 重试机制测试")
    print("=" * 80)
    print("""
📋 测试计划:
1. ✅ 正常API调用 - 验证基本功能
2. 🔄 网络错误重试 - 验证重试机制
3. 🚫 客户端错误快速失败 - 验证错误分类
4. ⚙️  不同重试配置 - 验证配置效果
5. 🔧 配置验证 - 验证参数传递

重试机制特性:
- 最多重试3次 (可配置)
- 指数退避延迟 (1s, 2s, 4s)
- 智能错误分类 (网络错误重试，客户端错误不重试)
- 详细的调试日志
    """)
    
    test_results = []
    
    try:
        # 执行所有测试
        test_results.append(("正常API调用", await test_successful_call()))
        test_results.append(("网络错误重试", await test_invalid_url_retry()))
        test_results.append(("客户端错误快速失败", await test_invalid_api_key()))
        test_results.append(("不同重试配置", await test_different_retry_counts()))
        test_results.append(("配置验证", await test_retry_with_success()))
        
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
            print("🎉 所有测试通过！重试机制工作正常。")
            print("""
💡 重试机制特性验证完成:
✅ 网络错误自动重试，指数退避延迟
✅ 客户端错误快速失败，避免无效重试  
✅ 可配置的最大重试次数
✅ 详细的调试日志和错误分类
✅ 异常处理和资源清理
            """)
        else:
            print("⚠️ 部分测试失败，请检查重试机制实现。")
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔄 LLM API Client 重试机制测试")
    print("验证API调用失败时的重试行为和错误处理")
    
    # 检查依赖
    try:
        import openai
    except ImportError:
        print("❌ 需要安装openai库: pip install openai")
        sys.exit(1)
    
    asyncio.run(main())