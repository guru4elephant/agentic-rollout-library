#!/usr/bin/env python3
"""
Test concurrent LLM API calls to diagnose concurrency issues.
"""

import asyncio
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import create_openai_api_handle_async


async def test_single_llm_call(llm_handle, call_id: int):
    """Test a single LLM API call."""
    start_time = time.time()

    print(f"[{call_id}] Starting at {time.strftime('%H:%M:%S')}")

    try:
        messages = [{"role": "user", "content": f"Say hello {call_id}"}]
        response = await llm_handle(messages)

        duration = time.time() - start_time
        print(f"[{call_id}] ✅ Success in {duration:.2f}s: {response['content'][:50]}...")
        return {"id": call_id, "status": "success", "duration": duration}

    except Exception as e:
        duration = time.time() - start_time
        print(f"[{call_id}] ❌ Error in {duration:.2f}s: {str(e)}")
        return {"id": call_id, "status": "error", "duration": duration, "error": str(e)}


async def test_concurrent_llm(num_concurrent: int = 5):
    """Test concurrent LLM API calls."""

    print(f"\n{'='*60}")
    print(f"Testing {num_concurrent} concurrent LLM API calls")
    print(f"{'='*60}\n")

    # Create LLM handle
    llm_handle = create_openai_api_handle_async(
        base_url="http://211.23.3.237:27544/v1",
        api_key="sk-qq7xJtnAdB1Gv6IkHTQhDAPuUAT700vF3CMmGinILsmP2HuY",
        model="deepseek-v3-1-terminus"
    )

    # Create tasks
    start_time = time.time()
    tasks = [test_single_llm_call(llm_handle, i) for i in range(num_concurrent)]

    # Run concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_duration = time.time() - start_time

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    success = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    errors = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "error")
    exceptions = sum(1 for r in results if isinstance(r, Exception))

    print(f"✅ Success: {success}")
    print(f"❌ Errors: {errors}")
    print(f"🔥 Exceptions: {exceptions}")
    print(f"⏱️  Total time: {total_duration:.2f}s")

    if success > 0:
        avg_duration = sum(r["duration"] for r in results if isinstance(r, dict) and r.get("status") == "success") / success
        print(f"📊 Average call duration: {avg_duration:.2f}s")
        print(f"🚀 Speedup: {avg_duration * num_concurrent / total_duration:.2f}x")

    # Detail
    print(f"\n{'='*60}")
    print(f"DETAILED RESULTS")
    print(f"{'='*60}")
    for i, result in enumerate(results):
        if isinstance(result, dict):
            status = "✅" if result["status"] == "success" else "❌"
            print(f"{status} Call {i}: {result['status']} in {result['duration']:.2f}s")
        else:
            print(f"🔥 Call {i}: Exception - {str(result)}")


async def test_sequential_llm(num_calls: int = 5):
    """Test sequential LLM API calls for baseline."""

    print(f"\n{'='*60}")
    print(f"Testing {num_calls} sequential LLM API calls (baseline)")
    print(f"{'='*60}\n")

    # Create LLM handle
    llm_handle = create_openai_api_handle_async(
        base_url="http://211.23.3.237:27544/v1",
        api_key="sk-qq7xJtnAdB1Gv6IkHTQhDAPuUAT700vF3CMmGinILsmP2HuY",
        model="deepseek-v3-1-terminus"
    )

    start_time = time.time()
    results = []

    for i in range(num_calls):
        result = await test_single_llm_call(llm_handle, i)
        results.append(result)

    total_duration = time.time() - start_time

    print(f"\n⏱️  Total sequential time: {total_duration:.2f}s")
    avg_duration = total_duration / num_calls
    print(f"📊 Average call duration: {avg_duration:.2f}s")


async def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM API concurrency")
    parser.add_argument("--concurrent", type=int, default=5, help="Number of concurrent calls")
    parser.add_argument("--sequential", action="store_true", help="Test sequential calls (baseline)")
    args = parser.parse_args()

    if args.sequential:
        await test_sequential_llm(args.concurrent)
    else:
        await test_concurrent_llm(args.concurrent)


if __name__ == "__main__":
    asyncio.run(main())
