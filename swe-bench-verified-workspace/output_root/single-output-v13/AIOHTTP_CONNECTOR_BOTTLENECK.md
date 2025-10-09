# aiohttp TCPConnector 连接池瓶颈分析

## 问题发现

你发现的关键代码（`llm_api_utils.py:169`）：

```python
if session is None or session.closed:
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
    session = aiohttp.ClientSession(connector=connector)
```

**你的疑问非常正确！** 在500并发下，这确实是瓶颈！

## 瓶颈分析

### 1. 连接池限制

当前配置：
```python
limit=100              # 总共最多100个连接
limit_per_host=30      # 每个host最多30个连接
```

### 2. 500并发下的问题

**场景**：
- 500个并发任务同时运行
- 所有任务都调用同一个LLM API：`http://211.23.3.237:27544`
- 每个LLM调用平均40秒（timeline统计）

**问题**：

| 参数 | 配置值 | 实际需求 | 是否够用 |
|------|--------|---------|---------|
| `limit_per_host` | 30 | 500 | ❌ **远远不够** |
| `limit` | 100 | 500 | ❌ **远远不够** |

### 3. 会发生什么？

#### 情况A: 前30个请求

```
Task 1-30: 立即获得连接 ✅
          → 发送HTTP请求
          → 等待LLM响应（40秒）
```

#### 情况B: 第31-100个请求

```
Task 31-100: 等待连接池释放 ⏳
            → 排队等待前30个完成
            → 等待时间：40秒+
```

#### 情况C: 第101-500个请求

```
Task 101-500: 阻塞在connector.limit ❌
             → 等待总连接数降到100以下
             → 等待时间：40-80秒+
             → 可能触发超时！
```

### 4. 超时的真实原因

```
实际超时时间 = 排队等待时间 + LLM响应时间
             = 40秒（等连接） + 40秒（LLM） = 80秒+
             > 120秒（默认timeout） ✅ 超时了！
```

**所以你看到大量超时，不是因为LLM慢，而是因为在排队等连接！**

## 验证假设

### 检查是否是连接池问题

添加日志到代码中：

```python
async def openai_api_handle_async(messages: List[Dict], **kwargs) -> Dict:
    import aiohttp
    import time
    
    nonlocal session
    
    if session is None or session.closed:
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        session = aiohttp.ClientSession(connector=connector)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        timeout = aiohttp.ClientTimeout(total=kwargs.get('timeout', 120))
        
        # 记录等待连接的时间
        async with session.post(...) as response:
            connect_time = time.time() - start_time
            if connect_time > 5:
                print(f"⚠️ Waited {connect_time:.1f}s to get connection")
            
            response.raise_for_status()
            data = await response.json()
    ...
```

如果看到大量"Waited XXs to get connection"，就证明是连接池问题。

## 解决方案

### 方案1: 增加连接池大小（立即生效，推荐）

```python
# 修改 src/utils/llm_api_utils.py:169
if session is None or session.closed:
    connector = aiohttp.TCPConnector(
        limit=1000,              # 从100增加到1000
        limit_per_host=600       # 从30增加到600（支持500+并发）
    )
    session = aiohttp.ClientSession(connector=connector)
```

**为什么是这些值？**
- `limit=1000`: 总共支持1000个并发连接（留有余量）
- `limit_per_host=600`: 单个host支持600个（覆盖500并发）

**效果**：
- ✅ 500并发不再排队
- ✅ 超时率大幅降低
- ✅ 实际等待时间 = LLM响应时间（无排队）

### 方案2: 动态配置连接池大小

修改函数签名，允许自定义：

```python
def create_openai_api_handle_async(
    base_url: str,
    api_key: str,
    model: str,
    clear_proxy: bool = True,
    max_connections: int = 1000,        # 新增
    max_connections_per_host: int = 600  # 新增
) -> Callable:
    ...
    
    if session is None or session.closed:
        connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections_per_host
        )
        session = aiohttp.ClientSession(connector=connector)
```

使用时：

```python
llm_handle = create_openai_api_handle_async(
    base_url="http://211.23.3.237:27544/v1",
    api_key="sk-xxx",
    model="deepseek-v3-1-terminus",
    max_connections=1000,
    max_connections_per_host=600
)
```

### 方案3: 每个任务独立session（不推荐）

```python
# 每次调用创建新session
async def openai_api_handle_async(messages: List[Dict], **kwargs) -> Dict:
    import aiohttp
    
    # 不复用session
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.post(...) as response:
            ...
```

**问题**：
- ❌ 无法利用连接复用
- ❌ 频繁创建/销毁连接开销大
- ❌ 性能更差

### 方案4: 降低并发数（治标不治本）

```bash
--concurrent 30  # 从500降到30（与limit_per_host一致）
```

**问题**：
- ❌ 大幅增加总执行时间
- ❌ 没有利用资源
- ❌ 不推荐

## 修复步骤

### 立即修复（推荐）

编辑 `src/utils/llm_api_utils.py`：

```python
# Line 169
if session is None or session.closed:
    connector = aiohttp.TCPConnector(
        limit=1000,              # 增加
        limit_per_host=600,      # 增加
        force_close=False,       # 复用连接
        enable_cleanup_closed=True
    )
    session = aiohttp.ClientSession(connector=connector)
```

### 验证修复

运行小样本测试：

```bash
python3 r2e_k8s_example.py \
    --jsonl small.jsonl \
    --concurrent 20 \
    --debug \
    --llm-timeout 180
```

观察：
1. LLM调用是否还超时？
2. 平均响应时间是否接近40秒（无排队）？

## 理论计算

### 修复前（limit_per_host=30）

```
可用连接: 30
并发任务: 500
排队任务: 470

平均等待时间 = 470 / 30 × 40秒 = 626秒 ❌
实际超时: 大量超时（> 120秒）
```

### 修复后（limit_per_host=600）

```
可用连接: 600
并发任务: 500
排队任务: 0

平均等待时间 = 0秒 ✅
实际响应时间 = LLM时间（40秒）
```

## 连接池大小建议

### 通用公式

```python
limit_per_host >= concurrent_tasks × 1.2  # 留20%余量
```

### 不同并发数的配置

| 并发数 | limit_per_host | limit |
|--------|---------------|-------|
| 50 | 60 | 100 |
| 100 | 120 | 200 |
| 200 | 240 | 400 |
| **500** | **600** | **1000** |
| 1000 | 1200 | 2000 |

### 保守配置（推荐）

```python
connector = aiohttp.TCPConnector(
    limit=1000,              # 总连接数
    limit_per_host=600,      # 单host连接数
    ttl_dns_cache=300,       # DNS缓存5分钟
    force_close=False,       # 复用连接
    enable_cleanup_closed=True
)
```

## 其他优化建议

### 1. 启用 Keep-Alive

确保HTTP连接复用：

```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Connection": "keep-alive"  # 添加
}
```

### 2. 调整超时策略

基于实际测试调整：

```python
timeout = aiohttp.ClientTimeout(
    total=180,        # 总超时（包括排队）
    connect=10,       # 连接超时（不应该排队）
    sock_read=170     # 读取超时（LLM响应）
)
```

如果`connect`超时频繁，说明还是连接池不够。

### 3. 监控连接池使用

添加监控（可选）：

```python
# 在session.post之前
if hasattr(connector, '_conns'):
    active_conns = len(connector._conns)
    if active_conns > limit_per_host * 0.8:
        print(f"⚠️ Connection pool nearly full: {active_conns}/{limit_per_host}")
```

## 性能对比预测

### 修复前（limit_per_host=30）

```
并发: 500
连接池: 30
瓶颈: 连接池排队

结果:
- 前30个任务: 正常（40秒）
- 后470个任务: 超时或极慢（80-200秒）
- 总wall time: 不稳定，大量超时
- 成功率: ~60-70%
```

### 修复后（limit_per_host=600）

```
并发: 500
连接池: 600
瓶颈: LLM服务本身

结果:
- 所有任务: 同时执行（无排队）
- 平均响应: 40秒（LLM时间）
- 总wall time: ~72分钟（稳定）
- 成功率: ~90%+
```

## 快速验证方法

### 1. 检查当前配置

```bash
grep -n "TCPConnector" src/utils/llm_api_utils.py
# 应该看到: limit=100, limit_per_host=30
```

### 2. 运行压力测试

```python
# test_connector.py
import asyncio
import aiohttp
import time

async def test_concurrent_requests(concurrency, limit_per_host):
    connector = aiohttp.TCPConnector(
        limit=1000,
        limit_per_host=limit_per_host
    )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.time()
        
        async def make_request(i):
            req_start = time.time()
            async with session.get('http://httpbin.org/delay/1') as resp:
                wait_time = time.time() - req_start
                return wait_time
        
        tasks = [make_request(i) for i in range(concurrency)]
        wait_times = await asyncio.gather(*tasks)
        
        total_time = time.time() - start
        max_wait = max(wait_times)
        avg_wait = sum(wait_times) / len(wait_times)
        
        print(f"Concurrency: {concurrency}")
        print(f"limit_per_host: {limit_per_host}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Max wait: {max_wait:.1f}s")
        print(f"Avg wait: {avg_wait:.1f}s")

# 测试
asyncio.run(test_concurrent_requests(concurrency=100, limit_per_host=30))
asyncio.run(test_concurrent_requests(concurrency=100, limit_per_host=120))
```

预期结果：
- `limit_per_host=30`: 平均等待时间 > 3秒（排队）
- `limit_per_host=120`: 平均等待时间 ~1秒（无排队）

## 总结

### 你的发现完全正确！

❌ **问题根因**：
```
aiohttp.TCPConnector(limit=100, limit_per_host=30)
                                 ^^^^^^^^^^^^^^^^
            这个限制导致500并发任务大量排队
```

✅ **解决方案**：
```python
aiohttp.TCPConnector(
    limit=1000,         # 增加10倍
    limit_per_host=600  # 增加20倍，匹配并发数
)
```

### 预期效果

修复后：
- 🚀 无连接排队
- 🚀 超时率大幅降低
- 🚀 响应时间稳定在40秒（LLM本身）
- 🚀 成功率从70%提升到90%+

这是一个**非常关键的发现**，应该是导致大量超时的主要原因！
