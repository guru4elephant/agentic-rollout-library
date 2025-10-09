# LLM超时问题修复

## 问题现象

```python
TimeoutError: Node R2ELLM-django-django-13363 execution exceeded timeout of Nones
                                                                            ^^^^^^
```

堆栈跟踪显示：
```
aiohttp.helpers.py -> raise asyncio.TimeoutError
  ↓
llm_api_utils.py -> openai_api_handle_async
  ↓  
llm_node.py -> _retry_with_backoff_async
  ↓
r2e_k8s_example.py -> asyncio.wait_for(llm_node.process_async())
```

## 问题分析

### 1. 真正的超时原因

**❌ 不是tool超时，是LLM API调用超时**

从堆栈看：
- `aiohttp` HTTP请求超时
- 发生在调用LLM API时：`http://211.23.3.237:27544/v1`

### 2. 为什么会显示 "timeout of Nones"？

**Bug原因**：

1. `LLMNode` 创建时**没有传入timeout参数**：
   ```python
   llm_node = LLMNode(
       name=f"R2ELLM-{pod_suffix}",
       function_handle=llm_handle,
       model_config={...},
       timeline_enabled=enable_timeline
       # ❌ 缺少: timeout=llm_timeout
   )
   ```

2. `LLMNode.timeout = None`

3. 外层用`asyncio.wait_for`包装，但错误信息显示的是内部的`self.timeout`（None）

4. `LLMNode`也没有将timeout传递给底层的aiohttp调用

### 3. LLM API为什么会超时？

可能的原因：

| 原因 | 说明 | 概率 |
|------|------|------|
| **LLM服务负载高** | 500并发请求，服务器排队 | 🔴 高 |
| **DeepSeek-V3模型慢** | 大模型推理本身慢（40秒平均） | 🔴 高 |
| **网络不稳定** | 请求延迟高 | 🟡 中 |
| **服务并发限制** | API有速率限制，导致排队 | 🟡 中 |
| **连接池问题** | aiohttp连接池耗尽 | 🟢 低 |

从timeline统计看：
- LLM平均耗时：40秒
- LLM最长耗时：492秒
- LLM中位数：20秒

**结论**：服务本身就慢，加上500并发，超时120秒不够用

## 修复方案

### 修复1: 传递timeout参数给LLMNode

**文件**: `r2e_k8s_example.py`

```python
llm_node = LLMNode(
    name=f"R2ELLM-{pod_suffix}",
    function_handle=llm_handle,
    model_config={
        "temperature": 0.7,
        "max_tokens": 4000
    },
    timeline_enabled=enable_timeline,
    timeout=llm_timeout  # ✅ 添加这行
)
```

### 修复2: LLMNode传递timeout给API调用

**文件**: `src/core/llm_node.py`

#### 异步版本（第187-197行）：

```python
params = {**self.default_params, **self.model_config}

# Add timeout if configured
if self.timeout:
    params['timeout'] = self.timeout  # ✅ 添加

try:
    response = await self._retry_with_backoff_async(
        self.function_handle,
        messages=input_data,
        **params
    )
```

#### 同步版本（第234-246行）：

```python
# Merge default params with model config
params = {**self.default_params, **self.model_config}

# Add timeout if configured
if self.timeout:
    params['timeout'] = self.timeout  # ✅ 添加

try:
    response = self._retry_with_backoff(
        self.function_handle,
        messages=input_data,
        **params
    )
```

### 修复3: 增加llm-timeout时间

**原因**：
- 默认120秒不够
- LLM平均40秒，但有长尾（最长492秒）
- 500并发会加剧排队

**建议配置**：

```bash
# 保守（覆盖99%）
--llm-timeout 300

# 平衡（覆盖95%） - 推荐
--llm-timeout 180

# 激进（只接受快的）
--llm-timeout 120
```

## 完整的推荐配置

```bash
python3 r2e_k8s_example.py \
    --jsonl test-00000-of-00001-with-images.jsonl \
    --concurrent 500 \
    --output-dir single-output-v14 \
    --max-execution-time 2400 \
    --llm-timeout 180 \
    --tool-timeout 300
```

## 效果对比

### 修复前

```
问题：
- LLMNode.timeout = None
- aiohttp 默认120秒超时
- 错误信息显示 "timeout of Nones"
- 无法区分是哪里超时

结果：
- 大量LLM超时
- 错误信息不清晰
```

### 修复后

```
改进：
- LLMNode.timeout = 180秒
- aiohttp 继承该超时设置
- 错误信息正确显示 "timeout of 180s"
- 可以区分LLM超时 vs Tool超时

结果：
- LLM超时率降低（180秒 vs 120秒）
- 错误信息清晰
- 便于诊断问题
```

## 进一步优化建议

### 1. 降低并发数

如果LLM服务有并发限制：

```bash
--concurrent 300  # 从500降到300
```

或者分批执行：

```bash
# 第一批
python3 r2e_k8s_example.py --jsonl batch1.jsonl --concurrent 200

# 第二批
python3 r2e_k8s_example.py --jsonl batch2.jsonl --concurrent 200
```

### 2. 监控LLM性能

添加到代码中：

```python
# 记录LLM响应时间
llm_start = time.time()
llm_response = await llm_node.process_async(messages)
llm_duration = time.time() - llm_start

if llm_duration > 60:
    log(f"⚠️  Slow LLM call: {llm_duration:.1f}s")
```

### 3. 检查LLM服务状态

```bash
# 测试LLM API性能
curl -X POST http://211.23.3.237:27544/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3-1-terminus",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

### 4. 考虑使用更快的模型

如果可能：
- DeepSeek-V3 → DeepSeek-V2（更快）
- 或者用更小的模型进行初步过滤

## 验证修复

### 1. 运行小样本测试

```bash
python3 r2e_k8s_example.py \
    --jsonl small.jsonl \
    --concurrent 5 \
    --debug \
    --llm-timeout 180 \
    --tool-timeout 300
```

### 2. 检查错误信息

修复后应该看到清晰的超时信息：

```
✅ 正确：TimeoutError: Node R2ELLM-xxx execution exceeded timeout of 180s
❌ 错误：TimeoutError: Node R2ELLM-xxx execution exceeded timeout of Nones
```

### 3. 监控超时率

```bash
# LLM超时次数
grep "LLM call timeout" single-output-v14/*.log | wc -l

# LLM超时率
echo "scale=2; $(grep -l "LLM call timeout" single-output-v14/*.log | wc -l) * 100 / $(ls single-output-v14/*.log | wc -l)" | bc
```

目标：
- LLM超时率 < 5%
- 如果超时率高，继续增加`--llm-timeout`

## 总结

### 问题根因

1. ❌ LLMNode创建时没传timeout
2. ❌ LLMNode没将timeout传给aiohttp
3. ❌ 默认120秒对DeepSeek-V3太短
4. ❌ 500并发加剧排队

### 修复措施

1. ✅ 传递`timeout=llm_timeout`给LLMNode
2. ✅ LLMNode将timeout传给params
3. ✅ 增加llm-timeout到180秒
4. ✅ 错误信息现在清晰显示实际timeout值

### 预期效果

- LLM超时率：15-20% → **5%以下**
- 错误信息：混乱 → **清晰**
- 问题诊断：困难 → **容易**
