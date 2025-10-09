# 超时参数调优指南

## 问题：大面积失败

### 原因分析

你的配置：
```bash
--max-execution-time 1200  # 20分钟
--tool-timeout 90          # 90秒
--llm-timeout 120          # 默认120秒
```

**问题**：
1. **Tool超时90秒太短**
   - Timeline显示：平均7秒，但P95可能在60-120秒
   - 复杂操作（如编辑大文件、搜索整个代码库）需要更长时间
   - 建议：至少180-300秒

2. **总超时20分钟偏短**
   - Timeline显示：中位数35.5分钟
   - 20分钟会导致50%以上样本超时
   - 建议：30-40分钟

## 推荐配置

### 方案1: 平衡配置（推荐）

**目标**: 过滤掉最慢的10%样本，保留90%正常样本

```bash
python3 r2e_k8s_example.py \
    --jsonl test-00000-of-00001-with-images.jsonl \
    --concurrent 500 \
    --timeline \
    --output-dir single-output-v14 \
    --max-execution-time 2400 \
    --llm-timeout 180 \
    --tool-timeout 300
```

**参数说明**：
- `--max-execution-time 2400` (40分钟)
  - 基于P90时间设置
  - 过滤掉最慢的10%样本
  - 预计wall time: ~40分钟

- `--llm-timeout 180` (3分钟)
  - 中位数20秒，平均40秒
  - 180秒覆盖99%的正常调用
  - 过滤492秒这种异常情况

- `--tool-timeout 300` (5分钟)
  - 平均7秒，但有长尾
  - 300秒覆盖绝大多数正常操作
  - 过滤2226秒这种异常情况

**预期结果**：
- 成功率: ~90%
- Wall time: ~40分钟（比72分钟快45%）
- 失败原因: 主要是复杂任务超时

### 方案2: 激进配置（追求速度）

**目标**: 只保留能快速完成的样本

```bash
python3 r2e_k8s_example.py \
    --jsonl test-00000-of-00001-with-images.jsonl \
    --concurrent 500 \
    --timeline \
    --output-dir single-output-v14 \
    --max-execution-time 1800 \
    --llm-timeout 120 \
    --tool-timeout 180
```

**参数说明**：
- `--max-execution-time 1800` (30分钟) - 基于中位数
- `--llm-timeout 120` (2分钟) - 过滤慢调用
- `--tool-timeout 180` (3分钟) - 过滤慢操作

**预期结果**：
- 成功率: ~50-60%
- Wall time: ~30分钟（比72分钟快58%）
- 失败原因: 超时居多

### 方案3: 保守配置（追求成功率）

**目标**: 尽量保留所有样本，只过滤真正卡住的

```bash
python3 r2e_k8s_example.py \
    --jsonl test-00000-of-00001-with-images.jsonl \
    --concurrent 500 \
    --timeline \
    --output-dir single-output-v14 \
    --max-execution-time 3600 \
    --llm-timeout 300 \
    --tool-timeout 600
```

**参数说明**：
- `--max-execution-time 3600` (60分钟) - 只过滤极端情况
- `--llm-timeout 300` (5分钟) - 只过滤真正卡住的
- `--tool-timeout 600` (10分钟) - 宽松限制

**预期结果**：
- 成功率: ~98%
- Wall time: ~60分钟（比72分钟快17%）
- 失败原因: 主要是真正的错误

## 参数设置原则

### 1. LLM超时设置

基于timeline统计：
- **中位数**: 20秒
- **平均值**: 40秒
- **P95**: ~120秒
- **P99**: ~180秒
- **异常值**: 492秒

**建议**：
- 保守: 300秒（保留99.9%）
- 平衡: 180秒（保留99%）
- 激进: 120秒（保留95%）

### 2. Tool超时设置

基于timeline统计：
- **中位数**: 1秒
- **平均值**: 7秒
- **P95**: ~60秒
- **P99**: ~120秒
- **异常值**: 2226秒

**建议**：
- 保守: 600秒（10分钟）
- 平衡: 300秒（5分钟）
- 激进: 180秒（3分钟）

### 3. 总超时设置

基于样本耗时分布：
- **中位数**: 35.5分钟
- **平均值**: 35.8分钟
- **P90**: ~60分钟
- **最慢**: 72.1分钟

**建议**：
- 保守: 3600秒（60分钟，保留98%）
- 平衡: 2400秒（40分钟，保留90%）
- 激进: 1800秒（30分钟，保留50%）

## 调试建议

### 1. 先用小样本测试

```bash
# 用small.jsonl测试（20个样本）
python3 r2e_k8s_example.py \
    --jsonl small.jsonl \
    --concurrent 5 \
    --debug \
    --output-dir test-timeout \
    --max-execution-time 2400 \
    --llm-timeout 180 \
    --tool-timeout 300
```

观察：
- 是否有超时？哪种超时？
- 平均每个样本用时多少？
- 失败的原因是什么？

### 2. 分析失败原因

检查日志：
```bash
# 查看超时样本
grep -l "timeout" single-output-v14/*.log

# 查看失败样本  
grep -l "failed" single-output-v14/*.log

# 统计tool超时
grep "Tool execution timeout" single-output-v14/*.log | wc -l

# 统计LLM超时
grep "LLM call timeout" single-output-v14/*.log | wc -l
```

### 3. 根据结果调整

- **Tool超时多**：增加 `--tool-timeout`
- **LLM超时多**：增加 `--llm-timeout` 或检查API性能
- **总超时多**：增加 `--max-execution-time`
- **成功率低**：放宽所有限制

## 监控脚本

创建一个实时监控脚本：

```bash
#!/bin/bash
# monitor_progress.sh

OUTPUT_DIR="single-output-v14"

while true; do
    clear
    echo "=== 执行进度监控 ==="
    echo
    
    total=$(ls $OUTPUT_DIR/*.log 2>/dev/null | wc -l)
    echo "已处理样本: $total"
    
    success=$(grep -l "success" $OUTPUT_DIR/*.context 2>/dev/null | wc -l)
    echo "成功: $success"
    
    timeout=$(grep -l "Execution time limit reached" $OUTPUT_DIR/*.log 2>/dev/null | wc -l)
    echo "总超时: $timeout"
    
    llm_timeout=$(grep "LLM call timeout" $OUTPUT_DIR/*.log 2>/dev/null | wc -l)
    echo "LLM超时: $llm_timeout"
    
    tool_timeout=$(grep "Tool execution timeout" $OUTPUT_DIR/*.log 2>/dev/null | wc -l)
    echo "Tool超时: $tool_timeout"
    
    echo
    echo "成功率: $(echo "scale=1; $success * 100 / $total" | bc)%"
    
    sleep 10
done
```

## 总结

**你当前的问题**：
- `--tool-timeout 90` **太短**，导致大量正常操作超时
- `--max-execution-time 1200` (20分钟) 偏短，会超时50%样本

**立即修复建议**：

```bash
python3 r2e_k8s_example.py \
    --jsonl test-00000-of-00001-with-images.jsonl \
    --concurrent 500 \
    --timeline \
    --output-dir single-output-v14 \
    --max-execution-time 2400 \
    --llm-timeout 180 \
    --tool-timeout 300
```

这样应该能达到：
- ✅ 成功率: ~90%
- ✅ Wall time: ~40分钟
- ✅ 过滤掉真正卡住的样本
