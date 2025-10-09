#!/bin/bash
# 测试 find 命令在 pod 中的实际性能

echo "=== 测试 Find 命令性能 ==="
echo

# 创建测试pod
POD_NAME="find-test-$(date +%s)"
NAMESPACE="qianfan-train-cpu-ns"
IMAGE="swe-bench/django__django-15315:latest"

echo "1. 创建测试 pod..."
cat <<YAML | kubectl --kubeconfig ./cpu_config2 apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: $POD_NAME
  namespace: $NAMESPACE
spec:
  containers:
  - name: test
    image: $IMAGE
    command: ["sleep", "300"]
    resources:
      requests:
        cpu: "0.3"
        memory: "1Gi"
  nodeSelector:
    nvme: "ok"
YAML

echo "2. 等待 pod 就绪..."
kubectl --kubeconfig ./cpu_config2 wait --for=condition=Ready pod/$POD_NAME -n $NAMESPACE --timeout=120s

echo
echo "3. 测试不同的查找方式..."
echo

# 测试1: 原始 find 命令
echo "测试1: find . -type f -name \"*.py\" | grep admin | head -20"
time kubectl --kubeconfig ./cpu_config2 exec $POD_NAME -n $NAMESPACE -- bash -c "cd /testbed && timeout 120 find . -type f -name '*.py' 2>/dev/null | grep admin | head -20" 2>&1 | tail -5
echo

# 测试2: 跳过隐藏目录
echo "测试2: find . -path '*/.*' -prune -o -type f -name \"*.py\" -print | grep admin | head -20"
time kubectl --kubeconfig ./cpu_config2 exec $POD_NAME -n $NAMESPACE -- bash -c "cd /testbed && timeout 120 find . -path '*/.*' -prune -o -type f -name '*.py' -print 2>/dev/null | grep admin | head -20" 2>&1 | tail -5
echo

# 测试3: 限制深度
echo "测试3: find . -maxdepth 3 -name \"*.py\" | grep admin | head -20"
time kubectl --kubeconfig ./cpu_config2 exec $POD_NAME -n $NAMESPACE -- bash -c "cd /testbed && timeout 120 find . -maxdepth 3 -name '*.py' 2>/dev/null | grep admin | head -20" 2>&1 | tail -5
echo

# 测试4: 使用 grep
echo "测试4: grep -r --include=\"*.py\" admin . | head -20"
time kubectl --kubeconfig ./cpu_config2 exec $POD_NAME -n $NAMESPACE -- bash -c "cd /testbed && timeout 120 grep -r --include='*.py' admin . 2>/dev/null | head -20" 2>&1 | tail -5
echo

echo "4. 清理测试 pod..."
kubectl --kubeconfig ./cpu_config2 delete pod $POD_NAME -n $NAMESPACE --force --grace-period=0 >/dev/null 2>&1

echo
echo "=== 测试完成 ==="
