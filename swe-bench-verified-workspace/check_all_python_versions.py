#!/usr/bin/env python3
import json
import subprocess
import sys
import time
from collections import defaultdict

def get_python_version_in_pod(image, timeout=120):
    """启动pod并检查Python版本"""
    import random
    pod_name = f"py-check-{random.randint(10000, 99999)}"
    
    try:
        # 创建pod
        print(f"  Creating pod for {image.split('/')[-1][:40]}...")
        
        pod_yaml = f"""
apiVersion: v1
kind: Pod
metadata:
  name: {pod_name}
  namespace: qianfan-train-cpu-ns
spec:
  containers:
  - name: test
    image: {image}
    command: ["sleep", "300"]
  nodeSelector:
    nvme: "ok"
"""
        
        with open(f"/tmp/{pod_name}.yaml", "w") as f:
            f.write(pod_yaml)
        
        # 创建pod
        subprocess.run(
            ["kubectl", "--kubeconfig", "./cpu_config2", "apply", "-f", f"/tmp/{pod_name}.yaml"],
            capture_output=True,
            timeout=30
        )
        
        # 等待pod ready
        print(f"  Waiting for pod ready...")
        result = subprocess.run(
            ["kubectl", "--kubeconfig", "./cpu_config2", "wait", 
             f"--for=condition=Ready", f"pod/{pod_name}", 
             "-n", "qianfan-train-cpu-ns", f"--timeout={timeout}s"],
            capture_output=True,
            timeout=timeout + 10,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  ⚠️  Pod not ready: {result.stderr}")
            return None
        
        # 检查python3版本
        result = subprocess.run(
            ["kubectl", "--kubeconfig", "./cpu_config2", "exec", 
             pod_name, "-n", "qianfan-train-cpu-ns", "--",
             "python3", "--version"],
            capture_output=True,
            timeout=10,
            text=True
        )
        
        version = None
        if result.returncode == 0:
            version = result.stdout.strip()
        
        # 检查conda testbed版本
        conda_result = subprocess.run(
            ["kubectl", "--kubeconfig", "./cpu_config2", "exec", 
             pod_name, "-n", "qianfan-train-cpu-ns", "--",
             "/opt/miniconda3/envs/testbed/bin/python", "--version"],
            capture_output=True,
            timeout=10,
            text=True
        )
        
        conda_version = None
        if conda_result.returncode == 0:
            conda_version = conda_result.stdout.strip()
        
        return {
            "python3": version,
            "conda_testbed": conda_version
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None
    finally:
        # 删除pod
        subprocess.run(
            ["kubectl", "--kubeconfig", "./cpu_config2", "delete", 
             "pod", pod_name, "-n", "qianfan-train-cpu-ns", 
             "--force", "--grace-period=0"],
            capture_output=True,
            timeout=30
        )

def main():
    jsonl_file = "test-00000-of-00001-with-images.jsonl"
    
    print(f"📋 Loading instances from {jsonl_file}...")
    instances = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    
    print(f"✅ Loaded {len(instances)} instances\n")
    
    # 统计结果
    version_stats = defaultdict(int)
    conda_stats = defaultdict(int)
    results = []
    
    for idx, instance in enumerate(instances, 1):
        instance_id = instance.get("instance_id", "unknown")
        image = instance.get("image")
        
        if not image:
            print(f"[{idx}/{len(instances)}] ⚠️  {instance_id}: No image")
            continue
        
        print(f"\n[{idx}/{len(instances)}] 🔍 {instance_id}")
        print(f"  Image: {image}")
        
        versions = get_python_version_in_pod(image)
        
        if versions:
            py3_ver = versions.get("python3", "N/A")
            conda_ver = versions.get("conda_testbed", "N/A")
            
            print(f"  ✅ python3: {py3_ver}")
            print(f"  ✅ conda testbed: {conda_ver}")
            
            version_stats[py3_ver] += 1
            conda_stats[conda_ver] += 1
            
            results.append({
                "instance_id": instance_id,
                "image": image,
                "python3": py3_ver,
                "conda_testbed": conda_ver
            })
        else:
            print(f"  ❌ Failed to get versions")
    
    # 打印统计结果
    print("\n" + "="*80)
    print("📊 PYTHON VERSION STATISTICS")
    print("="*80)
    
    print("\n🐍 System python3 versions:")
    for version, count in sorted(version_stats.items()):
        print(f"  {version}: {count} images")
    
    print("\n🐍 Conda testbed python versions:")
    for version, count in sorted(conda_stats.items()):
        print(f"  {version}: {count} images")
    
    # 保存详细结果
    output_file = "python_version_report.json"
    with open(output_file, 'w') as f:
        json.dump({
            "total_images": len(instances),
            "python3_stats": dict(version_stats),
            "conda_testbed_stats": dict(conda_stats),
            "details": results
        }, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {output_file}")

if __name__ == "__main__":
    main()
