#!/bin/bash

# Cleanup script for removing K8S pods with specific prefix
# Usage: ./cleanup_pods.sh [prefix] [namespace]
# Example: ./cleanup_pods.sh r2e- default

set -e

# Default values
PREFIX="${1:-r2e-}"
#NAMESPACE="${2:-default}"
NAMESPACE=qianfan-train-cpu-ns

echo "========================================="
echo "K8S Pod Cleanup Script"
echo "========================================="
echo "Prefix: ${PREFIX}"
echo "Namespace: ${NAMESPACE}"
echo ""

# Find all pods with the prefix
echo "🔍 Finding pods with prefix '${PREFIX}'..."
PODS=$(kubectl --kubeconfig ./cpu_config2 get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | grep "^${PREFIX}" | awk '{print $1}' || true)

if [ -z "$PODS" ]; then
    echo "✅ No pods found with prefix '${PREFIX}' in namespace '${NAMESPACE}'"
    exit 0
fi

# Count pods
POD_COUNT=$(echo "$PODS" | wc -l | tr -d ' ')
echo "📊 Found ${POD_COUNT} pod(s) to delete:"
echo "$PODS" | sed 's/^/  - /'
echo ""

# Ask for confirmation
echo

# Delete pods concurrently
echo ""
echo "🗑️  Deleting pods concurrently..."

# Create temporary directory for background job tracking
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

# Start deletion jobs in background
echo "$PODS" | while IFS= read -r pod; do
    (
        if kubectl --kubeconfig ./cpu_config2 -n "${NAMESPACE}" delete pod "${pod}" --force --grace-period=0 >/dev/null 2>&1; then
            echo "success" > "${TEMP_DIR}/${pod}.status"
            echo "  ✅ Deleted ${pod}"
        else
            echo "failed" > "${TEMP_DIR}/${pod}.status"
            echo "  ❌ Failed ${pod}"
        fi
    ) &
done

# Wait for all background jobs to complete
wait

# Count results
DELETED=$(find "${TEMP_DIR}" -name "*.status" -exec grep -l "success" {} \; 2>/dev/null | wc -l | tr -d ' ')
FAILED=$(find "${TEMP_DIR}" -name "*.status" -exec grep -l "failed" {} \; 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "========================================="
echo "Summary:"
echo "  ✅ Deleted: ${DELETED}"
echo "  ❌ Failed: ${FAILED}"
echo "  📊 Total: ${POD_COUNT}"
echo "========================================="
