#!/usr/bin/env python3
"""
Generate SWE-bench Verified Submission File

Reads JSONL input file and patch files from output directory to generate
a submission file in the SWE-bench verified format.

Usage:
    python generate_swe_verified.py --jsonl <input.jsonl> \
                                     --output-dir <patches_dir> \
                                     --model-name <model_name> \
                                     --output <submission.jsonl>

Example:
    python generate_swe_verified.py \
        --jsonl test-00000-of-00001-with-images.jsonl \
        --output-dir dsv31t-r2e-mini-50-output-v2 \
        --model-name kimi-k2 \
        --output predictions.jsonl
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of instances."""
    instances = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def read_patch_file(patch_path: Path) -> Optional[str]:
    """Read patch file content."""
    if not patch_path.exists():
        return None
    
    try:
        with open(patch_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Failed to read {patch_path}: {e}", file=sys.stderr)
        return None


def find_patch_file(output_dir: Path, instance_id: str) -> Optional[Path]:
    """Find patch file for given instance_id.
    
    Tries multiple naming patterns:
    1. {instance_id}.patch
    2. patches/{instance_id}_*.patch
    """
    # Try direct match
    patch_file = output_dir / f"{instance_id}.patch"
    if patch_file.exists():
        return patch_file
    
    # Try patches subdirectory with timestamp pattern
    patches_dir = output_dir / "patches"
    if patches_dir.exists():
        # Find files matching pattern: {instance_id}_*.patch
        matching_patches = list(patches_dir.glob(f"{instance_id}_*.patch"))
        if matching_patches:
            # Return the most recent one (by name, which includes timestamp)
            return sorted(matching_patches)[-1]
    
    return None


def generate_submission(
    jsonl_file: str,
    output_dir: str,
    model_name: str,
    output_file: str
) -> None:
    """Generate submission file in SWE-bench verified format."""
    
    # Load input instances
    print(f"Loading instances from {jsonl_file}...")
    instances = load_jsonl(jsonl_file)
    print(f"Loaded {len(instances)} instances")
    
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Generate predictions
    predictions = []
    missing_patches = []
    
    for instance in instances:
        instance_id = instance.get('instance_id')
        if not instance_id:
            print(f"Warning: Instance missing 'instance_id', skipping", file=sys.stderr)
            continue
        
        # Find and read patch file
        patch_file = find_patch_file(output_dir_path, instance_id)
        
        if patch_file:
            patch_content = read_patch_file(patch_file)
            if patch_content:
                prediction = {
                    "model_name_or_path": model_name,
                    "instance_id": instance_id,
                    "model_patch": patch_content
                }
                predictions.append(prediction)
            else:
                missing_patches.append(instance_id)
        else:
            missing_patches.append(instance_id)
    
    # Write output file
    print(f"\nWriting {len(predictions)} predictions to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    # Print summary
    print(f"\n{'='*80}")
    print("GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total instances: {len(instances)}")
    print(f"Predictions generated: {len(predictions)}")
    print(f"Missing patches: {len(missing_patches)}")
    
    if missing_patches:
        print(f"\nInstances with missing patches ({len(missing_patches)}):")
        for instance_id in sorted(missing_patches)[:10]:
            print(f"  - {instance_id}")
        if len(missing_patches) > 10:
            print(f"  ... and {len(missing_patches) - 10} more")
    
    print(f"\n✓ Submission file written to: {output_file}")
    print(f"  Coverage: {len(predictions)}/{len(instances)} ({len(predictions)/len(instances)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate SWE-bench verified submission file from patches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate_swe_verified.py \\
      --jsonl input.jsonl \\
      --output-dir output_dir \\
      --model-name my-model \\
      --output predictions.jsonl

  # Real example
  python generate_swe_verified.py \\
      --jsonl test-00000-of-00001-with-images.jsonl \\
      --output-dir dsv31t-r2e-mini-50-output-v2 \\
      --model-name kimi-k2 \\
      --output kimi-k2-predictions.jsonl
        """
    )
    
    parser.add_argument(
        '--jsonl',
        required=True,
        help='Input JSONL file with instances'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory containing patch files'
    )
    
    parser.add_argument(
        '--model-name',
        required=True,
        help='Model name or path for the submission'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSONL file for submission'
    )
    
    args = parser.parse_args()
    
    try:
        generate_submission(
            args.jsonl,
            args.output_dir,
            args.model_name,
            args.output
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
