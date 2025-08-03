#!/usr/bin/env python3
"""
Test XML action parser
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.parsers.xml_action_parser import parse_xml_action


def test_xml_parser():
    print("🧪 Testing XML Action Parser")
    print("="*60)
    
    # Test 1: Simple function call
    print("\n📝 Test 1: Simple function call")
    test1 = """<function=execute_bash>
<parameter=cmd>pwd</parameter>
</function>"""
    
    result = parse_xml_action(test1)
    print(f"Input: {test1}")
    print(f"Result: {result}")
    
    # Test 2: Function with thought
    print("\n📝 Test 2: Function with thought")
    test2 = """Let me check the current directory first.

<function=execute_bash>
<parameter=cmd>ls -la</parameter>
</function>"""
    
    result = parse_xml_action(test2)
    print(f"Input: {test2[:50]}...")
    print(f"Result type: {type(result)}")
    if isinstance(result, list):
        print(f"Number of steps: {len(result)}")
        for i, step in enumerate(result):
            print(f"  Step {i+1}: {step.step_type.value} - {step.content[:50]}...")
    
    # Test 3: File editor with multiple parameters
    print("\n📝 Test 3: File editor with multiple parameters")
    test3 = """<function=file_editor>
<parameter=command>create</parameter>
<parameter=path>/tmp/test.py</parameter>
<parameter=file_text>
def hello():
    print("Hello World")
</parameter>
</function>"""
    
    result = parse_xml_action(test3)
    print(f"Result: {result}")
    
    # Test 4: No function (should return None)
    print("\n📝 Test 4: No function")
    test4 = """This is just some regular text without any function call."""
    
    result = parse_xml_action(test4)
    print(f"Input: {test4}")
    print(f"Result: {result}")
    
    print("\n✅ XML Parser tests completed!")


if __name__ == "__main__":
    test_xml_parser()