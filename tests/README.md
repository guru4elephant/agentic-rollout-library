# Tests Directory

This directory contains comprehensive tests and examples for the Agentic Rollout Library.

## Test Categories

### 🧪 Core Functionality Tests

#### **test_general_agent.py**
- **Purpose**: Tests GeneralAgent functionality with modern tool system
- **Features**: Mock LLM testing, JSON Action format, error handling
- **Status**: ✅ Updated and compatible
- **Usage**: `python test_general_agent.py`

#### **test_debug_mode.py**
- **Purpose**: Tests GeneralAgent debug mode functionality
- **Features**: LLM input/output logging, debug parameter validation
- **Status**: ✅ Current and functional
- **Usage**: `python test_debug_mode.py`

#### **test_json_action_format.py**
- **Purpose**: Tests JSON Action format parsing and validation
- **Features**: System prompt generation, JSON schema validation, parsing tests
- **Status**: ✅ Updated imports
- **Usage**: `python test_json_action_format.py`

#### **test_factory_pattern.py**
- **Purpose**: Tests the factory pattern for tool and agent creation
- **Features**: Tool factory, agent factory, batch creation
- **Status**: ✅ Current and functional
- **Usage**: `python test_factory_pattern.py`

### 🔗 LLM Integration Tests

#### **test_llm_client.py**
- **Purpose**: Tests LLMAPIClient functionality and features
- **Features**: Connection testing, parameter validation, error handling
- **Status**: ✅ Updated to use create_llm_client
- **Usage**: `python test_llm_client.py`

#### **test_llm_retry_mechanism.py**
- **Purpose**: Tests LLM retry mechanism and failure recovery
- **Features**: Retry logic, exponential backoff, error classification
- **Status**: ✅ Updated imports
- **Usage**: `python test_llm_retry_mechanism.py`

### 🚀 Integration and Rollout Tests

#### **test_general_agent_rollout.py**
- **Purpose**: Complete GeneralAgent rollout testing with K8S tools
- **Features**: Custom system prompt, K8S execution, trajectory saving
- **Status**: ✅ Current and functional
- **Configuration**: Uses swebench-xarray-pod for K8S execution
- **Usage**: `python test_general_agent_rollout.py`

#### **trajectory_save_test.py**
- **Purpose**: Tests trajectory saving functionality
- **Features**: Complete trajectory preservation, JSON/TXT formats
- **Status**: ✅ Updated imports
- **Usage**: `python trajectory_save_test.py`

### 📝 Example Programs

#### **real_llm_example.py**
- **Purpose**: K8S Pod monitoring with real LLM API
- **Features**: System monitoring, K8S tool usage, API integration
- **Status**: ✅ Updated to unified tool architecture
- **Usage**: `python real_llm_example.py`

#### **unified_tools_example.py**
- **Purpose**: Demonstrates unified tool interface (local vs K8S)
- **Features**: Execution mode switching, transparent tool usage
- **Status**: ✅ Updated imports
- **Usage**: `python unified_tools_example.py`

#### **debug_mode_example.py**
- **Purpose**: Demonstrates LLM debug mode capabilities
- **Features**: Debug logging comparison, API call inspection
- **Status**: ✅ Updated imports
- **Usage**: `python debug_mode_example.py`

#### **enhanced_debug_example.py**
- **Purpose**: Advanced debug mode demonstration
- **Features**: Comprehensive logging, debug utilities
- **Status**: ✅ Updated imports
- **Usage**: `python enhanced_debug_example.py`

### 📊 Feature-Specific Tests

#### **test_improved_system_prompt.py**
- **Purpose**: Tests system prompt improvements and JSON schema integration
- **Features**: Prompt validation, schema generation, tool documentation
- **Status**: ✅ Updated imports
- **Usage**: `python test_improved_system_prompt.py`

## Current Architecture

### ✅ Modern Features Supported

1. **Unified Tool Architecture**
   - Tools support both local and K8S execution via `execution_mode` parameter
   - Transparent switching between execution environments
   - Consistent tool interface regardless of execution mode

2. **JSON Action Format**
   - All tests use modern JSON-structured Actions
   - Proper JSON schema validation and parsing
   - Legacy format support with fallback mechanisms

3. **LLM Client Integration**
   - All tests use `create_llm_client()` factory function
   - Debug mode support for API call inspection
   - Retry mechanism with intelligent error handling

4. **Enhanced Debug Capabilities**
   - GeneralAgent debug mode logs all LLM inputs/outputs
   - LLMAPIClient debug mode for API call inspection
   - Comprehensive logging with structured format

5. **Trajectory Management**
   - Complete trajectory saving without content truncation
   - JSON and TXT format support
   - Full metadata and context preservation

## Configuration

### API Configuration
Most tests use the following default configuration:
```python
API_KEY = "sk-qq7xJtnAdB1Gv6IkHTQhDAPuUAT700vF3CMmGinILsmP2HuY"
BASE_URL = "http://211.23.3.237:27544"
MODEL_NAME = "gpt-4.1"  # or "claude-sonnet-4-20250514"
```

### K8S Configuration
K8S-enabled tests use:
```python
execution_mode = "k8s"
pod_name = "swebench-xarray-pod"
namespace = "default"
```

## Dependencies

All tests require:
- `openai` library for LLM API integration
- `pydantic` for data validation
- `json-repair` for robust JSON parsing (optional but recommended)
- `kodo` library for K8S functionality (for K8S tests)

## Usage Notes

1. **Mock vs Real LLM**: Some tests (like `test_general_agent.py`) use mock LLM for deterministic testing, while others (like `test_general_agent_rollout.py`) use real LLM APIs.

2. **Debug Mode**: Most tests can be run with debug mode enabled to see detailed execution logs.

3. **Execution Modes**: Tests demonstrate both local and K8S execution modes, showcasing the unified tool architecture.

4. **JSON Format**: All tests have been updated to use the modern JSON Action format with proper schema validation.

## Recent Updates (2025-08-02)

- ✅ Fixed BashExecutor JSON parameter parsing issue
- ✅ Updated all imports from `LLMAPIClient` to `create_llm_client`
- ✅ Modernized test_general_agent.py with current tool system
- ✅ Verified all tests use JSON Action format
- ✅ Updated API client usage across all files
- ✅ Fixed duplicate import statements
- ✅ Validated compatibility with current architecture

All tests are now compatible with the current agentic rollout library architecture and should run without compatibility issues.

## Quick Start

### Running Basic Tests
```bash
# Mock LLM tests (no API required)
python tests/test_general_agent.py
python tests/test_factory_pattern.py

# JSON parsing tests
python tests/test_json_action_format.py
```

### Running Integration Tests (requires API)
```bash
# Debug mode test
python tests/test_debug_mode.py

# Full rollout test
python tests/test_general_agent_rollout.py

# LLM client test
python tests/test_llm_client.py
```

### Running Examples
```bash
# Unified tools demo
python tests/unified_tools_example.py

# K8S monitoring demo
python tests/real_llm_example.py

# Debug mode demo
python tests/debug_mode_example.py
```