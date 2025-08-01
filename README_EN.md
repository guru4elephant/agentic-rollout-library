# Agentic Rollout Library

A flexible and powerful library for creating agent rollouts that can work independently or integrate seamlessly with [VERL (Versatile Environment for Reinforcement Learning)](https://github.com/volcengine/verl). This library provides a comprehensive framework for building multi-step agentic trajectories with tool integration, customizable agents, and extensive trajectory management capabilities.

## 🌟 Key Features

### 🤖 Dual Usage Modes
- **Standalone Usage**: Complete independent functionality without VERL dependencies
- **VERL Integration**: Seamless integration as `verl.workers.rollout` rollout method
- **Flexible Architecture**: Graceful fallback when VERL is not available

### 🧠 Advanced Agent Framework
- **Base Agent System**: Abstract base class for custom agent implementations
- **ReAct Agents**: Built-in ReAct (Reasoning + Acting) agents with tool integration
- **Custom Agents**: Easy-to-extend framework for specialized agent behaviors
- **Trajectory Management**: Complete trajectory tracking with serialization support

### 🛠️ Unified Tool Framework
- **VERL Compatibility**: Seamless integration with VERL tools when available
- **Standalone Operation**: Full functionality without VERL dependencies
- **Core Tools**: Calculator, file editor, bash executor, and search tools
- **Tool Registry**: Centralized tool management and discovery
- **Custom Tools**: Easy framework for developing new tools
- **Security Features**: Safe execution with configurable restrictions

### 🔌 LLM Client Flexibility
- **OpenAI SDK Compatible**: Standard OpenAI API protocol support
- **Multiple Providers**: Support for various model names and base URLs
- **Claude Integration**: Specialized Claude API client implementation
- **Custom LLM Functions**: Easy integration of custom LLM backends

## 📁 Project Structure

```
agentic_rollout_library/
├── workers/                          # Core library modules
│   ├── __init__.py
│   ├── agentic_rollout.py           # Main rollout implementation
│   ├── core/                        # Core framework components
│   │   ├── base_agent.py           # Abstract base agent class
│   │   ├── base_tool.py            # Unified tool framework
│   │   ├── tool_registry.py        # Tool management system
│   │   ├── tool_schemas.py         # Tool schema definitions
│   │   ├── registry.py             # Agent registry system
│   │   └── trajectory.py           # Trajectory data structures
│   ├── agents/                      # Built-in agent implementations
│   │   ├── react_agent.py          # ReAct agent implementation
│   │   ├── coding_agent.py         # Specialized coding agent
│   │   └── tool_agent.py           # Tool-focused agent
│   └── tools/                       # Core tool implementations
│       ├── calculator_tool.py      # Mathematical computations
│       ├── bash_executor_tool.py   # Safe command execution
│       ├── file_editor_tool.py     # File operations
│       └── search_tool.py          # Text and file search
├── examples/                        # Usage examples and demos
│   ├── basic_usage.py              # Basic standalone usage
│   ├── advanced_integration.py     # Advanced integration examples
│   ├── claude_integration/         # Claude API integration demo
│   ├── k8s_swe_agent/             # Kubernetes SWE agent implementation
│   ├── trajectory_client/          # Trajectory client utilities
│   └── config/                     # Configuration templates
└── README.md                       # This file
```

## 🚀 Quick Start

### Installation

This library is configured with complete Python package management, supporting direct installation via pip, including automatic installation of kodo dependency (for K8s control and execution logic).

**Option 1: Install from Source (Recommended)**
```bash
# Clone the repository
git clone <repository-url>
cd agentic-rollout-library

# Install package with all dependencies (including kodo)
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

**Option 2: Virtual Environment Installation**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install package
pip install -e .
```

**Option 3: With VERL Integration**
```bash
# Ensure VERL is installed first
export PYTHONPATH="/path/to/verl:$PYTHONPATH"

# Clone and install
git clone <repository-url>
cd agentic-rollout-library
pip install -e .
```

**Dependency Notes:**
- Core dependencies are automatically installed, including `pydantic>=2.0.0` and `typing-extensions>=4.0.0`
- **kodo dependency** is automatically installed from `https://github.com/baidubce/kodo.git` for K8s control and execution logic
- Development dependencies include testing and code quality tools

### Basic Usage Example

```python
import asyncio
from workers.agentic_rollout import AgenticRollout, AgenticRolloutConfig
from workers.core.trajectory import Trajectory, TrajectoryStep, StepType

# Create configuration
config = AgenticRolloutConfig(
    agent_type="react",
    max_steps=10,
    max_tokens_per_step=512,
    temperature=0.7
)

# Define your LLM function (OpenAI SDK compatible)
async def llm_generate_func(messages, max_tokens=512, temperature=0.7, **kwargs):
    # Your LLM implementation here
    # Works with OpenAI, Claude, or any compatible API
    pass

# Create rollout instance
rollout = AgenticRollout(config=config, llm_generate_func=llm_generate_func)

# Run trajectory
prompt_data = {"content": "Solve this math problem: What is 15 * 24?"}
trajectory = await rollout.agent.run_trajectory(
    prompt=prompt_data,
    llm_generate_func=llm_generate_func,
    request_id="math_example"
)

print(f"Final response: {trajectory.get_final_response()}")
print(f"Total steps: {len(trajectory.steps)}")
```

## 📖 Usage Examples

### 1. Standalone Agent with Tools

```python
from workers.core.base_agent import BaseAgent
from workers.core.trajectory import Trajectory, TrajectoryStep, StepType

class CalculatorTool:
    async def execute(self, expression: str):
        # Safe evaluation of mathematical expressions
        return {"result": eval(expression), "expression": expression}

# Create agent with tools
config = AgenticRolloutConfig(agent_type="react", max_steps=5)
rollout = AgenticRollout(config=config, llm_generate_func=your_llm_func)

# Add tools
rollout.tools = {"calculator": CalculatorTool()}
rollout.agent.set_tools(rollout.tools)

# Run with tool usage
result = await rollout.agent.run_trajectory(
    prompt={"content": "Calculate the square root of 144"},
    llm_generate_func=your_llm_func,
    request_id="calc_example"
)
```

### 2. Custom Agent Implementation

```python
from workers.core.base_agent import BaseAgent

class CustomReasoningAgent(BaseAgent):
    async def run_trajectory(self, prompt, llm_generate_func, request_id, **kwargs):
        trajectory = Trajectory(request_id=request_id)
        
        # Add initial observation
        obs_step = TrajectoryStep(
            step_type=StepType.OBSERVATION,
            content=str(prompt.get('content', prompt))
        )
        trajectory.add_step(obs_step)
        
        # Custom reasoning logic
        while self.should_continue(trajectory):
            # Generate LLM response
            messages = self.format_messages_for_llm(trajectory)
            response = await llm_generate_func(messages)
            
            # Parse and add step
            step = self.parse_llm_output(response)
            trajectory.add_step(step)
            
            # Handle tool calls if needed
            if step.tool_name:
                result_step = await self.execute_tool_call(
                    step.tool_name, step.tool_args, trajectory
                )
                trajectory.add_step(result_step)
        
        self.finalize_trajectory(trajectory)
        return trajectory

# Use custom agent
agent = CustomReasoningAgent(max_steps=8)
```

### 3. VERL Integration

When VERL is available, the library automatically integrates:

```python
# This works automatically when VERL is installed
from verl.workers.rollout.agentic_rollout import AgenticRollout
from verl.protocol import DataProto

# Use as a VERL rollout worker
rollout = AgenticRollout(config, llm_generate_func, tokenizer)
output_data = await rollout.generate_sequences(input_prompts)
```

### 4. Multi-Model Support

```python
# OpenAI API
async def openai_llm_func(messages, **kwargs):
    import openai
    client = openai.AsyncOpenAI(api_key="your-key")
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        **kwargs
    )
    return response.choices[0].message.content

# Claude API
async def claude_llm_func(messages, **kwargs):
    from examples.claude_integration.claude_llm_client import ClaudeAPIClient
    client = ClaudeAPIClient(api_key="your-key")
    return await client.generate(messages, **kwargs)

# Use with any compatible LLM
rollout = AgenticRollout(config=config, llm_generate_func=claude_llm_func)
```

## 🔧 Configuration

### AgenticRolloutConfig Options

```python
config = AgenticRolloutConfig(
    # Agent settings
    agent_type="react",                    # Agent type: "react", "coding", "tool"
    max_steps=10,                         # Maximum trajectory steps
    max_tokens_per_step=512,              # Max tokens per generation
    temperature=0.7,                      # LLM sampling temperature
    
    # Tool configuration
    tools_config={                        # Tool-specific settings
        "calculator": {"precision": 10},
        "search": {"max_results": 5}
    },
    
    # Performance settings
    batch_size=1,                         # Batch processing size
    concurrent_requests=4,                # Concurrent request limit
    
    # Output settings
    include_trajectory_in_output=True,    # Include full trajectory
    save_trajectories=True,               # Save to disk
    trajectory_save_path="./trajectories" # Save location
)
```

### Environment Variables

```bash
# LLM API Configuration
export OPENAI_API_KEY="your-openai-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

export ANTHROPIC_API_KEY="your-claude-key"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"

# Library Configuration
export AGENTIC_LOG_LEVEL="INFO"
export AGENTIC_SAVE_TRAJECTORIES="true"
```

## 🛠️ Advanced Features

### Trajectory Analysis

```python
# Analyze trajectory performance
trajectory = await agent.run_trajectory(...)

# Get trajectory statistics
print(f"Total steps: {len(trajectory.steps)}")
print(f"Tool calls: {len(trajectory.get_tool_calls())}")
print(f"Final reward: {trajectory.get_total_reward()}")

# Export trajectory
trajectory_dict = trajectory.to_dict()
with open("trajectory.json", "w") as f:
    json.dump(trajectory_dict, f, indent=2)
```

### Custom Tool Integration

```python
class WebSearchTool:
    def __init__(self, api_key):
        self.api_key = api_key
    
    async def execute(self, query: str, max_results: int = 5):
        # Implement web search logic
        results = await self.search_web(query)
        return {
            "query": query,
            "results": results[:max_results],
            "found": len(results)
        }

# Register tool
tools = {"web_search": WebSearchTool(api_key="your-key")}
rollout.agent.set_tools(tools)
```

### Batch Processing

```python
# Process multiple prompts concurrently
prompts = [
    {"content": "Solve math problem: 15 * 24"},
    {"content": "Write Python code to sort a list"},
    {"content": "Explain quantum computing"}
]

config = AgenticRolloutConfig(concurrent_requests=3)
rollout = AgenticRollout(config=config, llm_generate_func=llm_func)

# Process batch
trajectories = []
for i, prompt in enumerate(prompts):
    trajectory = await rollout.agent.run_trajectory(
        prompt=prompt,
        llm_generate_func=llm_func,
        request_id=f"batch_{i}"
    )
    trajectories.append(trajectory)
```

## 📊 Examples and Demos

The `examples/` directory contains comprehensive demonstrations:

- **`basic_usage.py`**: Fundamental usage patterns
- **`claude_integration/`**: Complete Claude API integration with ReAct agents
- **`k8s_swe_agent/`**: Kubernetes-powered software engineering agent
- **`trajectory_client/`**: Trajectory management and analysis utilities
- **`config/`**: Configuration templates for different scenarios

To run examples:
```bash
cd examples
python basic_usage.py

# Claude integration (requires API key)
cd claude_integration
export ANTHROPIC_API_KEY="your-key"
python simple_claude_example.py
```

## 🤝 Integration with VERL

When VERL is available, this library provides:

1. **Seamless Integration**: Works as a drop-in rollout method
2. **Protocol Compatibility**: Full DataProto support
3. **Tool Inheritance**: Inherits VERL's tool ecosystem
4. **Performance Features**: Leverages VERL's optimization features

```python
# In VERL environment
from verl.workers.rollout import AgenticRollout

# Use in VERL training pipeline
rollout_config = {
    "rollout_type": "agentic",
    "agent_config": {
        "agent_type": "react",
        "max_steps": 10
    }
}
```

## 🔍 Architecture Overview

### Core Components

1. **BaseAgent**: Abstract agent interface with trajectory management
2. **AgenticRollout**: Main rollout orchestrator with VERL compatibility
3. **Trajectory System**: Complete step tracking and serialization
4. **Tool Framework**: Flexible tool integration with async support
5. **Registry System**: Dynamic agent and tool registration

### Design Principles

- **Modular Architecture**: Clear separation of concerns
- **Async-First**: Full async/await support throughout
- **Extensible Design**: Easy to add new agents, tools, and integrations
- **Production Ready**: Comprehensive error handling and logging
- **Performance Focused**: Concurrent processing and efficient resource usage

## 🧪 Testing

```bash
# Run basic tests
python examples/test_agentic_rollout.py

# Run validation tests
python examples/validation_tests.py

# Test specific components
python examples/test_coding_agent.py
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow Python typing conventions
- Add comprehensive docstrings
- Include example usage in docstrings
- Write tests for new features
- Maintain backward compatibility

## 📄 License

This project follows the same license as the VERL library.

## ❓ FAQ

**Q: Can I use this library without VERL?**
A: Yes! The library is designed to work completely independently and gracefully handles VERL's absence.

**Q: What LLM providers are supported?**
A: Any provider that follows the OpenAI SDK protocol. We provide examples for OpenAI, Claude, and custom implementations.

**Q: How do I add custom tools?**
A: Create a class with an async `execute` method and register it with your agent using `agent.set_tools()`.

**Q: Can I use this for production workloads?**
A: Yes, the library includes comprehensive error handling, logging, and performance optimizations suitable for production use.

## 🆘 Support

- 📖 Check the [examples directory](./examples/) for usage patterns
- 🐛 Report issues in the project's issue tracker
- 💬 Join community discussions for questions and feature requests

---

**Made with ❤️ for the AI Agent Community**