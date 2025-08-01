# Agentic Rollout Library

> [English Version](README_EN.md) | 中文版

一个灵活强大的智能体rollout库，既可以独立使用，也可以与 [VERL (Versatile Environment for Reinforcement Learning)](https://github.com/volcengine/verl) 无缝集成。该库提供了一个全面的框架，用于构建具有工具集成、可定制智能体和广泛轨迹管理功能的多步智能体轨迹。

## 🌟 核心特性

### 🤖 双重使用模式
- **独立使用**：无需VERL依赖的完整独立功能
- **VERL集成**：作为 `verl.workers.rollout` rollout方法的无缝集成
- **灵活架构**：当VERL不可用时优雅降级

### 🧠 高级智能体框架
- **基础智能体系统**：用于自定义智能体实现的抽象基类
- **ReAct智能体**：内置ReAct（推理+行动）智能体，支持工具集成
- **自定义智能体**：易于扩展的专业化智能体行为框架
- **轨迹管理**：完整的轨迹跟踪和序列化支持

### 🛠️ 统一工具框架
- **VERL兼容性**：当VERL可用时与VERL工具无缝集成
- **独立运行**：无VERL依赖的完整功能
- **核心工具**：计算器、文件编辑器、bash执行器和搜索工具
- **工具注册表**：集中化工具管理和发现
- **自定义工具**：开发新工具的简易框架
- **安全特性**：具有可配置限制的安全执行

### 🔌 LLM客户端灵活性
- **OpenAI SDK兼容**：标准OpenAI API协议支持
- **多提供商支持**：支持各种模型名称和基础URL
- **Claude集成**：专门的Claude API客户端实现
- **自定义LLM函数**：轻松集成自定义LLM后端

## 📁 项目结构

```
agentic_rollout_library/
├── workers/                          # 核心库模块
│   ├── __init__.py
│   ├── agentic_rollout.py           # 主要rollout实现
│   ├── core/                        # 核心框架组件
│   │   ├── base_agent.py           # 抽象基础智能体类
│   │   ├── base_tool.py            # 统一工具框架
│   │   ├── tool_registry.py        # 工具管理系统
│   │   ├── tool_schemas.py         # 工具模式定义
│   │   ├── registry.py             # 智能体注册系统
│   │   └── trajectory.py           # 轨迹数据结构
│   ├── agents/                      # 内置智能体实现
│   │   ├── react_agent.py          # ReAct智能体实现
│   │   ├── coding_agent.py         # 专业化编程智能体
│   │   └── tool_agent.py           # 工具专用智能体
│   └── tools/                       # 核心工具实现
│       ├── calculator_tool.py      # 数学计算
│       ├── bash_executor_tool.py   # 安全命令执行
│       ├── file_editor_tool.py     # 文件操作
│       └── search_tool.py          # 文本和文件搜索
├── examples/                        # 使用示例和演示
│   ├── basic_usage.py              # 基础独立使用
│   ├── advanced_integration.py     # 高级集成示例
│   ├── core_tools_demo.py          # 核心工具演示
│   ├── tool_integration_demo.py    # 工具集成演示
│   └── config/                     # 配置模板
└── README.md                       # 本文件
```

## 🚀 快速开始

### 安装

该库已配置了完整的Python包管理，支持通过pip直接安装，包括自动安装kodo依赖（用于K8s控制和执行逻辑）。

**选项1：从源码安装（推荐）**
```bash
# 克隆仓库
git clone <repository-url>
cd agentic-rollout-library

# 安装包及所有依赖（包括kodo）
pip install -e .

# 或者安装开发依赖
pip install -e ".[dev]"
```

**选项2：创建虚拟环境安装**
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者 venv\Scripts\activate  # Windows

# 安装包
pip install -e .
```

**选项3：与VERL集成**
```bash
# 确保首先安装VERL
export PYTHONPATH="/path/to/verl:$PYTHONPATH"

# 克隆并安装
git clone <repository-url>
cd agentic-rollout-library
pip install -e .
```

**依赖说明：**
- 主要依赖会自动安装，包括 `pydantic>=2.0.0` 和 `typing-extensions>=4.0.0`
- **kodo依赖**会从 `https://github.com/baidubce/kodo.git` 自动安装，用于K8s控制和执行逻辑
- 开发依赖包括测试和代码质量工具

### 基础使用示例

```python
import asyncio
from workers import (
    AgenticRollout, AgenticRolloutConfig,
    CalculatorTool, get_global_tool_registry, register_tool
)

# 创建配置
config = AgenticRolloutConfig(
    agent_type="react",
    max_steps=10,
    max_tokens_per_step=512,
    temperature=0.7
)

# 定义你的LLM函数（OpenAI SDK兼容）
async def llm_generate_func(messages, max_tokens=512, temperature=0.7, **kwargs):
    # 你的LLM实现代码
    # 适用于OpenAI、Claude或任何兼容API
    pass

# 注册工具
register_tool(CalculatorTool, {"debug": True})

# 创建rollout实例
rollout = AgenticRollout(config=config, llm_generate_func=llm_generate_func)

# 运行轨迹
prompt_data = {"content": "计算15 * 24的结果是什么？"}
trajectory = await rollout.agent.run_trajectory(
    prompt=prompt_data,
    llm_generate_func=llm_generate_func,
    request_id="math_example"
)

print(f"最终响应: {trajectory.get_final_response()}")
print(f"总步数: {len(trajectory.steps)}")
```

## 📖 使用示例

### 1. 带工具的独立智能体

```python
from workers import (
    AgenticRollout, AgenticRolloutConfig,
    CalculatorTool, FileEditorTool, BashExecutorTool,
    get_global_tool_registry, register_tool
)

# 注册多个工具
register_tool(CalculatorTool, {"precision": 10})
register_tool(FileEditorTool, {"max_file_size": 1024*1024})
register_tool(BashExecutorTool, {"timeout": 30})

# 创建智能体配置
config = AgenticRolloutConfig(agent_type="react", max_steps=8)
rollout = AgenticRollout(config=config, llm_generate_func=your_llm_func)

# 运行复杂任务
result = await rollout.agent.run_trajectory(
    prompt={"content": "创建一个Python脚本来计算斐波那契数列的前10项"},
    llm_generate_func=your_llm_func,
    request_id="fibonacci_task"
)
```

### 2. 自定义智能体实现

```python
from workers.core.base_agent import BaseAgent
from workers.core.trajectory import Trajectory, TrajectoryStep, StepType

class CustomReasoningAgent(BaseAgent):
    async def run_trajectory(self, prompt, llm_generate_func, request_id, **kwargs):
        trajectory = Trajectory(request_id=request_id)
        
        # 添加初始观察
        obs_step = TrajectoryStep(
            step_type=StepType.OBSERVATION,
            content=str(prompt.get('content', prompt))
        )
        trajectory.add_step(obs_step)
        
        # 自定义推理逻辑
        while self.should_continue(trajectory):
            # 生成LLM响应
            messages = self.format_messages_for_llm(trajectory)
            response = await llm_generate_func(messages)
            
            # 解析并添加步骤
            step = self.parse_llm_output(response)
            trajectory.add_step(step)
            
            # 如需要，处理工具调用
            if step.tool_name:
                result_step = await self.execute_tool_call(
                    step.tool_name, step.tool_args, trajectory
                )
                trajectory.add_step(result_step)
        
        self.finalize_trajectory(trajectory)
        return trajectory

# 使用自定义智能体
agent = CustomReasoningAgent(max_steps=8)
```

### 3. VERL集成

当VERL可用时，库会自动集成：

```python
# 当安装了VERL时，这会自动工作
from verl.workers.rollout.agentic_rollout import AgenticRollout
from verl.protocol import DataProto

# 作为VERL rollout worker使用
rollout = AgenticRollout(config, llm_generate_func, tokenizer)
output_data = await rollout.generate_sequences(input_prompts)
```

### 4. 多模型支持

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
    from workers.tools.claude_llm_client import ClaudeAPIClient
    client = ClaudeAPIClient(api_key="your-key")
    return await client.generate(messages, **kwargs)

# 与任何兼容LLM一起使用
rollout = AgenticRollout(config=config, llm_generate_func=claude_llm_func)
```

## 🔧 配置

### AgenticRolloutConfig 选项

```python
config = AgenticRolloutConfig(
    # 智能体设置
    agent_type="react",                    # 智能体类型: "react", "coding", "tool"
    max_steps=10,                         # 最大轨迹步数
    max_tokens_per_step=512,              # 每步最大token数
    temperature=0.7,                      # LLM采样温度
    
    # 工具配置
    tools_config={                        # 工具特定设置
        "calculator": {"precision": 10},
        "search": {"max_results": 5}
    },
    
    # 性能设置
    batch_size=1,                         # 批处理大小
    concurrent_requests=4,                # 并发请求限制
    
    # 输出设置
    include_trajectory_in_output=True,    # 输出中包含完整轨迹
    save_trajectories=True,               # 保存轨迹到磁盘
    trajectory_save_path="./trajectories" # 保存位置
)
```

### 环境变量

```bash
# LLM API配置
export OPENAI_API_KEY="your-openai-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

export ANTHROPIC_API_KEY="your-claude-key"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"

# 库配置
export AGENTIC_LOG_LEVEL="INFO"
export AGENTIC_SAVE_TRAJECTORIES="true"
```

## 🛠️ 核心工具详解

### 计算器工具 (CalculatorTool)
高级数学计算工具，支持：
- 表达式求值和安全AST解析
- 基础算术运算
- 科学函数（三角函数、对数等）
- 列表统计运算
- 计算历史跟踪

```python
from workers import CalculatorTool, register_tool

register_tool(CalculatorTool, {"precision": 10})
# 支持: "2 + 3 * 4", "sqrt(16) + factorial(4)", "sin(pi/2)"
```

### Bash执行器工具 (BashExecutorTool)
安全的bash命令执行，具有：
- 危险命令安全过滤
- 超时保护
- 输出捕获和流式传输
- 工作目录支持

```python
from workers import BashExecutorTool, register_tool

register_tool(BashExecutorTool, {
    "timeout": 30,
    "blocked_commands": ["rm", "sudo"]
})
```

### 文件编辑器工具 (FileEditorTool)
全面的文件操作，支持：
- 文件和目录查看
- 文件创建和编辑
- 唯一性检查的字符串替换
- 特定行的文本插入
- 编辑历史和撤销功能

```python
from workers import FileEditorTool, register_tool

register_tool(FileEditorTool, {
    "max_file_size": 1024*1024,
    "allowed_extensions": [".py", ".txt", ".md"]
})
```

### 搜索工具 (SearchTool)
强大的搜索功能，包括：
- 文件中的文本搜索，支持正则表达式
- 文件名模式匹配
- 目录结构搜索
- 上下文行显示

```python
from workers import SearchTool, register_tool

register_tool(SearchTool, {
    "max_results": 100,
    "search_extensions": [".py", ".js", ".md"]
})
```

## 📊 示例和演示

`examples/` 目录包含全面的演示：

- **`basic_usage.py`**: 基础使用模式
- **`advanced_integration.py`**: 高级集成示例  
- **`core_tools_demo.py`**: 核心工具系统演示
- **`tool_integration_demo.py`**: 完整工具集成演示
- **`config/`**: 不同场景的配置模板

运行示例：
```bash
cd examples

# 基础功能
python basic_usage.py

# 工具系统演示
python core_tools_demo.py

# 完整集成演示
python tool_integration_demo.py
```

## 🤝 与VERL的集成

当VERL可用时，该库提供：

1. **无缝集成**：作为drop-in rollout方法工作
2. **协议兼容性**：完整的DataProto支持
3. **工具继承**：继承VERL的工具生态系统
4. **性能特性**：利用VERL的优化特性

```python
# 在VERL环境中
from verl.workers.rollout import AgenticRollout

# 在VERL训练管道中使用
rollout_config = {
    "rollout_type": "agentic",
    "agent_config": {
        "agent_type": "react",
        "max_steps": 10
    }
}
```

## 🔍 架构概览

### 核心组件

1. **BaseAgent**: 带轨迹管理的抽象智能体接口
2. **AgenticRollout**: 具有VERL兼容性的主要rollout编排器
3. **轨迹系统**: 完整的步骤跟踪和序列化
4. **工具框架**: 具有异步支持的灵活工具集成
5. **注册系统**: 动态智能体和工具注册

### 设计原则

- **模块化架构**: 清晰的关注点分离
- **异步优先**: 全面的async/await支持
- **可扩展设计**: 易于添加新智能体、工具和集成
- **生产就绪**: 全面的错误处理和日志记录
- **性能专注**: 并发处理和高效资源使用

## 🧪 测试

```bash
# 运行基础测试
python examples/test_agentic_rollout.py

# 运行验证测试
python examples/validation_tests.py

# 测试特定组件
python examples/test_coding_agent.py
```

## 🎯 应用场景

### 数学和科学计算
```python
# 复杂计算任务
trajectory = await agent.run_trajectory(
    prompt={"content": "计算正态分布的95%置信区间，均值为100，标准差为15"},
    llm_generate_func=llm_func
)
```

### 代码开发和测试
```python
# 软件开发工作流
trajectory = await agent.run_trajectory(
    prompt={"content": "创建一个快速排序算法的Python实现并编写单元测试"},
    llm_generate_func=llm_func
)
```

### 文件处理和分析
```python
# 文档分析任务
trajectory = await agent.run_trajectory(
    prompt={"content": "分析这个项目目录，找出所有Python文件的导入依赖关系"},
    llm_generate_func=llm_func
)
```

### 系统管理自动化
```python
# 系统运维任务
trajectory = await agent.run_trajectory(
    prompt={"content": "检查系统磁盘使用情况，如果超过80%使用率则清理临时文件"},
    llm_generate_func=llm_func
)
```

## 📝 贡献

1. Fork仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启Pull Request

### 开发指南

- 遵循Python类型约定
- 添加全面的docstring
- 在docstring中包含示例用法
- 为新功能编写测试
- 保持向后兼容性

## 🔧 高级特性

### 轨迹分析

```python
# 分析轨迹性能
trajectory = await agent.run_trajectory(...)

# 获取轨迹统计
print(f"总步数: {len(trajectory.steps)}")
print(f"工具调用: {len(trajectory.get_tool_calls())}")
print(f"最终奖励: {trajectory.get_total_reward()}")

# 导出轨迹
trajectory_dict = trajectory.to_dict()
with open("trajectory.json", "w", encoding="utf-8") as f:
    json.dump(trajectory_dict, f, indent=2, ensure_ascii=False)
```

### 自定义工具集成

```python
from workers.core.base_tool import AgenticBaseTool
from workers.core.tool_schemas import create_openai_tool_schema, ToolResult

class WebSearchTool(AgenticBaseTool):
    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = self.config.get("api_key")
    
    def get_openai_tool_schema(self):
        return create_openai_tool_schema(
            name="web_search",
            description="在网络上搜索信息",
            parameters={
                "query": {"type": "string", "description": "搜索查询"},
                "max_results": {"type": "integer", "description": "最大结果数", "default": 5}
            },
            required=["query"]
        )
    
    async def execute_tool(self, instance_id, parameters, **kwargs):
        query = parameters["query"]
        max_results = parameters.get("max_results", 5)
        
        # 实现网络搜索逻辑
        results = await self.search_web(query)
        
        return ToolResult(
            success=True,
            result={
                "query": query,
                "results": results[:max_results],
                "found": len(results)
            }
        )

# 注册工具
from workers import register_tool
register_tool(WebSearchTool, {"api_key": "your-key"})
```

### 批处理

```python
# 并发处理多个提示
prompts = [
    {"content": "解决数学问题: 15 * 24"},
    {"content": "编写Python代码对列表进行排序"},
    {"content": "解释量子计算的基本原理"}
]

config = AgenticRolloutConfig(concurrent_requests=3)
rollout = AgenticRollout(config=config, llm_generate_func=llm_func)

# 处理批次
trajectories = []
for i, prompt in enumerate(prompts):
    trajectory = await rollout.agent.run_trajectory(
        prompt=prompt,
        llm_generate_func=llm_func,
        request_id=f"batch_{i}"
    )
    trajectories.append(trajectory)
```

## ⚡ 性能优化

### 并发设置
```python
config = AgenticRolloutConfig(
    concurrent_requests=4,      # 并发请求数
    max_steps=10,              # 限制步数防止无限循环
    max_tokens_per_step=512,   # 控制生成长度
    batch_size=2               # 批处理大小
)
```

### 内存管理
```python
config = AgenticRolloutConfig(
    save_trajectories=False,           # 不保存到磁盘节省空间
    include_trajectory_in_output=True, # 仅在内存中保持轨迹
    trajectory_save_path=None          # 禁用文件保存
)
```

## 🆘 故障排除

### 常见问题

1. **智能体未找到**: 确保智能体类型正确注册
2. **工具执行失败**: 检查工具配置和可用性
3. **无限循环**: 调整 `max_steps` 和终止条件
4. **内存问题**: 减少 `concurrent_requests` 或 `max_tokens_per_step`

### 调试

启用详细日志记录：
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或只为特定组件启用
logging.getLogger("workers.tools").setLevel(logging.DEBUG)
```

保存轨迹用于分析：
```python
config = AgenticRolloutConfig(
    save_trajectories=True,
    trajectory_save_path="./debug_trajectories"
)
```

## 📄 许可证

该项目遵循与VERL库相同的许可证。

## ❓ 常见问题

**Q: 我可以在没有VERL的情况下使用这个库吗？**
A: 可以！该库设计为完全独立工作，并在VERL不可用时优雅降级。

**Q: 支持哪些LLM提供商？**
A: 任何遵循OpenAI SDK协议的提供商。我们为OpenAI、Claude和自定义实现提供示例。

**Q: 如何添加自定义工具？**
A: 创建一个带有async `execute_tool` 方法的类，并使用 `register_tool()` 注册到智能体。

**Q: 我可以将此用于生产工作负载吗？**
A: 可以，该库包含适合生产使用的全面错误处理、日志记录和性能优化。

## 🆘 支持

- 📖 查看 [examples目录](./examples/) 了解使用模式
- 📖 查看 [中文示例文档](./examples/README_CN.md) 了解详细用法
- 🐛 在项目的issue tracker中报告问题
- 💬 加入社区讨论以获取问题和功能请求的帮助

---

**为AI智能体社区倾心打造 ❤️**