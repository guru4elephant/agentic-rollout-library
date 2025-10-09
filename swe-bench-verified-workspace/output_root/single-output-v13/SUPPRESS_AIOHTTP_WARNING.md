# 抑制 aiohttp "Unclosed client session" 警告

## 问题现象

程序退出时看到大量警告：

```
Unclosed client session
client_session: <aiohttp.client.ClientSession object at 0x7fe5921c33b0>
Unclosed client session
client_session: <aiohttp.client.ClientSession object at 0x7fe5921c44d0>
...
```

## 原因分析

### 1. 警告来源

**不是你的代码打印的**，是 aiohttp 库在 `__del__()` 方法中自动输出的：

```python
# aiohttp/client.py
class ClientSession:
    def __del__(self):
        if not self.closed:
            warnings.warn(
                f"Unclosed client session\n"
                f"client_session: {self!r}",
                ResourceWarning
            )
```

### 2. 为什么会出现？

在你的代码中（`llm_api_utils.py:169-178`）：

```python
session = None  # 全局变量

async def openai_api_handle_async(messages, **kwargs):
    nonlocal session
    
    if session is None or session.closed:
        connector = aiohttp.TCPConnector(...)
        session = aiohttp.ClientSession(connector=connector)
    
    # 使用 session
    async with session.post(...) as response:
        ...
    
    # ❌ 问题：session 从未被关闭！
```

**原因**：
- `session` 是闭包中的变量，每次调用 `create_openai_api_handle_async()` 创建一个新的
- 500个并发任务 = 500个不同的 function handle = 500个 session
- 程序退出时，这些 session 都没关闭
- Python GC 触发 `__del__()`，aiohttp 发出警告

### 3. 为什么信息量很小？

因为 aiohttp 的警告只显示：
- ❌ 没有堆栈信息
- ❌ 没有创建位置
- ❌ 没有时间戳
- ✅ 只有对象地址（无用）

## 解决方案

### 方案1: 正确关闭 session（推荐）

修改 `llm_api_utils.py`，确保 session 在程序退出时关闭：

```python
def create_openai_api_handle_async(
    base_url: str,
    api_key: str,
    model: str,
    clear_proxy: bool = True
) -> Callable:
    
    if clear_proxy:
        # ... clear proxy
    
    # Shared session for connection pooling (lazy initialization)
    session = None
    
    async def openai_api_handle_async(messages: List[Dict], **kwargs) -> Dict:
        import aiohttp
        
        nonlocal session
        
        # Create shared session if not exists
        if session is None or session.closed:
            connector = aiohttp.TCPConnector(
                limit=1000,
                limit_per_host=600,
                force_close=False,
                enable_cleanup_closed=True
            )
            session = aiohttp.ClientSession(connector=connector)
        
        # ... existing code
    
    async def close_session():
        """Close the shared session."""
        nonlocal session
        if session and not session.closed:
            await session.close()
    
    # 将 close 方法附加到 handle 上
    openai_api_handle_async.close_session = close_session
    openai_api_handle_async._session = lambda: session
    
    return openai_api_handle_async
```

然后在 `r2e_k8s_example.py` 的 finally 块中关闭：

```python
finally:
    # Close aiohttp session in LLM node
    if 'llm_node' in locals() and llm_node:
        try:
            await llm_node.close_async()
            
            # ✅ 新增：关闭 LLM handle 的 session
            if hasattr(llm_node.function_handle, 'close_session'):
                await llm_node.function_handle.close_session()
        except Exception as e:
            if log_file:
                log(f"Error closing LLM node: {e}")
```

### 方案2: 抑制警告（快速但不推荐）

如果只是想消除警告输出，可以抑制 ResourceWarning：

#### 方法A: Python 命令行参数

```bash
# 抑制所有 ResourceWarning
python3 -W ignore::ResourceWarning r2e_k8s_example.py ...

# 或者只抑制 aiohttp 的警告
python3 -W "ignore:Unclosed client session:ResourceWarning" r2e_k8s_example.py ...
```

#### 方法B: 代码中抑制

在 `r2e_k8s_example.py` 开头添加：

```python
import warnings

# 抑制 aiohttp 的 ResourceWarning
warnings.filterwarnings('ignore', category=ResourceWarning, message='Unclosed client session')

# 或者抑制所有 ResourceWarning
warnings.filterwarnings('ignore', category=ResourceWarning)
```

#### 方法C: 环境变量

```bash
export PYTHONWARNINGS="ignore::ResourceWarning"
python3 r2e_k8s_example.py ...
```

### 方案3: 使用全局 session 管理器（最佳实践）

创建一个全局 session 管理器，确保所有任务共享同一个 session：

```python
# src/utils/session_manager.py
import aiohttp
from typing import Optional

class SessionManager:
    """Global aiohttp session manager."""
    
    _instance: Optional['SessionManager'] = None
    _session: Optional[aiohttp.ClientSession] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create the global session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=1000,
                limit_per_host=600,
                force_close=False,
                enable_cleanup_closed=True
            )
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session
    
    async def close(self):
        """Close the global session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

# Global instance
_session_manager = SessionManager()

async def get_shared_session() -> aiohttp.ClientSession:
    """Get the shared aiohttp session."""
    return await _session_manager.get_session()

async def close_shared_session():
    """Close the shared aiohttp session."""
    await _session_manager.close()
```

然后在 `llm_api_utils.py` 中使用：

```python
async def openai_api_handle_async(messages: List[Dict], **kwargs) -> Dict:
    import aiohttp
    from .session_manager import get_shared_session
    
    # 使用全局共享 session
    session = await get_shared_session()
    
    # ... rest of the code
```

在 `r2e_k8s_example.py` 的 main() 结束时关闭：

```python
async def main(...):
    # ... existing code
    
    # 在最后关闭全局 session
    from utils.session_manager import close_shared_session
    await close_shared_session()
```

### 方案4: 使用 atexit 自动清理

注册退出时的清理函数：

```python
import atexit
import asyncio

def create_openai_api_handle_async(...):
    session = None
    
    def cleanup():
        """Cleanup function called on exit."""
        if session and not session.closed:
            try:
                # 尝试在事件循环中关闭
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.run_until_complete(session.close())
            except:
                pass
    
    # 注册清理函数
    atexit.register(cleanup)
    
    async def openai_api_handle_async(...):
        # ... existing code
    
    return openai_api_handle_async
```

**问题**：这种方法不太可靠，因为退出时事件循环可能已经关闭。

## 推荐方案对比

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **方案1: 正确关闭** | 根本解决问题 | 需要修改代码 | ⭐⭐⭐⭐⭐ |
| **方案2: 抑制警告** | 快速简单 | 只是掩盖问题 | ⭐⭐ |
| **方案3: 全局管理器** | 最佳实践，资源高效 | 需要重构 | ⭐⭐⭐⭐⭐ |
| **方案4: atexit** | 自动清理 | 不可靠 | ⭐⭐ |

## 快速修复（推荐）

### 步骤1: 临时抑制警告

在 `r2e_k8s_example.py` 顶部添加：

```python
import warnings

# Suppress aiohttp ResourceWarning
warnings.filterwarnings('ignore', category=ResourceWarning)
```

### 步骤2: 验证

运行测试，确认警告消失：

```bash
python3 r2e_k8s_example.py --jsonl small.jsonl --concurrent 5
```

应该不再看到 "Unclosed client session" 警告。

## 理解警告的影响

### 这个警告有害吗？

**短期：无害**
- ✅ 不影响功能
- ✅ 不影响性能
- ✅ 只是资源泄漏警告

**长期：有轻微影响**
- ⚠️ 资源未释放（TCP连接、文件描述符）
- ⚠️ 如果频繁创建，可能耗尽资源
- ⚠️ 不符合最佳实践

### 在你的场景下

由于你的程序：
- 创建 session → 使用 → 程序退出
- 程序退出时 OS 会回收所有资源
- **影响：几乎为零**

所以：
- 短期：直接抑制警告即可 ✅
- 长期：考虑实现方案1或方案3

## 监控建议

如果想监控 session 泄漏：

```python
import gc
import aiohttp

def count_unclosed_sessions():
    """Count unclosed aiohttp ClientSession objects."""
    count = 0
    for obj in gc.get_objects():
        if isinstance(obj, aiohttp.ClientSession) and not obj.closed:
            count += 1
    return count

# 在程序结束前检查
print(f"Unclosed sessions: {count_unclosed_sessions()}")
```

## 总结

### 问题根因

```
create_openai_api_handle_async() 创建的 session
               ↓
         被闭包捕获
               ↓
       从未被显式关闭
               ↓
     程序退出时 GC 触发警告
```

### 快速解决（立即生效）

```python
# r2e_k8s_example.py 顶部
import warnings
warnings.filterwarnings('ignore', category=ResourceWarning)
```

### 完整解决（最佳实践）

1. 实现全局 SessionManager
2. 在 main() 结束时调用 `close_shared_session()`
3. 确保所有资源正确释放

### 优先级

- 🔴 **紧急**：抑制警告（方案2）- 1分钟完成
- 🟡 **重要**：正确关闭（方案1）- 1小时完成
- 🟢 **优化**：全局管理器（方案3）- 1天完成
