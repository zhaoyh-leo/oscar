"""ProgressHandler — 事件驱动进度反馈，解耦业务逻辑与 UI。

设计原则：
- 观察者模式：业务逻辑只触发事件，不关心谁消费。
- 核心抽象：BaseProgressHandler 定义接口，RichProgressHandler 消费并渲染。
- LangGraph 集成：通过 Callbacks 机制自动注入每个 Node 的生命周期。
- 零 UI 依赖：核心审计代码不 import 任何 UI 库（rich / tqdm）。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from uuid import UUID


# ---------------------------------------------------------------------------
# 事件类型（纯数据，不依赖 UI）
# ---------------------------------------------------------------------------

class ProgressEvent:
    """一次进度事件，由业务逻辑触发，由 Handler 消费。"""

    def __init__(
        self,
        event_type: str,            # on_node_start / on_node_end / on_token_usage
        node_name: str = "",
        message: str = "",
        total: int = 0,
        current: int = 0,
        elapsed: float = 0.0,
        metadata: dict | None = None,
    ):
        self.event_type = event_type
        self.node_name = node_name
        self.message = message
        self.total = total
        self.current = current
        self.elapsed = elapsed
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# 抽象 Handler
# ---------------------------------------------------------------------------

class BaseProgressHandler(ABC):
    """进度处理器的抽象基类。业务逻辑层只依赖这个抽象。"""

    @abstractmethod
    def on_event(self, event: ProgressEvent):
        ...


# ---------------------------------------------------------------------------
# LangGraph Callback — 自动注入 Node 生命周期
# ---------------------------------------------------------------------------

class AuditCallbackHandler(BaseCallbackHandler):
    """消费 LangGraph Custom Events，将节点生命周期转为 ProgressEvent。

    与旧实现（监听全局 on_chain_start 并猜测运行树）相比：
    - on_chain_start 是「运行级」回调：LangGraph 编译图的根运行、节点内部的
      LLM 调用等都会触发它，且图根 serialized 按设计为 None —— 从它反推
      “哪个节点开始”在机制上就是不成立的，崩溃并非异常而是必然。
    - 本实现由节点通过 adispatch_custom_event 显式派发事件，handler 通过
      on_custom_event 精确接收。事件源与业务一一对应，无需任何猜测/过滤。
    """

    def __init__(self, handler: BaseProgressHandler):
        super().__init__()
        self._handler = handler
        self._timers: dict[str, float] = {}

    def on_custom_event(
        self, event_name: str, data: Any, *, run_id: UUID, config: Any = None, **kwargs
    ) -> None:
        """LangChain 在节点调用 adispatch_custom_event 时回调本方法。"""
        node = (data or {}).get("node_name", "unknown")

        if event_name == "on_node_start":
            self._timers[node] = time.time()
            self._handler.on_event(ProgressEvent(
                event_type="on_node_start",
                node_name=node,
                message=f"Starting {node}",
            ))
        elif event_name == "on_node_end":
            elapsed = time.time() - self._timers.pop(node, time.time())
            self._handler.on_event(ProgressEvent(
                event_type="on_node_end",
                node_name=node,
                message=f"Completed {node}",
                elapsed=elapsed,
            ))


# ---------------------------------------------------------------------------
# Rich 实现（仅当需要 CLI 渲染时才实例化）
# ---------------------------------------------------------------------------

class RichProgressHandler(BaseProgressHandler):
    """将 ProgressEvent 渲染为 rich 进度条。

    仅在 CLI 模式下使用；API 服务模式下可替换为 WebSocket 推送。
    """

    def __init__(self):
        self._console: Any = None
        self._progress: Any = None
        self._tasks: dict[str, Any] = {}
        self._node_order: list[str] = []
        self._step_count = 0

    def _lazy_init(self):
        if self._console is None:
            from rich.console import Console
            from rich.progress import (
                BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
            )
            self._console = Console()
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self._console,
                transient=False,
            )
            self._progress.start()

    def on_event(self, event: ProgressEvent):
        self._lazy_init()

        if event.event_type == "on_node_start":
            self._step_count += 1
            step = self._step_count
            desc = f"[cyan]Step {step}/{self._step_count + 5}: {event.message}[/cyan]"
            task = self._progress.add_task(desc, total=100)
            self._tasks[event.node_name] = task
            self._progress.update(task, advance=10)

        elif event.event_type == "on_node_end":
            task = self._tasks.pop(event.node_name, None)
            if task is not None:
                self._progress.update(
                    task,
                    description=f"[green]✓ {event.node_name} ({event.elapsed:.1f}s)[/green]",
                    completed=100,
                )
                self._progress.stop_task(task)
                self._progress.remove_task(task)

    def stop(self):
        if self._progress:
            self._progress.stop()
            self._console.print("[bold green]Audit complete![/bold green]")


# ---------------------------------------------------------------------------
# 空 Handler（用于测试 / API 模式）
# ---------------------------------------------------------------------------

class NullProgressHandler(BaseProgressHandler):
    """什么都不做，用于单元测试或 API 服务模式。"""

    def on_event(self, event: ProgressEvent):
        pass


# 全局单例（默认空实现）
progress_handler: BaseProgressHandler = NullProgressHandler()