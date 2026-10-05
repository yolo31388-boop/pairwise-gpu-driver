"""
游戏内GPU驱动与命令缓冲系统 - 核心模块

包含：
- GPUDriver：GPU驱动（设备管理 + 内存管理）
- CommandBuffer：命令缓冲（录制 + 提交）
- CommandQueue：命令队列（优先级 + 调度）
- Fence / Semaphore：同步原语（栅栏 + 信号量）
- MemoryBarrier：内存屏障（数据屏障 + 指令屏障）
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import heapq
import itertools
import threading
import time


@dataclass
class Config:
    pass


# ---------------------------------------------------------------------------
# GPU 设备管理
# ---------------------------------------------------------------------------

@dataclass
class GPUDevice:
    """GPU设备描述。"""
    device_id: int
    name: str = "GPU"
    memory_total: int = 0
    is_active: bool = True


# ---------------------------------------------------------------------------
# GPU 内存管理
# ---------------------------------------------------------------------------

@dataclass
class MemoryBlock:
    """一块GPU显存。"""
    handle: int
    offset: int
    size: int
    data: bytearray = field(default_factory=bytearray)


class MemoryManager:
    """GPU内存管理：分配 / 释放 / 读写 / 统计。"""

    def __init__(self, capacity: int = 1 << 30):
        if capacity <= 0:
            raise ValueError("capacity 必须为正数")
        self.capacity = capacity
        self._used = 0
        self._next_handle = 1
        self._blocks: Dict[int, MemoryBlock] = {}
        self._offset_cursor = 0

    @property
    def used(self) -> int:
        return self._used

    @property
    def free(self) -> int:
        return self.capacity - self._used

    def allocate(self, size: int) -> int:
        """分配 size 字节显存，返回句柄；显存不足抛出 MemoryError。"""
        if size <= 0:
            raise ValueError("分配大小必须为正数")
        if size > self.free:
            raise MemoryError("GPU显存不足")
        handle = self._next_handle
        self._next_handle += 1
        block = MemoryBlock(handle=handle, offset=self._offset_cursor,
                            size=size, data=bytearray(size))
        self._offset_cursor += size
        self._blocks[handle] = block
        self._used += size
        return handle

    def release(self, handle: int) -> None:
        """释放显存块。"""
        block = self._blocks.pop(handle, None)
        if block is None:
            raise KeyError(f"无效的显存句柄: {handle}")
        self._used -= block.size

    def write(self, handle: int, data: bytes, offset: int = 0) -> None:
        block = self._require(handle)
        if offset < 0 or offset + len(data) > block.size:
            raise ValueError("写入越界")
        block.data[offset:offset + len(data)] = data

    def read(self, handle: int, size: Optional[int] = None, offset: int = 0) -> bytes:
        block = self._require(handle)
        size = block.size - offset if size is None else size
        if offset < 0 or offset + size > block.size:
            raise ValueError("读取越界")
        return bytes(block.data[offset:offset + size])

    def _require(self, handle: int) -> MemoryBlock:
        block = self._blocks.get(handle)
        if block is None:
            raise KeyError(f"无效的显存句柄: {handle}")
        return block

    def reset(self) -> None:
        self._used = 0
        self._next_handle = 1
        self._blocks.clear()
        self._offset_cursor = 0


# ---------------------------------------------------------------------------
# 命令缓冲：录制 + 提交
# ---------------------------------------------------------------------------

class CommandBuffer:
    """命令缓冲：先录制GPU命令，再整体提交到命令队列执行。"""

    IDLE = "idle"
    RECORDING = "recording"
    RECORDED = "recorded"
    SUBMITTED = "submitted"
    EXECUTED = "executed"

    def __init__(self, name: str = "command_buffer"):
        self.name = name
        self.state = self.IDLE
        self._commands: List[Tuple[str, Dict[str, Any]]] = []

    @property
    def commands(self) -> List[Tuple[str, Dict[str, Any]]]:
        return list(self._commands)

    def __len__(self) -> int:
        return len(self._commands)

    def begin_recording(self) -> None:
        """开始录制。录制前自动清空旧命令。"""
        if self.state == self.SUBMITTED:
            raise RuntimeError("命令缓冲已提交，不能重复录制")
        self._commands.clear()
        self.state = self.RECORDING

    def record(self, command: str, **params: Any) -> None:
        """录制一条GPU命令。"""
        if self.state != self.RECORDING:
            raise RuntimeError("命令缓冲不在录制状态，请先 begin_recording()")
        if not command:
            raise ValueError("命令名不能为空")
        self._commands.append((command, dict(params)))

    def end_recording(self) -> None:
        """结束录制，缓冲进入可提交状态。"""
        if self.state != self.RECORDING:
            raise RuntimeError("命令缓冲不在录制状态")
        self.state = self.RECORDED

    def submit(self, queue: "CommandQueue", priority: int = 0) -> None:
        """把录制好的命令缓冲提交到命令队列。"""
        if self.state != self.RECORDED:
            raise RuntimeError("命令缓冲未录制完成，不能提交")
        queue.enqueue(self, priority=priority)
        self.state = self.SUBMITTED

    def reset(self) -> None:
        self._commands.clear()
        self.state = self.IDLE


# ---------------------------------------------------------------------------
# 命令队列：优先级 + 调度
# ---------------------------------------------------------------------------

class CommandQueue:
    """命令队列：按优先级调度执行命令缓冲（数值越小优先级越高）。"""

    def __init__(self, name: str = "graphics"):
        self.name = name
        self._counter = itertools.count()
        self._pending: List[Tuple[int, int, CommandBuffer]] = []
        self._executed: List[CommandBuffer] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def executed(self) -> List[CommandBuffer]:
        return list(self._executed)

    def enqueue(self, buffer: CommandBuffer, priority: int = 0) -> None:
        """按优先级入队一个命令缓冲。"""
        heapq.heappush(self._pending, (priority, next(self._counter), buffer))

    def schedule(self) -> Optional[CommandBuffer]:
        """调度：取出优先级最高（同优先级先入先出）的缓冲。"""
        if not self._pending:
            return None
        return heapq.heappop(self._pending)[2]

    def execute_next(self, executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None
                     ) -> Optional[CommandBuffer]:
        """调度并执行下一个命令缓冲，返回被执行的缓冲。"""
        buffer = self.schedule()
        if buffer is None:
            return None
        for command, params in buffer.commands:
            if executor is not None:
                executor(command, params)
        buffer.state = CommandBuffer.EXECUTED
        self._executed.append(buffer)
        return buffer

    def execute_all(self, executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None
                    ) -> List[CommandBuffer]:
        """按优先级顺序执行队列中全部命令缓冲。"""
        done = []
        while self._pending:
            done.append(self.execute_next(executor))
        return done

    def reset(self) -> None:
        self._pending.clear()
        self._executed.clear()


# ---------------------------------------------------------------------------
# 同步原语：栅栏 + 信号量
# ---------------------------------------------------------------------------

class Fence:
    """栅栏：GPU完成工作后signal，CPU通过wait同步。"""

    def __init__(self, signaled: bool = False):
        self._signaled = signaled
        self._cond = threading.Condition()

    @property
    def signaled(self) -> bool:
        return self._signaled

    def signal(self) -> None:
        with self._cond:
            self._signaled = True
            self._cond.notify_all()

    def reset(self) -> None:
        with self._cond:
            self._signaled = False

    def wait(self, timeout: Optional[float] = None) -> bool:
        """等待栅栏被置位，返回是否在超时前等到。"""
        with self._cond:
            if not self._signaled:
                self._cond.wait_for(lambda: self._signaled, timeout=timeout)
            return self._signaled


class Semaphore:
    """信号量：GPU/CPU之间的计数同步。"""

    def __init__(self, value: int = 0):
        if value < 0:
            raise ValueError("信号量初值不能为负")
        self._value = value
        self._cond = threading.Condition()

    @property
    def value(self) -> int:
        return self._value

    def signal(self, count: int = 1) -> None:
        with self._cond:
            self._value += count
            self._cond.notify_all()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """等待并消费一个计数，返回是否成功。"""
        with self._cond:
            if self._value <= 0:
                self._cond.wait_for(lambda: self._value > 0, timeout=timeout)
            if self._value <= 0:
                return False
            self._value -= 1
            return True


@dataclass
class SynchronizationPrimitives:
    """同步原语集合：栅栏 + 信号量。"""
    fence: Fence = field(default_factory=Fence)
    semaphore: Semaphore = field(default_factory=Semaphore)


# ---------------------------------------------------------------------------
# 内存屏障：数据屏障 + 指令屏障
# ---------------------------------------------------------------------------

class MemoryBarrier:
    """内存屏障：保证GPU/CPU之间的内存可见性与指令顺序。"""

    DATA = "data"
    INSTRUCTION = "instruction"
    FULL = "full"

    def __init__(self):
        self._barriers: List[Tuple[str, float]] = []
        self._lock = threading.Lock()

    @property
    def barriers(self) -> List[str]:
        with self._lock:
            return [kind for kind, _ in self._barriers]

    def data_barrier(self) -> None:
        """数据屏障：保证屏障前的数据写入对屏障后的读取可见。"""
        self._insert(self.DATA)

    def instruction_barrier(self) -> None:
        """指令屏障：保证屏障前的指令先于屏障后的指令执行。"""
        self._insert(self.INSTRUCTION)

    def full_barrier(self) -> None:
        """全屏障：同时保证数据可见性与指令顺序。"""
        self._insert(self.FULL)

    def _insert(self, kind: str) -> None:
        with self._lock:
            # 写后刷新，模拟屏障使此前的写入对其他观察者可见
            self._barriers.append((kind, time.monotonic()))

    def reset(self) -> None:
        with self._lock:
            self._barriers.clear()


# ---------------------------------------------------------------------------
# GPU 驱动：设备管理 + 内存管理 + 各子系统入口
# ---------------------------------------------------------------------------

class GPUDriver:
    """游戏内GPU驱动：统一管理设备、显存、命令缓冲、命令队列与同步。"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state: Dict[str, Any] = {"frame": 0, "running": True}
        self._history: List[str] = []

        capacity = int(self.config.get("memory_capacity", 1 << 30))
        self.memory = MemoryManager(capacity=capacity)
        self.queue = CommandQueue(name=self.config.get("queue_name", "graphics"))
        self.sync = SynchronizationPrimitives()
        self.barrier = MemoryBarrier()

        self._devices: Dict[int, GPUDevice] = {}
        self._next_device_id = 0
        self._buffers: List[CommandBuffer] = []

    # -- 设备管理 ---------------------------------------------------------

    def add_device(self, name: str = "GPU", memory_total: int = 0) -> GPUDevice:
        """注册一个GPU设备并返回它。"""
        device = GPUDevice(device_id=self._next_device_id, name=name,
                           memory_total=memory_total)
        self._devices[device.device_id] = device
        self._next_device_id += 1
        self._history.append(f"add_device:{device.device_id}")
        return device

    def remove_device(self, device_id: int) -> None:
        """移除一个GPU设备。"""
        if device_id not in self._devices:
            raise KeyError(f"设备不存在: {device_id}")
        del self._devices[device_id]
        self._history.append(f"remove_device:{device_id}")

    def get_device(self, device_id: int) -> GPUDevice:
        device = self._devices.get(device_id)
        if device is None:
            raise KeyError(f"设备不存在: {device_id}")
        return device

    @property
    def devices(self) -> List[GPUDevice]:
        return list(self._devices.values())

    @property
    def device_count(self) -> int:
        return len(self._devices)

    # -- 内存管理 ---------------------------------------------------------

    def allocate_memory(self, size: int) -> int:
        """分配GPU显存，返回句柄。"""
        handle = self.memory.allocate(size)
        self._history.append(f"allocate:{handle}:{size}")
        return handle

    def free_memory(self, handle: int) -> None:
        """释放GPU显存。"""
        self.memory.release(handle)
        self._history.append(f"free:{handle}")

    @property
    def memory_used(self) -> int:
        return self.memory.used

    @property
    def memory_free(self) -> int:
        return self.memory.free

    # -- 命令缓冲 ---------------------------------------------------------

    def command_buffer(self, name: str = "command_buffer") -> CommandBuffer:
        """创建一个新的命令缓冲。"""
        buffer = CommandBuffer(name=name)
        self._buffers.append(buffer)
        return buffer

    # -- 命令队列 ---------------------------------------------------------

    def command_queue(self) -> CommandQueue:
        """返回驱动管理的命令队列。"""
        return self.queue

    def submit(self, buffer: CommandBuffer, priority: int = 0) -> None:
        """把命令缓冲提交到驱动的命令队列。"""
        buffer.submit(self.queue, priority=priority)
        self._history.append(f"submit:{buffer.name}:{priority}")

    def dispatch(self) -> List[CommandBuffer]:
        """调度执行队列中所有命令缓冲，并置位栅栏通知CPU。"""
        done = self.queue.execute_all()
        if done:
            self.sync.fence.signal()
            self.sync.semaphore.signal(len(done))
        self._history.append(f"dispatch:{len(done)}")
        return done

    # -- 同步原语 / 内存屏障 ----------------------------------------------

    def synchronization_primitives(self) -> SynchronizationPrimitives:
        """返回驱动的同步原语（栅栏 + 信号量）。"""
        return self.sync

    def memory_barrier(self) -> MemoryBarrier:
        """返回驱动的内存屏障管理器。"""
        return self.barrier

    # -- 帧循环 -----------------------------------------------------------

    def update(self, dt: float):
        """推进一帧：调度执行已提交的命令缓冲。"""
        if not self._state.get("running"):
            return
        self._state["frame"] += 1
        self._state["last_dt"] = dt
        self.dispatch()

    @property
    def frame(self) -> int:
        return self._state["frame"]

    @property
    def history(self) -> List[str]:
        return list(self._history)

    def reset(self):
        self._state = {"frame": 0, "running": True}
        self._history = []
        self.memory.reset()
        self.queue.reset()
        self.barrier.reset()
        self.sync = SynchronizationPrimitives()
        self._devices = {}
        self._next_device_id = 0
        self._buffers = []
