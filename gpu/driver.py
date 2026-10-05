"""
游戏内GPU驱动与命令缓冲系统 - 核心模块

提供：
- GPU驱动：设备管理 + 内存管理
- 命令缓冲：录制 + 提交
- 命令队列：优先级 + 调度
- 同步原语：栅栏（fence）+ 信号量（semaphore）
- 内存屏障：数据屏障 + 指令屏障
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import itertools
import math
import time


@dataclass
class Config:
    pass


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class GPUDevice:
    """GPU设备：驱动管理的最小硬件单元。"""
    device_id: int
    name: str
    total_memory: int
    used_memory: int = 0
    queues: List[int] = field(default_factory=list)
    buffers: List[int] = field(default_factory=list)


@dataclass
class MemoryAllocation:
    """一块已分配的GPU内存。"""
    alloc_id: int
    device_id: int
    size: int
    usage: str = "device"


@dataclass
class CommandBuffer:
    """命令缓冲：initial -> recording -> executable -> submitted -> executed。"""
    buffer_id: int
    device_id: int
    state: str = "initial"
    commands: List[Tuple] = field(default_factory=list)


@dataclass
class CommandQueue:
    """命令队列：按优先级调度，数值越大优先级越高。"""
    queue_id: int
    device_id: int
    priority: int = 0
    pending: List[Tuple[int, Optional[int], Optional[int]]] = field(default_factory=list)


@dataclass
class Fence:
    """栅栏：用于GPU->CPU的单次同步。"""
    fence_id: int
    device_id: int
    signaled: bool = False


@dataclass
class Semaphore:
    """信号量：用于GPU内部/CPU-GPU间的计数同步。"""
    semaphore_id: int
    device_id: int
    value: int = 0


# ---------------------------------------------------------------------------
# GPU驱动
# ---------------------------------------------------------------------------

class GPUDriver:
    """游戏内GPU驱动：管理设备、内存、命令缓冲、命令队列与同步原语。"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history: List[Tuple] = []
        self._devices: Dict[int, GPUDevice] = {}
        self._allocations: Dict[int, MemoryAllocation] = {}
        self._buffers: Dict[int, CommandBuffer] = {}
        self._queues: Dict[int, CommandQueue] = {}
        self._fences: Dict[int, Fence] = {}
        self._semaphores: Dict[int, Semaphore] = {}
        self._ids = itertools.count(1)
        # 内存可见性模型：写入先进入pending，数据屏障之后才visible
        self._pending_writes: Dict[int, Dict[int, object]] = {}
        self._visible_data: Dict[int, Dict[int, object]] = {}
        # 指令顺序日志：设备上实际执行的指令/屏障序列
        self._instruction_log: Dict[int, List[str]] = {}

    # ------------------------------------------------------------------
    # 设备管理
    # ------------------------------------------------------------------

    def create_device(self, name: str = "gpu", total_memory: int = 1024) -> int:
        """创建并注册一个GPU设备，返回设备ID。"""
        if total_memory <= 0:
            raise ValueError("total_memory必须为正数")
        device_id = next(self._ids)
        self._devices[device_id] = GPUDevice(
            device_id=device_id, name=name, total_memory=total_memory,
        )
        self._pending_writes[device_id] = {}
        self._visible_data[device_id] = {}
        self._instruction_log[device_id] = []
        self._history.append(("create_device", device_id))
        return device_id

    def get_device(self, device_id: int) -> Optional[GPUDevice]:
        return self._devices.get(device_id)

    def list_devices(self) -> List[int]:
        return sorted(self._devices.keys())

    def destroy_device(self, device_id: int) -> bool:
        """销毁设备，并释放其上的内存、队列与缓冲。"""
        device = self._devices.get(device_id)
        if device is None:
            return False
        for alloc_id in [a for a, al in self._allocations.items()
                         if al.device_id == device_id]:
            del self._allocations[alloc_id]
        for queue_id in list(device.queues):
            del self._queues[queue_id]
        for buffer_id in list(device.buffers):
            del self._buffers[buffer_id]
        del self._devices[device_id]
        self._pending_writes.pop(device_id, None)
        self._visible_data.pop(device_id, None)
        self._instruction_log.pop(device_id, None)
        self._history.append(("destroy_device", device_id))
        return True

    def _require_device(self, device_id: int) -> GPUDevice:
        device = self._devices.get(device_id)
        if device is None:
            raise ValueError(f"设备不存在: {device_id}")
        return device

    # ------------------------------------------------------------------
    # 内存管理
    # ------------------------------------------------------------------

    def allocate_memory(self, device_id: int, size: int,
                        usage: str = "device") -> Optional[int]:
        """在设备上分配显存，成功返回分配ID，显存不足返回None。"""
        device = self._require_device(device_id)
        if size <= 0:
            raise ValueError("size必须为正数")
        if device.used_memory + size > device.total_memory:
            return None
        alloc_id = next(self._ids)
        self._allocations[alloc_id] = MemoryAllocation(
            alloc_id=alloc_id, device_id=device_id, size=size, usage=usage,
        )
        device.used_memory += size
        self._history.append(("allocate_memory", device_id, alloc_id, size))
        return alloc_id

    def free_memory(self, device_id: int, alloc_id: int) -> bool:
        """释放一块显存。"""
        device = self._require_device(device_id)
        alloc = self._allocations.get(alloc_id)
        if alloc is None or alloc.device_id != device_id:
            return False
        device.used_memory -= alloc.size
        del self._allocations[alloc_id]
        self._history.append(("free_memory", device_id, alloc_id))
        return True

    def memory_info(self, device_id: int) -> Dict[str, int]:
        device = self._require_device(device_id)
        return {
            "total": device.total_memory,
            "used": device.used_memory,
            "free": device.total_memory - device.used_memory,
        }

    # ------------------------------------------------------------------
    # 命令缓冲：录制 + 提交
    # ------------------------------------------------------------------

    def create_command_buffer(self, device_id: int) -> int:
        device = self._require_device(device_id)
        buffer_id = next(self._ids)
        self._buffers[buffer_id] = CommandBuffer(
            buffer_id=buffer_id, device_id=device_id,
        )
        device.buffers.append(buffer_id)
        return buffer_id

    def get_command_buffer(self, buffer_id: int) -> Optional[CommandBuffer]:
        return self._buffers.get(buffer_id)

    def _require_buffer(self, buffer_id: int) -> CommandBuffer:
        buf = self._buffers.get(buffer_id)
        if buf is None:
            raise ValueError(f"命令缓冲不存在: {buffer_id}")
        return buf

    def begin_recording(self, buffer_id: int) -> None:
        """开始录制：initial/executable -> recording。"""
        buf = self._require_buffer(buffer_id)
        if buf.state not in ("initial", "executable"):
            raise RuntimeError(f"当前状态不能开始录制: {buf.state}")
        if buf.state == "initial":
            buf.commands.clear()
        buf.state = "recording"

    def record_command(self, buffer_id: int, command: str, **params) -> None:
        """录制一条GPU命令，只能在recording状态下调用。"""
        buf = self._require_buffer(buffer_id)
        if buf.state != "recording":
            raise RuntimeError("命令缓冲不在录制状态")
        buf.commands.append((command, params))

    def record_write(self, buffer_id: int, address: int, data: object) -> None:
        """录制一条显存写入命令。"""
        self.record_command(buffer_id, "write", address=address, data=data)

    def record_data_barrier(self, buffer_id: int) -> None:
        """录制数据屏障：保证屏障前的写入对后续读取可见。"""
        self.record_command(buffer_id, "data_barrier")

    def record_instruction_barrier(self, buffer_id: int) -> None:
        """录制指令屏障：保证屏障前后的指令不被重排。"""
        self.record_command(buffer_id, "instruction_barrier")

    def end_recording(self, buffer_id: int) -> None:
        """结束录制：recording -> executable，之后才能提交。"""
        buf = self._require_buffer(buffer_id)
        if buf.state != "recording":
            raise RuntimeError(f"当前状态不能结束录制: {buf.state}")
        buf.state = "executable"

    def submit_command_buffer(self, queue_id: int, buffer_id: int,
                              fence_id: Optional[int] = None,
                              semaphore_id: Optional[int] = None) -> None:
        """把录制完成的命令缓冲提交到命令队列。

        可选地关联一个栅栏/信号量，缓冲执行完成时自动触发。
        """
        queue = self._queues.get(queue_id)
        if queue is None:
            raise ValueError(f"命令队列不存在: {queue_id}")
        buf = self._require_buffer(buffer_id)
        if buf.state != "executable":
            raise RuntimeError(f"命令缓冲不可提交，当前状态: {buf.state}")
        if buf.device_id != queue.device_id:
            raise ValueError("命令缓冲与队列不属于同一设备")
        if fence_id is not None and fence_id not in self._fences:
            raise ValueError(f"栅栏不存在: {fence_id}")
        if semaphore_id is not None and semaphore_id not in self._semaphores:
            raise ValueError(f"信号量不存在: {semaphore_id}")
        buf.state = "submitted"
        queue.pending.append((buffer_id, fence_id, semaphore_id))
        self._history.append(("submit", queue_id, buffer_id))

    # ------------------------------------------------------------------
    # 命令队列：优先级 + 调度
    # ------------------------------------------------------------------

    def create_command_queue(self, device_id: int, priority: int = 0) -> int:
        """创建命令队列，priority数值越大优先级越高。"""
        device = self._require_device(device_id)
        queue_id = next(self._ids)
        self._queues[queue_id] = CommandQueue(
            queue_id=queue_id, device_id=device_id, priority=priority,
        )
        device.queues.append(queue_id)
        return queue_id

    def get_command_queue(self, queue_id: int) -> Optional[CommandQueue]:
        return self._queues.get(queue_id)

    def _scheduled_queues(self) -> List[CommandQueue]:
        """按调度顺序返回非空队列：优先级降序，同优先级按创建先后。"""
        queues = [q for q in self._queues.values() if q.pending]
        queues.sort(key=lambda q: (-q.priority, q.queue_id))
        return queues

    def execute_next(self) -> Optional[int]:
        """调度器：从最高优先级的非空队列取一个缓冲执行，返回缓冲ID。"""
        queues = self._scheduled_queues()
        if not queues:
            return None
        queue = queues[0]
        buffer_id, fence_id, semaphore_id = queue.pending.pop(0)
        self._execute_buffer(buffer_id, fence_id, semaphore_id)
        return buffer_id

    def _execute_buffer(self, buffer_id: int,
                        fence_id: Optional[int],
                        semaphore_id: Optional[int]) -> None:
        buf = self._buffers[buffer_id]
        device_id = buf.device_id
        for command, params in buf.commands:
            if command == "write":
                # 写入先进入pending区，数据屏障之后才可见
                self._pending_writes[device_id][params["address"]] = params["data"]
                self._instruction_log[device_id].append("write")
            elif command == "data_barrier":
                # 数据屏障：屏障前的写入全部对CPU/后续读取可见
                self._visible_data[device_id].update(self._pending_writes[device_id])
                self._pending_writes[device_id].clear()
                self._instruction_log[device_id].append("data_barrier")
            elif command == "instruction_barrier":
                # 指令屏障：日志中落一个不可重排的边界
                self._instruction_log[device_id].append("instruction_barrier")
            else:
                self._instruction_log[device_id].append(command)
            self._history.append(("execute", buffer_id, command))
        buf.state = "executed"
        if fence_id is not None:
            self._fences[fence_id].signaled = True
        if semaphore_id is not None:
            self._semaphores[semaphore_id].value += 1

    # ------------------------------------------------------------------
    # 同步原语：栅栏 + 信号量
    # ------------------------------------------------------------------

    def create_fence(self, device_id: int, signaled: bool = False) -> int:
        self._require_device(device_id)
        fence_id = next(self._ids)
        self._fences[fence_id] = Fence(
            fence_id=fence_id, device_id=device_id, signaled=signaled,
        )
        return fence_id

    def fence_signaled(self, fence_id: int) -> bool:
        fence = self._fences.get(fence_id)
        if fence is None:
            raise ValueError(f"栅栏不存在: {fence_id}")
        return fence.signaled

    def signal_fence(self, fence_id: int) -> None:
        fence = self._fences.get(fence_id)
        if fence is None:
            raise ValueError(f"栅栏不存在: {fence_id}")
        fence.signaled = True

    def reset_fence(self, fence_id: int) -> None:
        fence = self._fences.get(fence_id)
        if fence is None:
            raise ValueError(f"栅栏不存在: {fence_id}")
        fence.signaled = False

    def wait_fence(self, fence_id: int, timeout: Optional[float] = None) -> bool:
        """等待栅栏触发；timeout为None时立即返回当前状态。"""
        fence = self._fences.get(fence_id)
        if fence is None:
            raise ValueError(f"栅栏不存在: {fence_id}")
        if fence.signaled or timeout is None:
            return fence.signaled
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if fence.signaled:
                return True
            time.sleep(0.001)
        return fence.signaled

    def create_semaphore(self, device_id: int, initial: int = 0) -> int:
        self._require_device(device_id)
        if initial < 0:
            raise ValueError("initial不能为负数")
        semaphore_id = next(self._ids)
        self._semaphores[semaphore_id] = Semaphore(
            semaphore_id=semaphore_id, device_id=device_id, value=initial,
        )
        return semaphore_id

    def signal_semaphore(self, semaphore_id: int) -> None:
        sem = self._semaphores.get(semaphore_id)
        if sem is None:
            raise ValueError(f"信号量不存在: {semaphore_id}")
        sem.value += 1

    def wait_semaphore(self, semaphore_id: int,
                       timeout: Optional[float] = None) -> bool:
        """获取一个信号量计数；没有计数且超时后返回False。"""
        sem = self._semaphores.get(semaphore_id)
        if sem is None:
            raise ValueError(f"信号量不存在: {semaphore_id}")
        if sem.value > 0:
            sem.value -= 1
            return True
        if timeout is None:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if sem.value > 0:
                sem.value -= 1
                return True
            time.sleep(0.001)
        return False

    # ------------------------------------------------------------------
    # 内存屏障 / 可见性查询
    # ------------------------------------------------------------------

    def read_visible(self, device_id: int, address: int) -> object:
        """CPU侧读取：只能读到数据屏障之后可见的数据。"""
        self._require_device(device_id)
        return self._visible_data[device_id].get(address)

    def instruction_log(self, device_id: int) -> List[str]:
        """设备上的实际指令执行序列（用于验证指令屏障的顺序保证）。"""
        self._require_device(device_id)
        return list(self._instruction_log[device_id])

    # ------------------------------------------------------------------
    # 帧更新 / 重置
    # ------------------------------------------------------------------

    def update(self, dt: float):
        """每帧调度：按优先级依次执行所有队列中的命令缓冲。"""
        while self._scheduled_queues():
            self.execute_next()

    def reset(self):
        self._state = {}
        self._history = []
        self._devices = {}
        self._allocations = {}
        self._buffers = {}
        self._queues = {}
        self._fences = {}
        self._semaphores = {}
        self._pending_writes = {}
        self._visible_data = {}
        self._instruction_log = {}
