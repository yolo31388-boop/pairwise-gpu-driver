"""
游戏内GPU驱动与命令缓冲系统 - 验收测试
运行方式：python -m pytest tests/test_gpu_driver.py -q
共 16 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu.driver import (
    GPUDriver,
    CommandBuffer,
    CommandQueue,
    Fence,
    Semaphore,
    MemoryBarrier,
)


class TestGPUDriver:
    def setup_method(self):
        self.system = GPUDriver()

    # -- GPU驱动：设备管理 -------------------------------------------------

    def test_case_01(self):
        """驱动可以注册并管理GPU设备"""
        device = self.system.add_device(name="RTX", memory_total=8 << 20)
        assert device.device_id == 0
        assert device.name == "RTX"
        assert self.system.device_count == 1
        assert self.system.get_device(0) is device

    def test_case_02(self):
        """驱动可以移除GPU设备，移除后查询报错"""
        first = self.system.add_device(name="GPU0")
        second = self.system.add_device(name="GPU1")
        assert first.device_id != second.device_id
        self.system.remove_device(first.device_id)
        assert self.system.device_count == 1
        with pytest.raises(KeyError):
            self.system.get_device(first.device_id)

    # -- GPU驱动：内存管理 -------------------------------------------------

    def test_case_03(self):
        """驱动可以分配显存并统计用量"""
        handle = self.system.allocate_memory(1024)
        assert self.system.memory_used == 1024
        assert self.system.memory_free == self.system.memory.capacity - 1024
        self.system.memory.write(handle, b"abc")
        assert self.system.memory.read(handle, 3) == b"abc"

    def test_case_04(self):
        """驱动可以释放显存，显存不足时报错"""
        handle = self.system.allocate_memory(512)
        self.system.free_memory(handle)
        assert self.system.memory_used == 0
        with pytest.raises(KeyError):
            self.system.free_memory(handle)
        with pytest.raises(MemoryError):
            self.system.allocate_memory(self.system.memory.capacity + 1)

    # -- 命令缓冲：录制 + 提交 ----------------------------------------------

    def test_case_05(self):
        """命令缓冲可以录制多条GPU命令"""
        buffer = self.system.command_buffer()
        buffer.begin_recording()
        buffer.record("draw", vertices=3)
        buffer.record("bind_texture", slot=0)
        buffer.end_recording()
        assert len(buffer) == 2
        assert buffer.commands[0][0] == "draw"
        assert buffer.state == CommandBuffer.RECORDED

    def test_case_06(self):
        """未录制的缓冲不能提交，录制外的record报错"""
        buffer = self.system.command_buffer()
        with pytest.raises(RuntimeError):
            buffer.record("draw")
        with pytest.raises(RuntimeError):
            buffer.submit(self.system.command_queue())
        buffer.begin_recording()
        buffer.record("draw")
        buffer.end_recording()
        buffer.submit(self.system.command_queue())
        assert buffer.state == CommandBuffer.SUBMITTED
        assert self.system.command_queue().pending_count == 1

    # -- 命令队列：优先级 + 调度 ----------------------------------------------

    def test_case_07(self):
        """命令队列按优先级调度，数值小者先执行"""
        queue = CommandQueue()
        order = []
        for name, priority in [("low", 10), ("high", 1), ("mid", 5)]:
            buf = CommandBuffer(name=name)
            buf.begin_recording()
            buf.record("draw")
            buf.end_recording()
            buf.submit(queue, priority=priority)
        while queue.pending_count:
            order.append(queue.schedule().name)
        assert order == ["high", "mid", "low"]

    def test_case_08(self):
        """命令队列调度执行缓冲中的命令"""
        executed = []
        queue = self.system.command_queue()
        buffer = self.system.command_buffer()
        buffer.begin_recording()
        buffer.record("draw", vertices=3)
        buffer.end_recording()
        self.system.submit(buffer, priority=0)
        done = queue.execute_all(lambda cmd, params: executed.append(cmd))
        assert executed == ["draw"]
        assert done[0].state == CommandBuffer.EXECUTED
        assert queue.pending_count == 0

    # -- 同步原语：栅栏 + 信号量 ----------------------------------------------

    def test_case_09(self):
        """栅栏：signal后wait成功，reset后wait超时"""
        fence = Fence()
        assert not fence.signaled
        assert not fence.wait(timeout=0.01)
        fence.signal()
        assert fence.signaled
        assert fence.wait(timeout=0.01)
        fence.reset()
        assert not fence.wait(timeout=0.01)

    def test_case_10(self):
        """信号量：signal增加计数，wait消费计数"""
        semaphore = Semaphore()
        assert not semaphore.wait(timeout=0.01)
        semaphore.signal(2)
        assert semaphore.value == 2
        assert semaphore.wait(timeout=0.01)
        assert semaphore.wait(timeout=0.01)
        assert not semaphore.wait(timeout=0.01)

    def test_case_11(self):
        """驱动dispatch后置位栅栏与信号量，CPU可同步"""
        sync = self.system.synchronization_primitives()
        buffer = self.system.command_buffer()
        buffer.begin_recording()
        buffer.record("draw")
        buffer.end_recording()
        self.system.submit(buffer)
        self.system.dispatch()
        assert sync.fence.wait(timeout=0.1)
        assert sync.semaphore.wait(timeout=0.1)

    # -- 内存屏障：数据屏障 + 指令屏障 -----------------------------------------

    def test_case_12(self):
        """数据屏障保证数据可见性"""
        barrier = self.system.memory_barrier()
        handle = self.system.allocate_memory(4)
        self.system.memory.write(handle, b"data")
        barrier.data_barrier()
        assert MemoryBarrier.DATA in barrier.barriers
        assert self.system.memory.read(handle) == b"data"

    def test_case_13(self):
        """指令屏障保证指令顺序"""
        barrier = self.system.memory_barrier()
        barrier.instruction_barrier()
        barrier.full_barrier()
        assert barrier.barriers == [MemoryBarrier.INSTRUCTION, MemoryBarrier.FULL]

    # -- 驱动整体行为 ---------------------------------------------------------

    def test_case_14(self):
        """update推进帧并调度执行已提交的命令缓冲"""
        buffer = self.system.command_buffer()
        buffer.begin_recording()
        buffer.record("draw")
        buffer.end_recording()
        self.system.submit(buffer)
        self.system.update(1 / 60)
        assert self.system.frame == 1
        assert buffer.state == CommandBuffer.EXECUTED
        assert self.system.command_queue().pending_count == 0

    def test_case_15(self):
        """reset清空设备、显存、队列与历史"""
        self.system.add_device(name="GPU0")
        handle = self.system.allocate_memory(256)
        buffer = self.system.command_buffer()
        buffer.begin_recording()
        buffer.record("draw")
        buffer.end_recording()
        self.system.submit(buffer)
        self.system.reset()
        assert self.system.device_count == 0
        assert self.system.memory_used == 0
        assert self.system.command_queue().pending_count == 0
        assert self.system.frame == 0
        assert self.system.history == []
        with pytest.raises(KeyError):
            self.system.free_memory(handle)

    def test_case_16(self):
        """完整管线：设备+显存+录制+按优先级提交+调度+同步"""
        self.system.add_device(name="RTX", memory_total=1 << 20)
        handle = self.system.allocate_memory(128)
        self.system.memory.write(handle, b"vertices")
        for name, priority in [("shadow", 5), ("main", 1)]:
            buf = self.system.command_buffer(name=name)
            buf.begin_recording()
            buf.record("draw", pass_name=name)
            buf.end_recording()
            self.system.submit(buf, priority=priority)
        done = self.system.dispatch()
        assert [b.name for b in done] == ["main", "shadow"]
        assert self.system.synchronization_primitives().fence.signaled
        self.system.memory_barrier().data_barrier()
        assert self.system.memory.read(handle, 8) == b"vertices"
        self.system.free_memory(handle)
        assert self.system.memory_used == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
