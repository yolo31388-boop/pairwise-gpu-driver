"""
游戏内GPU驱动与命令缓冲系统 - 验收测试
运行方式：python -m pytest tests/test_gpu_driver.py -q
共 16 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu.driver import GPUDriver


class TestGPUDriver:
    def setup_method(self):
        self.system = GPUDriver()
        self.dev = self.system.create_device(name="gpu0", total_memory=1024)

    def _recorded_buffer(self, *commands):
        """创建一个录制完成的命令缓冲。commands为(command, params)元组。"""
        buf = self.system.create_command_buffer(self.dev)
        self.system.begin_recording(buf)
        for name, params in commands:
            self.system.record_command(buf, name, **params)
        self.system.end_recording(buf)
        return buf

    # --- 设备管理 ---

    def test_case_01(self):
        """创建设备后可以查询，设备出现在设备列表中"""
        dev = self.system.get_device(self.dev)
        assert dev is not None
        assert dev.name == "gpu0"
        assert dev.total_memory == 1024
        assert dev.used_memory == 0
        assert self.dev in self.system.list_devices()

    def test_case_02(self):
        """销毁设备后不能再查询或使用该设备"""
        assert self.system.destroy_device(self.dev) is True
        assert self.system.get_device(self.dev) is None
        assert self.dev not in self.system.list_devices()
        assert self.system.destroy_device(self.dev) is False
        with pytest.raises(ValueError):
            self.system.create_command_queue(self.dev)

    # --- 内存管理 ---

    def test_case_03(self):
        """分配显存后内存统计正确更新"""
        alloc = self.system.allocate_memory(self.dev, 256, usage="vertex")
        assert alloc is not None
        info = self.system.memory_info(self.dev)
        assert info["total"] == 1024
        assert info["used"] == 256
        assert info["free"] == 768

    def test_case_04(self):
        """显存不足时分配失败；释放后空间回收"""
        big = self.system.allocate_memory(self.dev, 1024)
        assert big is not None
        assert self.system.allocate_memory(self.dev, 1) is None
        assert self.system.free_memory(self.dev, big) is True
        assert self.system.memory_info(self.dev)["free"] == 1024
        assert self.system.free_memory(self.dev, big) is False

    # --- 命令缓冲：录制 ---

    def test_case_05(self):
        """录制状态下的命令按顺序进入缓冲"""
        buf = self.system.create_command_buffer(self.dev)
        self.system.begin_recording(buf)
        self.system.record_command(buf, "draw", vertices=3)
        self.system.record_command(buf, "draw", vertices=6)
        self.system.end_recording(buf)
        cb = self.system.get_command_buffer(buf)
        assert cb.state == "executable"
        assert [c for c, _ in cb.commands] == ["draw", "draw"]
        assert cb.commands[0][1]["vertices"] == 3

    def test_case_06(self):
        """非录制状态禁止录制；录制未完成禁止提交"""
        buf = self.system.create_command_buffer(self.dev)
        with pytest.raises(RuntimeError):
            self.system.record_command(buf, "draw")
        queue = self.system.create_command_queue(self.dev)
        with pytest.raises(RuntimeError):
            self.system.submit_command_buffer(queue, buf)
        self.system.begin_recording(buf)
        with pytest.raises(RuntimeError):
            self.system.submit_command_buffer(queue, buf)

    # --- 命令缓冲：提交 ---

    def test_case_07(self):
        """提交到队列后由驱动执行，状态流转为executed"""
        queue = self.system.create_command_queue(self.dev)
        buf = self._recorded_buffer(("draw", {"vertices": 3}))
        self.system.submit_command_buffer(queue, buf)
        assert self.system.get_command_buffer(buf).state == "submitted"
        self.system.update(0.016)
        assert self.system.get_command_buffer(buf).state == "executed"
        executed = [h for h in self.system._history if h[0] == "execute"]
        assert ("execute", buf, "draw") in executed

    def test_case_08(self):
        """跨设备提交被拒绝"""
        other = self.system.create_device(name="gpu1", total_memory=512)
        queue = self.system.create_command_queue(other)
        buf = self._recorded_buffer(("draw", {}))
        with pytest.raises(ValueError):
            self.system.submit_command_buffer(queue, buf)

    # --- 命令队列：优先级 ---

    def test_case_09(self):
        """高优先级队列的命令先执行"""
        low = self.system.create_command_queue(self.dev, priority=1)
        high = self.system.create_command_queue(self.dev, priority=10)
        buf_low = self._recorded_buffer(("draw", {"tag": "low"}))
        buf_high = self._recorded_buffer(("draw", {"tag": "high"}))
        self.system.submit_command_buffer(low, buf_low)
        self.system.submit_command_buffer(high, buf_high)
        self.system.update(0.016)
        executed = [h[1] for h in self.system._history if h[0] == "execute"]
        assert executed[0] == buf_high
        assert buf_low in executed

    def test_case_10(self):
        """同优先级按提交顺序，调度器每次只取一个缓冲"""
        q1 = self.system.create_command_queue(self.dev, priority=5)
        q2 = self.system.create_command_queue(self.dev, priority=5)
        b1 = self._recorded_buffer(("draw", {"tag": "b1"}))
        b2 = self._recorded_buffer(("draw", {"tag": "b2"}))
        self.system.submit_command_buffer(q1, b1)
        self.system.submit_command_buffer(q2, b2)
        assert self.system.execute_next() == b1
        assert self.system.execute_next() == b2
        assert self.system.execute_next() is None

    # --- 同步原语：栅栏 ---

    def test_case_11(self):
        """栅栏未触发时等待返回False，触发后返回True"""
        fence = self.system.create_fence(self.dev)
        assert self.system.fence_signaled(fence) is False
        assert self.system.wait_fence(fence) is False
        self.system.signal_fence(fence)
        assert self.system.fence_signaled(fence) is True
        assert self.system.wait_fence(fence) is True
        self.system.reset_fence(fence)
        assert self.system.fence_signaled(fence) is False

    def test_case_12(self):
        """命令缓冲执行完成时自动触发关联的栅栏"""
        queue = self.system.create_command_queue(self.dev)
        fence = self.system.create_fence(self.dev)
        buf = self._recorded_buffer(("draw", {}))
        self.system.submit_command_buffer(queue, buf, fence_id=fence)
        assert self.system.wait_fence(fence) is False
        self.system.update(0.016)
        assert self.system.wait_fence(fence) is True

    # --- 同步原语：信号量 ---

    def test_case_13(self):
        """信号量signal增加计数，wait消费计数"""
        sem = self.system.create_semaphore(self.dev, initial=1)
        assert self.system.wait_semaphore(sem) is True
        assert self.system.wait_semaphore(sem) is False
        self.system.signal_semaphore(sem)
        self.system.signal_semaphore(sem)
        assert self.system.wait_semaphore(sem) is True
        assert self.system.wait_semaphore(sem) is True
        assert self.system.wait_semaphore(sem) is False

    def test_case_14(self):
        """GPU执行完成时自动signal关联的信号量，实现GPU->CPU同步"""
        queue = self.system.create_command_queue(self.dev)
        sem = self.system.create_semaphore(self.dev)
        buf = self._recorded_buffer(("draw", {}))
        self.system.submit_command_buffer(queue, buf, semaphore_id=sem)
        assert self.system.wait_semaphore(sem) is False
        self.system.update(0.016)
        assert self.system.wait_semaphore(sem) is True

    # --- 内存屏障：数据屏障 ---

    def test_case_15(self):
        """数据屏障保证屏障前的写入对CPU可见"""
        queue = self.system.create_command_queue(self.dev)
        buf = self.system.create_command_buffer(self.dev)
        self.system.begin_recording(buf)
        self.system.record_write(buf, 0x100, "texture")
        self.system.record_data_barrier(buf)
        self.system.record_write(buf, 0x200, "mesh")
        self.system.end_recording(buf)
        self.system.submit_command_buffer(queue, buf)
        self.system.update(0.016)
        # 屏障前的写入可见
        assert self.system.read_visible(self.dev, 0x100) == "texture"
        # 屏障后的写入尚未可见（没有第二个屏障）
        assert self.system.read_visible(self.dev, 0x200) is None

    # --- 内存屏障：指令屏障 ---

    def test_case_16(self):
        """指令屏障保证屏障两侧指令按录制顺序执行、不被重排"""
        queue = self.system.create_command_queue(self.dev)
        buf = self.system.create_command_buffer(self.dev)
        self.system.begin_recording(buf)
        self.system.record_command(buf, "draw")
        self.system.record_instruction_barrier(buf)
        self.system.record_command(buf, "dispatch")
        self.system.end_recording(buf)
        self.system.submit_command_buffer(queue, buf)
        self.system.update(0.016)
        log = self.system.instruction_log(self.dev)
        assert log == ["draw", "instruction_barrier", "dispatch"]
        # 屏障前的指令一定先于屏障后的指令完成
        assert log.index("draw") < log.index("instruction_barrier")
        assert log.index("instruction_barrier") < log.index("dispatch")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
