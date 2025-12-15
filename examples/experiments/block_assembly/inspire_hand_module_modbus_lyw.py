from pymodbus.client.sync import ModbusSerialClient
from pymodbus.exceptions import ModbusIOException
import time
import numpy as np

class Hand:
    def __init__(self, lower_limit, upper_limit, port='/dev/ttyUSB0', node_id=1):
        """
        初始化ROH机械手

        Args:
            port: 串口号
            node_id: 设备ID，默认为1
        """
        self.client = ModbusSerialClient(
            method='rtu',
            port=port,
            baudrate=115200,
            bytesize=8,
            parity='N',
            stopbits=1,
        )
        self.node_id = node_id
        self.finger_names = {
            0: "小指",
            1: "无名指",
            2: "中指",
            3: "食指",
            4: "大拇指弯曲",
            5: "大拇指旋转"
        }

        # 连接设备
        if not self.client.connect():
            raise ConnectionError("无法连接到机械手")

        # 等待设备就绪
        time.sleep(1)

        # 初始化机械手
        # self._init_hand()

        self.old_lower_set = [1000, 1000, 1000, 1000, 1000, 1000]
        self.old_upper_set = [0, 0, 0, 0, 0, 0]
        self.old_lower_get = [1000, 1000, 1000, 1000, 1000, 1000]
        self.old_upper_get = [0, 0, 0, 0, 0, 0]
        self.new_upper = upper_limit
        self.new_lower = lower_limit

    def _read_register(self, address, count=1):
        """
        安全地读取寄存器

        Args:
            address: 寄存器地址
            count: 要读取的寄存器数量

        Returns:
            读取到的值列表，失败返回None
        """
        try:
            result = self.client.read_holding_registers(address, count, unit=self.node_id)
            if result.isError():
                print(f"读取寄存器{address}失败")
                return None
            return result.registers
        except ModbusIOException as e:
            print(f"ModBus通信错误: {str(e)}")
            return None
        except Exception as e:
            print(f"读取寄存器时发生错误: {str(e)}")
            return None

    def _write_register(self, address, value):
        """
        安全地写入寄存器

        Args:
            address: 寄存器地址
            value: 要写入的值

        Returns:
            是否成功
        """
        try:
            result = self.client.write_register(address, value, unit=self.node_id)
            if result.isError():
                print(f"写入寄存器{address}失败")
                return False
            return True
        except ModbusIOException as e:
            print(f"ModBus通信错误: {str(e)}")
            return False
        except Exception as e:
            print(f"写入寄存器时发生错误: {str(e)}")
            return False

    def _batch_read_registers(self, start_address, count, max_retries=3):
        """
        带重试机制的批量读取寄存器
        
        Args:
            start_address: 起始寄存器地址
            count: 要读取的寄存器数量
            max_retries: 最大重试次数
        """
        for attempt in range(max_retries):
            try:
                result = self.client.read_holding_registers(start_address, count, unit=self.node_id)
                if not result.isError():
                    return result.registers
                
                # 如果失败，打印详细信息并重试
                print(f"批量读取尝试 {attempt + 1}/{max_retries} 失败")
                print(f"起始地址: {start_address}, 数量: {count}")
                print(f"错误信息: {result}")
                
                if attempt < max_retries - 1:
                    time.sleep(0.01 * (attempt + 1))  # 递增延迟
                    continue
                    
            except Exception as e:
                print(f"ModBus通信错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(0.01 * (attempt + 1))
                    continue
        
        # 如果所有重试都失败，尝试单个读取
        print("批量读取失败，切换到单个读取模式")
        try:
            registers = []
            for addr in range(start_address, start_address + count):
                result = self._read_register(addr)
                if result is None:
                    return None
                registers.extend(result)
            return registers
        except Exception as e:
            print(f"单个读取也失败: {str(e)}")
            return None

    def _batch_write_registers(self, start_address, values):
        """
        批量写入连续的寄存器

        Args:
            start_address: 起始寄存器地址
            values: 要写入的值列表

        Returns:
            是否成功
        """
        try:
            result = self.client.write_registers(start_address, values, unit=self.node_id)
            if result.isError():
                print(f"批量写入寄存器失败，起始地址: {start_address}")
                return False
            return True
        except ModbusIOException as e:
            print(f"ModBus通信错误: {str(e)}")
            return False
        except Exception as e:
            print(f"写入寄存器时发生错误: {str(e)}")
            return False

    def _init_hand(self):
        """初始化机械手"""
        try:
            # 设置自检级别为1(允许开机归零)
            if not self._write_register(1008, 1):
                raise RuntimeError("设置自检级别失败")

            # 开始初始化
            if not self._write_register(1013, 1):
                raise RuntimeError("开始初始化失败")

            self._write_register(1013, 1)

            # 等待初始化完成
            # retry_count = 0
            # while retry_count < 30:  # 最多等待30秒
            #     error = self._check_error()
            #     if error is None:
            #         time.sleep(1)
            #     elif error == 0:  # 初始化完成
            #         print("初始化finish")
            #         break
            #     elif error != 1:  # 不是等待初始化状态
            #         raise RuntimeError(f"初始化过程出错，错误代码: {error}")
            #     time.sleep(1)
            #     retry_count += 1

            # if retry_count >= 30:
            #     raise RuntimeError("初始化超时")

        except Exception as e:
            raise RuntimeError(f"初始化失败: {str(e)}")

    def _check_error(self):
        """
        检查错误状态

        Returns:
            错误代码，None表示读取失败
        """
        result = self._read_register(1006)
        if result is None:
            return None

        error = result[0]
        if error:
            error_dict = {
                1: "等待初始化",
                2: "等待校正",
                3: "无效数据",
                4: "电机堵转",
                5: "操作失败",
                6: "保存失败"
            }
            print(f"错误: {error_dict.get(error, '未知错误')}")
        
        return error

    def set_hand_angle(self, angles):
        """
        同时设置多个手指的角度

        Args:
            angles: 长度为6的列表，包含所有手指的目标角度 (0-100的百分比)
                [拇指, 食指, 中指, 无名指, 小指, 拇指旋转]
        """
        if len(angles) != 6:
            raise ValueError("angles必须是长度为6的列表")
        
        try:
            # 计算所有目标位置
            # 使用numpy进行批量计算
            # print(angles)
            targets = self.unscale_angles(angles)
            # print(targets)
            targets = np.clip(targets, 0, 1000)

            # 批量写入所有目标位置
            if not self._batch_write_registers(1486, targets):
                raise RuntimeError("批量写入目标位置失败")
            
            # print(f"设置所有手指角度为: {angles}")
            
        except Exception as e:
            raise RuntimeError(f"设置角度失败: {str(e)}")

    def set_hand_speed(self, speed):
        """
        同时设置多个手指的速度
        speed的值需要是0-1000的值

        Args:
            angles: 长度为6的列表，包含所有手指的目标速度 (0-100的百分比)
                [拇指, 食指, 中指, 无名指, 小指, 拇指旋转]
        """
        if len(speed) != 6:
            raise ValueError("speed必须是长度为6的列表")
        
        try:
            # 计算所有目标位置
            # 使用numpy进行批量计算
            # print(angles)
            # print(targets)

            # 批量写入所有目标位置
            if not self._batch_write_registers(1522, speed):
                raise RuntimeError("批量写入目标速度失败")
            
            # print(f"设置所有手指角度为: {angles}")
            
        except Exception as e:
            raise RuntimeError(f"设置速度失败: {str(e)}")

    def get_hand_angle(self):
        """
        同时读取所有手指的当前角度，带错误处理
        """
        try:
            # 尝试批量读取
            current_positions = self._batch_read_registers(1546, 6)
            # print("current_position", current_positions)
            if current_positions is None:
                # 如果批量读取失败，尝试逐个读取
                current_positions = []
                for i in range(6):
                    result = self._read_register(1546 + i)
                    if result is None:
                        raise RuntimeError(f"读取手指 {i} 位置失败")
                    current_positions.extend(result)
            
            # 计算角度
            current_positions = np.array(current_positions)
            angles = self.scale_angles(current_positions)
            
            return angles.tolist()
            
        except Exception as e:
            print(f"读取角度失败: {str(e)}")
            # 返回上一次的有效值，或者默认值
            if hasattr(self, '_last_valid_angles'):
                print("使用上一次的有效值")
                return self._last_valid_angles
            return [0.5] * 6  # 默认中间位置

    def get_hand_force(self):
        """
        同时读取所有手指的当前受力，带错误处理
        """
        try:
            # 尝试批量读取
            current_force = self._batch_read_registers(1582, 6)
            if current_force is None:
                # 如果批量读取失败，尝试逐个读取
                current_force = []
                for i in range(6):
                    result = self._read_register(1582 + i)
                    if result is None:
                        raise RuntimeError(f"读取手指 {i} 受力失败")
                    current_force.extend(result)
            
            # 计算角度
            current_force = np.array(current_force)/1000*9.8
            return current_force.tolist()
            
        except Exception as e:
            print(f"读取角度失败: {str(e)}")
            # 返回上一次的有效值，或者默认值
            if hasattr(self, '_last_valid_angles'):
                print("使用上一次的有效值")
                return self._last_valid_angles
            return [0.5] * 6  # 默认中间位置

    def close(self):
        """关闭连接"""
        if hasattr(self, 'client'):
            self.client.close()

    def __del__(self):
        """析构函数，确保关闭连接"""
        self.close()

    def scale_angles(self, angles):
        """
        Scale a list of angles from one range to another.

        Parameters:
        angles (list or np.ndarray): Input list of angles to be scaled.
        old_upper (np.ndarray): Old upper bounds for each angle.
        old_lower (np.ndarray): Old lower bounds for each angle.
        new_upper (np.ndarray): New upper bounds for each angle.
        new_lower (np.ndarray): New lower bounds for each angle.

        Returns:
        np.ndarray: Scaled angles in the new range.
        """
        angles = np.array(angles, dtype=np.float32)
        old_upper = np.array(self.old_upper_get, dtype=np.float32)
        old_lower = np.array(self.old_lower_get, dtype=np.float32)
        new_upper = np.array(self.new_upper, dtype=np.float32)
        new_lower = np.array(self.new_lower, dtype=np.float32)

        # Scale the angles to the [0, 1] range first
        scaled_to_unit_range = (angles - old_lower) / (old_upper - old_lower)
        
        # Map from [0, 1] to the new range [new_lower, new_upper]
        scaled_angles = scaled_to_unit_range * (new_upper - new_lower) + new_lower
        
        return scaled_angles

    def unscale_angles(self, scaled_angles):
        """
        Unscale a list of angles from the new range back to the original range.

        Parameters:
        scaled_angles (list or np.ndarray): Input list of scaled angles to be unscaled.
        old_upper (np.ndarray): Original upper bounds for each angle.
        old_lower (np.ndarray): Original lower bounds for each angle.
        new_upper (np.ndarray): New upper bounds for each angle.
        new_lower (np.ndarray): New lower bounds for each angle.

        Returns:
        np.ndarray: Angles mapped back to the original range.
        """
        scaled_angles = np.array(scaled_angles, dtype=np.float32)
        old_upper = np.array(self.old_upper_set, dtype=np.float32)
        old_lower = np.array(self.old_lower_set, dtype=np.float32)
        new_upper = np.array(self.new_upper, dtype=np.float32)
        new_lower = np.array(self.new_lower, dtype=np.float32)

        # Convert the scaled angles from the new range [new_lower, new_upper] to [0, 1]
        unit_range_scaled = (scaled_angles - new_lower) / (new_upper - new_lower)
        
        # Map from [0, 1] back to the original range [old_lower, old_upper]
        original_angles = unit_range_scaled * (old_upper - old_lower) + old_lower
        
        return original_angles.astype(int)

if __name__ == "__main__":
    try:
        hand = Hand(lower_limit=[0., 0., 0., 0., 0., 0.2], upper_limit=[1.7000, 1.7000, 1.7000, 1.7000, 0.5000, 1.3000], port='/dev/ttyUSB0')

        # 测试同时控制所有手指
        print("测试所有手指...")

        # # 同时设置所有手指到50%位置
        hand.set_hand_angle([1.0, 1.7000, 1.7000, 1.7000, 0.0, 1])
        time.sleep(1)

        # 读取所有手指当前角度
        current_angles = hand.get_hand_angle()
        print(current_angles)

        # steps = 10
        # min_value = 0.45
        # max_value = 1.0

        # while True:  # 无限循环
        #     # 使用正弦函数生成往复运动
        #     for i in range(steps):
        #         # 使用正弦函数生成0到1之间的值，然后映射到0.5-1区间
        #         t = i / steps * 2 * np.pi  # 0 到 2π
        #         current_value = min_value + (max_value - min_value) * (np.sin(t) + 1) / 2
                
        #         # 前5个数设置为当前值，最后一个保持为0
        #         angles = [current_value] * 5 + [1.0]
                
        #         # 设置角度
        #         hand.set_angles(angles)
        #         time.sleep(0.01)
                
        #         # 读取当前角度
        #         current_angles = hand.get_angles()
        #         print(f"Current value: {current_value:.3f}")

    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        hand.close()