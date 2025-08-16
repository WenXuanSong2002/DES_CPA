import numpy as np
import matplotlib.pyplot as plt
from typing import List
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# DES常量表
IP_Table = np.array([
57,49,41,33,25,17,9,1,
59,51,43,35,27,19,11,3,
61,53,45,37,29,21,13,5,
63,55,47,39,31,23,15,7,
56,48,40,32,24,16,8,0,
58,50,42,34,26,18,10,2,
60,52,44,36,28,20,12,4,
62,54,46,38,30,22,14,6
], dtype=np.uint8)  

IP_Inv_Table = np.array([
39,7,47,15,55,23,63,31,
38,6,46,14,54,22,62,30,
37,5,45,13,53,21,61,29,
36,4,44,12,52,20,60,28,
35,3,43,11,51,19,59,27,
34,2,42,10,50,18,58,26,
33,1,41,9,49,17,57,25,
32,0,40,8,48,16,56,24
], dtype=np.uint8)

E_Table = np.array([
31, 0, 1, 2, 3, 4,
3, 4, 5, 6, 7, 8,
7, 8, 9, 10, 11, 12,
11, 12, 13, 14, 15, 16,
15, 16, 17, 18, 19, 20,
19, 20, 21, 22, 23, 24,
23, 24, 25, 26, 27, 28,
27, 28, 29, 30, 31, 0
], dtype=np.uint8)

P_Table = np.array([
15,6,19,20,
28,11,27,16,
0,14,22,25,
4,17,30,9,
1,7,23,13,
31,26,2,8,
18,12,29,5,
21,10,3,24    
], dtype=np.uint8)  

Sbox = np.array([
    #S1
    [
        [14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
        [0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],
        [4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
        [15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13]
    ],
    #S2
    [
        [15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10],
        [3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5],
        [0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15],
        [13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9]
    ],
    #S3
    [
        [10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8],
        [13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1],
        [13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7],
        [1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12]
    ],
    #S4
    [
        [7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15],
        [13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9],
        [10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4],
        [3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14]
    ],
    #S5
    [
        [2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],
        [14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6],
        [4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14],
        [11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3]
    ],
    #S6
    [
        [12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11],
        [10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8],
        [9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6],
        [4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13]
    ],
    #S7
    [
        [4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1],
        [13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6],
        [1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2],
        [6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12]
    ],
    #S8
    [
        [13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7],
        [1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2],
        [7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8],
        [2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11]
    ],
], dtype=np.uint8)

PC_1 = np.array([
56,48,40,32,24,16,8,
0,57,49,41,33,25,17,
9,1,58,50,42,34,26,
18,10,2,59,51,43,35,
62,54,46,38,30,22,14,
6,61,53,45,37,29,21,
13,5,60,52,44,36,28,
20,12,4,27,19,11,3
], dtype=np.uint8)  

PC_2 = np.array([
13,16,10,23,0,4,2,27,
14,5,20,9,22,18,11,3,
25,7,15,6,26,19,12,1,
40,51,30,36,46,54,29,39,
50,44,32,47,43,48,38,55,
33,52,45,41,49,35,28,31
], dtype=np.uint8)  

Move_Times = np.array([1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1], dtype=np.uint8)

def to_bit(data: int, bits: int) -> np.ndarray:
    """将整数转换为二进制位的 NumPy 数组"""
    data_bits = np.zeros(bits, dtype=np.uint8)
    
    for i in range(bits):
        data_bits[i] = (data >> (bits - 1 - i)) & 1
    return data_bits 

def to_data(data_bits: np.ndarray, bits: int) -> int:
    """将二进制位的 NumPy 数组转换为整数"""
    data = 0
    for i in range(bits):
        data |= int(data_bits[i]) << (bits - 1 - i)
    return data

def split_text(text_bit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将 64 位拆分为左右两个 32 位部分"""
    left = text_bit[:32]
    right = text_bit[32:]
    return left, right

def e_extension(text_bit: np.ndarray) -> np.ndarray:
    """执行 E 扩展操作"""
    result_bit = np.zeros(48, dtype=np.uint8)
    for i in range(48):
        result_bit[i] = text_bit[E_Table[i]]
    return result_bit

def s_substitution(temp_bit: np.ndarray) -> np.ndarray:
    """执行 S 盒置换操作"""
    result = np.zeros(32, dtype=np.uint8)
    for i in range(8):
        # 计算行和列
        row = (temp_bit[i * 6] << 1) | temp_bit[i * 6 + 5]
        col = (temp_bit[i * 6 + 1] << 3) | (temp_bit[i * 6 + 2] << 2) | (temp_bit[i * 6 + 3] << 1) | temp_bit[i * 6 + 4]
        
        # 查找 S 盒并转换为 4 位二进制
        value = Sbox[i, row, col]
        for j in range(4):
            result[i * 4 + j] = (value >> (3 - j)) & 1
    return result

def hamming_weight(data: int) -> int:
    """计算汉明重量（二进制表示中1的个数）"""
    return bin(data).count('1')

def key_permutation(key: int) -> np.ndarray:
    """生成 16 轮子密钥"""
    K = np.zeros(16, dtype=np.uint64)
    
    # 初始密钥转换为二进制
    key_bit = to_bit(key, 64)
    # PC-1 置换
    Key_56_bit = np.zeros(56, dtype=np.uint8)
    for i in range(56):
        Key_56_bit[i] = key_bit[PC_1[i]]

    # 拆分为 C0 和 D0
    C_bit = Key_56_bit[:28].copy()
    D_bit = Key_56_bit[28:].copy()
    
    # 16 轮密钥生成
    for i in range(16):
        # 左移操作
        shift = int(Move_Times[i])
        # 循环左移
        C_bit = np.roll(C_bit, -shift)
        D_bit = np.roll(D_bit, -shift)
        # 合并 C 和 D
        CD_bit = np.concatenate([C_bit, D_bit])
        # PC-2 置换
        K_bit = np.zeros(48, dtype=np.uint8)
        for j in range(48):
            K_bit[j] = CD_bit[PC_2[j]]
        
        # 转换为整数并确保结果是48位
        K[i] = (to_data(K_bit, 48) & ((1 << 48) - 1)) & 0xFFFF_FFFF_FFFF
    return K

def f_function(text: int, key_48: int) -> int:
    """ F 函数"""
    # 32 位文本转换为二进制
    text_bit = to_bit(text, 32)
    
    # E 扩展
    temp_1_bit = e_extension(text_bit)
    temp_1 = to_data(temp_1_bit, 48) & 0xFFFF_FFFF_FFFF
    # 与子密钥异或
    temp_yh = temp_1 ^ key_48
    temp_yh_bit = to_bit(temp_yh, 48)
    # S 盒置换
    temp_2_bit = s_substitution(temp_yh_bit)
    temp_2 = to_data(temp_2_bit, 32) & 0xFFFF_FFFF
    # P 置换
    result_bit = p_permutation(temp_2_bit)
    # 转换回整数
    result = to_data(result_bit, 32) & 0xFFFF_FFFF

    return result

def p_permutation(temp_bit:np.ndarray) -> np.ndarray:
    """执行 p 置换操作"""
    result_bit = np.zeros(32, dtype=np.uint8)
    for i in range(32):
        result_bit[i] = temp_bit[P_Table[i]]
    return result_bit

def des_encrypt(plaintext: int, key: int) -> int:
    """执行 DES 加密"""
    # 生成 16 轮子密钥
    sub_keys = key_permutation(key)
    
    # 初始置换
    text_bit = to_bit(plaintext, 64)
    result_bit = np.zeros(64, dtype=np.uint8)
    for i in range(64):
        result_bit[i] = text_bit[IP_Table[i]]
    
    # 拆分为左右两部分
    left_bit, right_bit = split_text(result_bit)
    L = to_data(left_bit, 32)
    R = to_data(right_bit, 32)
    
    # 16 轮 Feistel 网络
    for i in range(16):
        temp = L
        L = R & 0xFFFFFFFF  # 确保32位
        R = (temp ^ f_function(R, sub_keys[i])) & 0xFFFFFFFF  # 确保32位
    
    # 交换左右部分并合并
    merged = ((R & 0xFFFF_FFFF) << 32) | (L & 0xFFFF_FFFF)
    
    # 最终置换
    merged_bit = to_bit(merged, 64)
    result_bit = np.zeros(64, dtype=np.uint8)
    for i in range(64):
        result_bit[i] = merged_bit[IP_Inv_Table[i]]
    result = to_data(result_bit, 64) & 0xFFFF_FFFF_FFFF_FFFF
    
    return result

def get_power_consumption(data: int) -> float:
    """获取功耗值，直接使用汉明重量"""
    return float(hamming_weight(data))

def get_sbox_output(plaintext: int, key: int, round_num: int) -> List[int]:
    """获取第round_num轮8个S盒的输出值"""
    # 生成子密钥
    sub_keys = key_permutation(key)
    
    # 初始置换
    text_bit = to_bit(plaintext, 64)
    result_bit = np.zeros(64, dtype=np.uint8)
    for i in range(64):
        result_bit[i] = text_bit[IP_Table[i]]
    
    # 拆分为左右两部分
    left_bit, right_bit = split_text(result_bit)
    L = to_data(left_bit, 32)
    R = to_data(right_bit, 32)
    
    # 运行到指定轮次
    for i in range(round_num):
        temp = L
        L = R & 0xFFFFFFFF
        R = (temp ^ f_function(R, sub_keys[i])) & 0xFFFFFFFF
    
    # 获取该轮S盒输入和输出
    right_bit = to_bit(R, 32)
    extended_bit = e_extension(right_bit)
    extended_data = to_data(extended_bit, 48) & 0xFFFF_FFFF_FFFF
    subkey_data = sub_keys[round_num-1]
    
    # 异或得到S盒实际输入
    sbox_input_data = extended_data ^ subkey_data
    sbox_input_bits = to_bit(sbox_input_data, 48)
    
    # S盒替换得到输出
    sbox_output_bits = s_substitution(sbox_input_bits)
    sbox_outputs = []
    for i in range(8):
        sbox_bits = sbox_output_bits[i*4:(i+1)*4]
        sbox_value = to_data(sbox_bits, 4)
        sbox_outputs.append(sbox_value)
    
    return sbox_outputs

class DESCPAAttack:
    def __init__(self, num_traces: int = 5000):
        self.num_traces = num_traces
        self.plaintexts = []
        self.power_traces = []
        self.sbox_outputs = []
        self.sbox_inputs = []  # 添加缺少的属性
        
    def generate_traces(self, target_key: int):
        """生成功耗轨迹用于CPA攻击"""
        print(f"Generating {self.num_traces} power traces...")
        np.random.seed(42)  # 固定随机种子以确保结果可重现
        
        for i in range(self.num_traces):
            # 生成随机明文
            plaintext = np.random.randint(0, 2**64, dtype=np.uint64)
            self.plaintexts.append(plaintext)
            
            # 模拟加密过程并记录功耗
            # 这里我们关注最后一轮的S盒操作
            sbox_outputs = get_sbox_output(plaintext, target_key, 16)
            self.sbox_outputs.append(sbox_outputs)
            
            # 记录S盒输入（用于验证）
            # 我们需要计算实际的S盒输入来用于验证
            sub_keys = key_permutation(target_key)
            right_bits = to_bit(plaintext, 64)
            right_bits = right_bits[32:]  # 取右半部分
            # 应用15轮加密得到第16轮的右半部分
            # 简化处理：直接计算最后一轮的S盒输入
            extended_bit = e_extension(right_bits)
            extended_data = to_data(extended_bit, 48) & 0xFFFF_FFFF_FFFF
            subkey_data = sub_keys[15]  # 最后一轮子密钥
            
            # 异或得到S盒实际输入
            sbox_input_data = extended_data ^ subkey_data
            sbox_input_bits = to_bit(sbox_input_data, 48)
            
            # 提取8个S盒的输入值
            sbox_inputs = []
            for j in range(8):
                sbox_bits = sbox_input_bits[j*6:(j+1)*6]
                sbox_value = to_data(sbox_bits, 6)
                sbox_inputs.append(sbox_value)
            self.sbox_inputs.append(sbox_inputs)
            
            # 生成功耗轨迹（直接使用汉明重量作为功耗值）
            trace = []
            for t in range(100):  # 100个时间点
                # 功耗变化，主要在S盒操作时功耗较高
                if 40 <= t <= 60:  # 假设S盒操作在40-60时间点之间
                    power = get_power_consumption(sbox_outputs[t % 8])  # 使用S盒输出的汉明重量作为功耗
                else:
                    power = 2.0  # 基础功耗
                trace.append(power)
            self.power_traces.append(trace)
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i+1}/{self.num_traces} traces")
    
    def perform_cpa(self, target_sbox: int = 0) -> dict:
        """执行CPA攻击以恢复指定S盒对应的子密钥位"""
        print(f"Performing CPA attack on S-box {target_sbox+1}...")
        
        # 存储所有轨迹
        power_matrix = np.array(self.power_traces)  # shape: (num_traces, 100)
        plaintexts = np.array(self.plaintexts)      # shape: (num_traces,)
        
        # 存储所有猜测密钥下的中间值汉明重量
        correlations = {}
        key_candidates = range(64)  # 6位S盒输入，所以有64种可能
        
        for key_guess in key_candidates:
            # 计算所有轨迹中该S盒输入对应中间值的汉明重量
            hypothetical_hw = []
            for i in range(self.num_traces):
                # 获取明文
                plaintext = plaintexts[i]
                
                # 计算扩展右半部分
                right_bits = to_bit(plaintext, 64)
                right_bits = right_bits[32:]  # 取右半部分
                extended_bit = e_extension(right_bits)
                extended_data = to_data(extended_bit, 48) & 0xFFFF_FFFF_FFFF
                
                # 提取当前S盒的6位扩展数据
                sbox_extended_bits = extended_bit[target_sbox*6:(target_sbox+1)*6]
                sbox_extended_data = to_data(sbox_extended_bits, 6)
                
                # 计算假设的中间值 = 扩展数据 ^ 子密钥猜测
                intermediate_value = sbox_extended_data ^ key_guess
                
                # 计算汉明重量作为功耗模型
                hw = hamming_weight(intermediate_value)
                hypothetical_hw.append(hw)
            
            # 计算这个猜测密钥与所有时间点功耗的相关性
            hw_array = np.array(hypothetical_hw)
            correlations[key_guess] = []
            
            # 计算与每个时间点功耗的相关系数
            for t in range(power_matrix.shape[1]):
                power_at_t = power_matrix[:, t]
                # 避免除零错误
                if np.std(hw_array) == 0 or np.std(power_at_t) == 0:
                    correlation = 0
                else:
                    correlation = np.corrcoef(hw_array, power_at_t)[0, 1]
                correlations[key_guess].append(correlation if not np.isnan(correlation) else 0)
        
        return correlations
    
    def recover_key(self) -> List[int]:
        """恢复所有8个S盒对应的子密钥"""
        recovered_subkeys = []
        
        for sbox_index in range(8):
            print(f"\nRecovering subkey for S-box {sbox_index+1}...")
            correlations = self.perform_cpa(sbox_index)
            
            # 找到相关系数最大的密钥猜测
            max_correlation = -1
            best_key = -1
            
            for key_guess, corr_values in correlations.items():
                max_corr_for_key = np.max(np.abs(corr_values))
                if max_corr_for_key > max_correlation:
                    max_correlation = max_corr_for_key
                    best_key = key_guess
            
            recovered_subkeys.append(best_key)
            print(f"S-box {sbox_index+1} subkey: {best_key:06b} (decimal: {best_key})")
            print(f"Max correlation: {max_correlation:.4f}")
        
        return recovered_subkeys

def demonstrate_cpa_attack():
    """演示CPA攻击过程"""
    print("DES CPA Attack Simulation")
    print("=" * 50)
    
    # 设置目标密钥（我们希望攻击者通过CPA恢复这个密钥的部分信息）
    target_key = 0x133457799BBCDFF1
    print(f"Target key: 0x{target_key:016X}")
    
    # 创建CPA攻击实例
    cpa_attack = DESCPAAttack(num_traces=2000)
    
    # 生成功耗轨迹
    cpa_attack.generate_traces(target_key)
    
    # 执行攻击
    recovered_subkeys = cpa_attack.recover_key()
    
    print("\n" + "=" * 50)
    print("Attack Results Summary:")
    print("=" * 50)
    print("Recovered S-box input subkeys:")
    for i, subkey in enumerate(recovered_subkeys):
        print(f"  S{i+1}: {subkey:06b} (decimal: {subkey})")
    
    # 为了验证，我们计算真实子密钥
    print("\nReal S-box input subkeys (last round):")
    real_subkeys = cpa_attack.sbox_inputs[0]  # 取第一条轨迹的真实值
    for i, subkey in enumerate(real_subkeys):
        print(f"  S{i+1}: {subkey:06b} (decimal: {subkey})")
    
    # 绘制其中一个S盒的攻击结果
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制几个候选密钥的相关系数曲线
    correlations = cpa_attack.perform_cpa(0)  # 第一个S盒
    real_subkey = real_subkeys[0]
    recovered_subkey = recovered_subkeys[0]
    
    # 初始化 sample_key
    sample_key = list(correlations.keys())[0]
    
    # 绘制真实子密钥的相关系数曲线（如果存在）
    if real_subkey in correlations:
        ax.plot(correlations[real_subkey], label=f'Real subkey ({real_subkey:06b})', linewidth=2)
    else:
        # 如果真实子密钥不在相关性数据中，使用近似值
        # 找到最接近真实子密钥的键
        closest_key = min(correlations.keys(), key=lambda x: abs(x - real_subkey))
        ax.plot(correlations[closest_key], label=f'Real subkey approx ({closest_key:06b})', linewidth=2)
    
    # 绘制恢复子密钥的相关系数曲线（如果有效）
    if recovered_subkey >= 0 and recovered_subkey in correlations:
        ax.plot(correlations[recovered_subkey], label=f'Recovered subkey ({recovered_subkey:06b})', linewidth=2)
    else:
        # 绘制一个随机密钥作为示例
        sample_key = list(correlations.keys())[0]  # 使用第一个键作为示例
        ax.plot(correlations[sample_key], label=f'Sample key ({sample_key:06b})', linewidth=2)
    
    # 绘制几个随机候选密钥的相关系数曲线
    random.seed(42)
    keys_list = list(correlations.keys())
    plotted_keys = {real_subkey, recovered_subkey, sample_key}
    count = 0
    for key in keys_list:
        if key not in plotted_keys and count < 3:
            ax.plot(correlations[key], label=f'Random candidate {key:06b}', alpha=0.7)
            count += 1
    
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('CPA Attack Correlation Curves - S-box 1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return recovered_subkeys

if __name__ == "__main__":
    # 运行CPA攻击演示
    recovered_keys = demonstrate_cpa_attack()