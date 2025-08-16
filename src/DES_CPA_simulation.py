import numpy as np
import matplotlib.pyplot as plt
import secrets
import math

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
L = np.uint32
R = np.uint32
L_bit = np.zeros(32, dtype=np.uint8)
R_bit = np.zeros(32, dtype=np.uint8)
key_true = 0x133457799BBCDFF1

def HD_calculate(data1:int, data2:int) -> int:
    """记录时间并计算汉明距离"""
    #计算汉明距离
    xor_result = data1 ^ data2
    return bin(xor_result).count('1')

def HW(data: int):
    """计算汉明重量"""
    return bin(data).count('1')
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

def ip_permutation(text: int) -> np.ndarray:
    """执行 IP 置换操作"""
    text_bit = to_bit(text, 64)
    result_bit = np.zeros(64, dtype=np.uint8)
    for i in range(64):
        result_bit[i] = text_bit[IP_Table[i]]
    return result_bit

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

def p_permutation(temp_bit:np.ndarray) -> np.ndarray:
    """执行 p 置换操作"""
    result_bit = np.zeros(32, dtype=np.uint8)
    for i in range(32):
        result_bit[i] = temp_bit[P_Table[i]]
    return result_bit

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
    C_bit = Key_56_bit[:28].copy()  # 直接使用位数组
    D_bit = Key_56_bit[28:].copy()  # 直接使用位数组
    
    # 16 轮密钥生成
    for i in range(16):
        # 左移操作
        shift = int(Move_Times[i])
        # 循环左移
        C_bit = np.roll(C_bit, -shift)
        D_bit = np.roll(D_bit, -shift)
        # 合并 C 和 D
        CD_bit = np.concatenate([C_bit, D_bit])
        CD = to_data(CD_bit, 56)&0xFF_FFFF_FFFF_FFFF
        # PC-2 置换
        K_bit = np.zeros(48, dtype=np.uint8)
        for j in range(48):
            K_bit[j] = CD_bit[PC_2[j]]
        
        # 转换为整数并确保结果是48位
        K[i] = (to_data(K_bit, 48) & ((1 << 48) - 1)) & 0xFFFF_FFFF_FFFF
    return K
def f_function(text: int, key_48: int, HD_list: list[int]) -> int:
    """ F 函数"""
    # 32 位文本转换为二进制
    text_bit = to_bit(text, 32)
    
    # E 扩展
    temp_1_bit = e_extension(text_bit)
    temp_1 = to_data(temp_1_bit, 48) & 0xFFFF_FFFF_FFFF
    #计算 E 扩展的汉明距离
    HD_list.append(HD_calculate(text, temp_1))
    # 与子密钥异或
    temp_yh = temp_1 ^ key_48
    temp_yh_bit = to_bit(temp_yh, 48)
    #计算并存储异或操作的汉明距离
    HD_list.append(HD_calculate(temp_1, temp_yh))
    # S 盒置换
    temp_2_bit = s_substitution(temp_yh_bit)
    temp_2 = to_data(temp_2_bit, 32) & 0xFFFF_FFFF
    # 计算并存储 S 盒置换的汉明距离
    HD_list.append(HD_calculate(temp_yh, temp_2))
    # P 置换
    result_bit = p_permutation(temp_2_bit)
    # 转换回整数
    result = to_data(result_bit, 32) & 0xFFFF_FFFF
    # 计算并存储 P 置换的汉明距离
    HD_list.append(HD_calculate(temp_2, result))

    return result

def merge(input_1: int, input_2: int) -> int:
    """合并两个 32 位为一个 64 位"""
    return ((input_1 & 0xFFFF_FFFF) << 32) | (input_2 & 0xFFFF_FFFF)

def ip_inverse_permutation(text: int, HD_list: list[int]) -> int:
    """执行 IP^-1 置换"""
    # 64 位文本转换为二进制
    text_bit = to_bit(text, 64)
    # IP^-1 置换
    result_bit = np.zeros(64, dtype=np.uint8)
    for i in range(64):
        result_bit[i] = text_bit[IP_Inv_Table[i]]
    result = to_data(result_bit, 64) & 0xFFFF_FFFF_FFFF_FFFF
    # 计算 IP^-1 置换的汉明距离
    HD_list.append(HD_calculate(text, result))

    # 转换回整数
    return result

def des_encrypt(plaintext: int, key: int, HD_list: list[int]) -> int:
    """执行 DES 加密"""
    # 生成 16 轮子密钥
    sub_keys = key_permutation(key)
    
    # 初始置换
    permuted_bit = ip_permutation(plaintext)
    # 计算IP 置换的汉明距离
    HD_list.append(HD_calculate(plaintext, to_data(permuted_bit, 64) & 0xFFFF_FFFF_FFFF_FFFF))
    # 拆分为左右两部分
    left_bit, right_bit = split_text(permuted_bit)
    L = to_data(left_bit, 32)
    R = to_data(right_bit, 32)
    
    # 16 轮 Feistel 网络
    for i in range(16):
        temp = L
        L = R & 0xFFFFFFFF  # 确保32位
        R = (temp ^ f_function(R, sub_keys[i], HD_list)) & 0xFFFFFFFF  # 确保32位
        HD_list.append(HD_calculate(temp, R))
    
    # 交换左右部分并合并
    temp_merged = merge(L, R)
    merged = merge(R, L)  # 注意：这里R和L的顺序是正确的
    HD_list.append(HD_calculate(temp_merged, merged))
    # 最终置换
    return ip_inverse_permutation(merged, HD_list)


'''def des_decrypt(ciphertext: int, key: int) -> int:
    """执行 DES 解密"""
    # 生成 16 轮子密钥（与加密使用相同的密钥生成算法）
    sub_keys = key_permutation(key)
    
    # 初始置换
    permuted = ip_permutation(ciphertext)
    
    # 拆分为左右两部分
    left, right = split_text(permuted)
    L = to_data(left, 32)
    R = to_data(right, 32)
    
    # 16 轮 Feistel 网络（解密时子密钥顺序相反）
    for i in reversed(range(16)):
        temp = L
        L = R
        R = temp ^ f_function(R, sub_keys[i])
    
    # 交换左右部分并合并
    merged = merge(R, L)
    
    # 最终置换
    return ip_inverse_permutation(merged) 
'''
# # 噪声功耗模型
# def noisy_power_model(HD, alpha=1.0, beta=0.5, noise_std=0.1) ->float:
#     base_power = alpha * (HD ** beta)
#     noise = np.random.normal(0, noise_std) 
#     return base_power + noise

#np.random.seed(42)

# power_values = [noisy_power_model(hd) for hd in HD_list]
# power_list = list(power_values)

def get_sample_points(plaintexts_list: list[int], key: int) -> list[list[int]]:
    """获取样本点"""
    T_list = []
    for plaintext in plaintexts_list:
        HD_list = []
        des_encrypt(plaintext, key, HD_list)
        T_list.append(HD_list)
    
    return T_list

def get_before_S(plaintext: int) -> int:
    """获取进入S盒之前的值"""
    # 初始置换
    permuted_bit = ip_permutation(plaintext)
    # 将置换后的明文分为两组32位
    R = permuted_bit[32:]
    # E 扩展
    plaintext_e_bit = e_extension(R)
    plaintext_e = (to_data(plaintext_e_bit, 48)) & 0xFFFF_FFFF_FFFF
    return plaintext_e

def s_substitution_2(temp_bit: np.ndarray, n: int) -> np.ndarray:
    """执行 S 盒置换操作"""
    result_bit = np.zeros(4, dtype=np.uint8)
    # 计算行和列
    row = (temp_bit[0] << 1) | temp_bit[5]
    col = (temp_bit[1] << 3) | (temp_bit[2] << 2) | (temp_bit[3] << 1) | temp_bit[4]
    
    # 查找 S 盒并转换为 4 位二进制
    value = Sbox[n, row, col]
    for j in range(4):
        result_bit[j] = (value >> (3 - j)) & 1
    return result_bit

def get_median(plaintext_e: int, key: int, n: int) -> tuple[int, int]:
    """获取中间值"""
    temp_1 = (plaintext_e >> (42 - n * 6)) & 0x3F
    # 与子密钥异或
    temp_yh = temp_1 ^ key
    temp_yh_bit = to_bit(temp_yh, 6)
    # S 盒置换
    temp_2_bit = s_substitution_2(temp_yh_bit, n)
    temp_2 = to_data(temp_2_bit, 4) & 0xF

    return temp_yh,temp_2

def power_consumption (plaintexts_e_list:list[int]) -> list[list[list[int]]]:
    """求得能量消耗矩阵"""
    power_consumption_matrix = [[[0 for _ in range(1000)] for _ in range(64)] for n in range(8)]
    for n in range(8):
        for i in range(64):
            s_output = [get_median(plaintext_e, i,n)[1] for plaintext_e in plaintexts_e_list]
            # 计算汉明距离并存储结果
            power_consumption_matrix[n][i] = [HW(element_out) for element_out in s_output]

    return power_consumption_matrix

def H_average(H: list[list[list[int]]]) -> list[list[float]]:
    """计算每种密钥假设下1000组明文所消耗功耗的平均值"""
    H_average = [[sum(H[i][j]) / len(H[i][j]) for j in range(64)] for i in range(8)]
    return H_average

def T_average(T: list[list[int]]) -> list[float]:
    """计算每个样本点下1000组明文所消耗功耗的平均值"""
    T_average = list[float]()
    for i in range(len(T[0])):
        T_average.append(sum([T[j][i] for j in range(len(T))]) / len(T))
    return T_average

def calculate_correlation(H: list[list[list[int]]], T: list[list[int]], H_average: list[list[float]], T_average: list[float]) -> list[list[list[float]]]:
    """计算相关系数"""
    R = [[[0.0 for _ in range(len(T[0]))] for _ in range(64)] for _ in range(8)]
    for n in range(8):
        for i in range(64):
            for j in range(len(T[0])):
                # 为每个j值重新初始化计算变量
                sum_val = 0.0
                temp1 = 0.0
                temp2 = 0.0
                for d in range(1000):
                    h_d = (H[n][i][d]-H_average[n][i])
                    t_d = (T[d][j]-T_average[j])
                    sum_val += h_d * t_d
                    temp1 += h_d * h_d
                    temp2 += t_d * t_d
                denominator = math.sqrt(temp1 * temp2)
                R[n][i][j] = sum_val / denominator if denominator != 0 else 0
    
    return R

# def plt_show():
#     # 设置图片清晰度
#     plt.rcParams['figure.dpi'] = 300
#     # 创建图形对象，设置尺寸为5x3英寸
#     plt.figure(figsize=(5, 3))
#     # 绘制汉明重量随样本点变化的曲线（蓝色实线，线宽2）
#     sample_points = list(range(len(HD_list)))
#     plt.plot(sample_points, HD_list, 'b-', linewidth=1, label='Hamming Weight')
#     # 在曲线上叠加数据点（红色圆点，大小25，z轴顺序2确保显示在最上层）
#     plt.scatter(sample_points, HD_list, color='red', s=2, zorder=2)
#     # 设置图表标题（字体大小16，与图表间距25）
#     plt.title('Hamming Weight over Sample Points', fontsize=16, pad=25)
#     # 设置坐标轴标签（字体大小6）
#     plt.xlabel('Sample Points', fontsize=6)
#     plt.ylabel('Hamming Weight', fontsize=6)
#     # 添加网格线（虚线样式，透明度70%）
#     plt.grid(True, linestyle='--', alpha=0.7)
#     # 显示图例
#     plt.legend()
#     # 自动调整布局，确保所有元素都适合图表区域
#     plt.tight_layout()
#     # 显示图表
#     plt.show()

# 生成1000个明文并存储到列表
plaintexts_list = [int.from_bytes(secrets.token_bytes(8), 'big') for _ in range(1000)]

#获取样本点——1000*83
T_list = get_sample_points(plaintexts_list,key_true)

# 经过E扩展后的明文列表
plaintexts_e_list = [get_before_S(plaintext) for plaintext in plaintexts_list]

# 获取能量消耗矩阵——8*64*1000
H_list = power_consumption(plaintexts_e_list)

# 获取每种密钥假设下1000组明文所消耗功耗的平均值——8*64
H_average_list = H_average(H_list)

# 获取每个样本点下1000组明文所消耗功耗的平均值——83
T_average_list = T_average(T_list)

# 获取相关系数矩阵——8*64*83
r = calculate_correlation(H_list, T_list, H_average_list, T_average_list)

# 获取相关系数矩阵每一列的最大值——8*64
r_column_max = [[max(r[n][i]) for i in range(64)] for n in range(8)]

def show_figure(r_column_max):
    """绘制各S盒相关系数对比图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2行4列的子图，设置整体大小
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('各S盒相关系数对比', fontsize=16, y=0.95)  # 总标题
    
    # 遍历每个子图位置（2行4列）
    for row in range(2):
        for col in range(4):
            n = row * 4 + col  # 计算对应的S盒索引（0-7）
            ax = axes[row, col]  # 获取当前子图
            
            # 获取当前S盒的数据
            data = r_column_max[n]
            x = range(len(data))
            
            # 绘制当前S盒的相关系数曲线
            ax.plot(x, data, 'b-', linewidth=1.5, 
                    marker='o', markersize=3, label=f'S{n}盒')
            
            # 找到最高点的位置和值
            max_value = max(data)
            max_index = data.index(max_value)
            
            # 标记最高点（红色）
            ax.plot(max_index, max_value, 'ro', markersize=4, label='最大值点')
            
            # 添加红色虚线投影到x轴
            ax.axvline(x=max_index, color='red', linestyle='--', alpha=0.7)
            
            # 在x轴上显示最高点坐标
            ax.text(max_index, ax.get_ylim()[0], f'({max_index}, {max_value:.4f})', 
                    ha='center', va='top', color='red', fontweight='bold')
            
            # 设置子图标题和坐标轴标签
            ax.set_title(f'S{n}盒', fontsize=12)
            ax.set_xlabel('假设密钥', fontsize=10)
            ax.set_ylabel('相关系数', fontsize=10)
            
            # 添加网格和图例
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=8)
    
    # 调整子图间距，避免重叠
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 调整顶部间距，避免总标题被遮挡
    plt.show()
    return 0

def merge_key(max_list: list[int]) -> int:
    """将最大相关系数对应的假设密钥索引合并为一个整数"""
    key = 0
    for i in range(8):
        key |= (max_list[i] << (42 - i * 6))
    return key

def right_shift_28(bits: np.ndarray) -> np.ndarray:
    """28位位数组右移1位（最后一位移到首位）"""
    return np.concatenate([[bits[-1]], bits[:-1]])

def recover_master_key(k1: int) -> list[int]:
    """从48位第一轮子密钥k1恢复64位主密钥"""
    candidate_key_list = list[int]()
    # 确定PC-2中未使用的56位位置（8个未知位）
    all_56_pos = set(range(56))
    pc2_used_pos = set(PC_2)
    unknown_pos = sorted(all_56_pos - pc2_used_pos)  # 8个未知位的位置
    
    # 将k1转换为48位位数组
    k1_bits = to_bit(k1, 48)
    
    # 遍历所有可能的8位未知值（0-255）
    for unknown_value in range(256):
        # 构造56位C1D1位数组
        c1d1_bits = np.zeros(56, dtype=np.uint8)
        # 填充已知位（来自k1）
        for i in range(48):
            c1d1_bits[PC_2[i]] = k1_bits[i]
        # 填充未知位
        unknown_bits = to_bit(unknown_value, 8)
        for i in range(8):
            c1d1_bits[unknown_pos[i]] = unknown_bits[i]
        
        # 拆分C1和D1，右移1位得到C0和D0
        c1 = c1d1_bits[:28]
        d1 = c1d1_bits[28:]
        c0 = right_shift_28(c1)  # 反向左移（左移的逆操作是右移）
        d0 = right_shift_28(d1)
        
        # 合并C0和D0得到56位C0D0
        c0d0_bits = np.concatenate([c0, d0])
        
        # 反向PC-1置换，构造64位主密钥（含奇偶校验位）
        master_bits = np.zeros(64, dtype=np.uint8)
        for i in range(56):
            master_bits[PC_1[i]] = c0d0_bits[i]
        
        # 计算奇偶校验位（每个字节第8位确保奇数个1）
        for byte in range(8):
            start = byte * 8
            end = start + 7  # 前7位是数据位
            parity_pos = start + 7  # 校验位位置
            bit_count = np.sum(master_bits[start:end])
            master_bits[parity_pos] = 1 if (bit_count % 2 == 0) else 0
        
        # 验证候选主密钥：生成第一轮子密钥是否与k1一致
        candidate_key = to_data(master_bits, 64)
        if key_permutation(candidate_key)[0] == k1:
            candidate_key_list.append(candidate_key)
    
    return candidate_key_list

# 获取攻击出的正确密钥
max_list = [max_item.index(max(max_item)) for max_item in r_column_max]

print("各S盒最大差分值对应的假设密钥索引：", max_list)

# 合并最大相关系数对应的假设密钥索引为一个整数
merged_key = merge_key(max_list)

print("合并后的假设密钥：", hex(merged_key))

# 从48位第一轮子密钥恢复64位主密钥
candidate_key_list = recover_master_key(merged_key)


print("恢复的主密钥：", [hex(candidate_key) for candidate_key in candidate_key_list])
print(len(candidate_key_list), "个候选主密钥")

plaintext = 0x0123456789ABCDEF
ciphertext = 0x85E813540F0AB405
for candidate_key in candidate_key_list:
    if des_encrypt(plaintext, candidate_key, []) == ciphertext:
        print("正确的主密钥：", hex(candidate_key))
        break

show_figure(r_column_max)