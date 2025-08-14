import numpy as np

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
Key_64 = np.uint64
Key_56 = np.uint64
K = np.zeros(16, dtype=Key_64)
L = np.uint32
R = np.uint32
L_bit = np.zeros(32, dtype=np.uint8)
R_bit = np.zeros(32, dtype=np.uint8)




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

def merge(input_1: int, input_2: int) -> int:
    """合并两个 32 位为一个 64 位"""
    return ((input_1 & 0xFFFF_FFFF) << 32) | (input_2 & 0xFFFF_FFFF)

def ip_inverse_permutation(text: int) -> int:
    """执行 IP^-1 置换"""
    # 64 位文本转换为二进制
    text_bit = to_bit(text, 64)
    # IP^-1 置换
    result_bit = np.zeros(64, dtype=np.uint8)
    for i in range(64):
        result_bit[i] = text_bit[IP_Inv_Table[i]]
    result = to_data(result_bit, 64) & 0xFFFF_FFFF_FFFF_FFFF

    # 转换回整数
    return result

def des_encrypt(plaintext: int, key: int) -> int:
    """执行 DES 加密"""
    # 生成 16 轮子密钥
    sub_keys = key_permutation(key)
    
    # 初始置换
    permuted_bit = ip_permutation(plaintext)
    # 拆分为左右两部分
    left_bit, right_bit = split_text(permuted_bit)
    L = to_data(left_bit, 32)
    R = to_data(right_bit, 32)
    
    # 16 轮 Feistel 网络
    for i in range(16):
        temp = L
        L = R & 0xFFFFFFFF  # 确保32位
        R = (temp ^ f_function(R, sub_keys[i])) & 0xFFFFFFFF  # 确保32位
    
    # 交换左右部分并合并
    merged = merge(R, L)  # 注意：这里R和L的顺序是正确的
    
    # 最终置换
    return ip_inverse_permutation(merged)

def des_decrypt(ciphertext: int, key: int) -> int:
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

# # 先执行DES加密，这样会填充HD_list
# plaintext = 0x0123456789ABCDEF  
# key = 0x133457799BBCDFF1
# ciphertext = des_encrypt(plaintext, key)
# plaintext_decrypt = des_decrypt(ciphertext, key)
# print(f"明文: 0x{plaintext:016X}")
# print(f"密钥: 0x{key:016X}")
# print(f"密文: 0x{ciphertext:016X}")
# print(f"解密后的明文: 0x{plaintext_decrypt:016X}")
