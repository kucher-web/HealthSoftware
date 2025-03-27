# 完整实现qv.m和相关函数的Python版本

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

# 用python实现qv_kinematics.m方法
def qv_kinematics(thetay, thetax, R_mount):
    # 旋转矩阵生成函数
    def roty(deg):
        θ = np.radians(deg)
        return np.array([
            [np.cos(θ), 0, np.sin(θ)],
            [0, 1, 0],
            [-np.sin(θ), 0, np.cos(θ)]
        ])
    
    def rotx(deg):
        θ = np.radians(deg)
        return np.array([
            [1, 0, 0],
            [0, np.cos(θ), -np.sin(θ)],
            [0, np.sin(θ), np.cos(θ)]
        ])
    
    # 核心计算
    gamma = 29  # 天线固定角
    qv = R_mount @ roty(gamma) @ roty(thetay) @ rotx(thetax) @ roty(gamma).T @ np.array([0, 0, 1])
    
    # 单位矢量化
    x, y, z = qv / np.linalg.norm(qv)
    
    # 俯仰角计算
    thetap = np.degrees(np.arccos(z))
    
    # 方位角计算
    if y >= 0:
        faip = np.degrees(np.arctan2(y, x))
    else:
        faip = 360 + np.degrees(np.arctan2(y, x))
    
    return faip, thetap

def siyuanshuchengfa(p, q):
    """
    Python实现四元数乘法 (siyuanshuchengfa.m)
    
    参数:
    p, q: 两个四元数 [q1, q2, q3, q4]，其中q4是标量部分
    
    返回:
    normalizedMatrixQ: 归一化后的四元数乘积
    """
    # 构建四元数乘法矩阵
    Q = np.array([
        [p[3], -p[2], p[1], p[0]],
        [p[2], p[3], -p[0], p[1]],
        [-p[1], p[0], p[3], p[2]],
        [-p[0], -p[1], -p[2], p[3]]
    ]) @ q.T
    
    # 归一化
    Norm = np.linalg.norm(Q)
    normalizedMatrixQ = Q / Norm
    
    # 如果标量部分为负，取反
    if normalizedMatrixQ[3] < 0:
        normalizedMatrixQ = -normalizedMatrixQ
    
    return normalizedMatrixQ

def RV2RTP_84(theta, phi, RV, Q):
    """
    Python实现RV2RTP_84.m函数
    
    参数:
    theta: 俯仰角 (度)
    phi: 方位角 (度)
    RV: 卫星位置和速度向量 [x, y, z, vx, vy, vz]
    Q: 卫星姿态四元数 [q1, q2, q3, q4]，其中q1是标量部分
    
    返回:
    d: 在WGS84坐标系下的位置坐标矢量
    """
    # 计算本体坐标系下的单位方向矢量
    v_body = np.array([
        np.cos(np.radians(90-theta)) * np.cos(np.radians(phi)),
        np.cos(np.radians(90-theta)) * np.sin(np.radians(phi)),
        np.sin(np.radians(90-theta))
    ])
    
    # 本体到质心轨道
    q1, q2, q3, q4 = Q
    
    A = np.array([
        [2*(q1**2+q4**2)-1, 2*(q1*q2+q3*q4), 2*(q1*q3-q2*q4)],
        [2*(q1*q2-q3*q4), 2*(q2**2+q4**2)-1, 2*(q2*q3+q1*q4)],
        [2*(q1*q3+q2*q4), 2*(q2*q3-q1*q4), 2*(q3**2+q4**2)-1]
    ])
    
    A1 = np.linalg.inv(A)
    porb = A1 @ v_body
    
    # 轨道系到84系，修正速度值
    omg = 7.2921151467e-5  # 地球自转角速度
    r = RV[:3]
    v = RV[3:6] + omg * np.array([-r[1], r[0], 0])
    
    # 构建轨道坐标系
    R = np.zeros((3, 3))
    R[2, :] = -r / np.linalg.norm(r)
    R[1, :] = np.cross(v, r) / np.linalg.norm(np.cross(v, r))
    R[0, :] = np.cross(R[1, :], R[2, :])
    
    R1 = np.linalg.inv(R)
    d = R1 @ porb  # 生成在84坐标系下的位置坐标矢量
    
    return d

# Python实现qv.m的功能



# 用python实现OutExtra.m的功能
def OutExtra(p, W, delt):
    """
    根据姿态四元数积分初值与角速度，积分计算姿态四元数。

    参数:
    p : array_like
        姿态四元数初值，格式为 [q1, q2, q3, q4]，其中q1为标量部分
    W : array_like
        姿态角速度ω，格式为 [wx, wy, wz]
    delt : float
        积分时间 delta t

    返回:
    Q : ndarray
        计算得到的姿态四元数
    """
    # 计算角速度的模
    modW = np.sqrt(np.sum(W**2))  # 注意原始数据以deg为单位
    
    # 转换为弧度单位
    theta = modW * delt * np.pi / 180
    
    # 计算四元数增量
    q = np.zeros(4)
    q[0:3] = (W / modW) * np.sin(theta/2)
    q[3] = np.cos(theta/2)
    
    # 构建四元数乘法矩阵
    M = np.array([
        [p[3], -p[2],  p[1], p[0]],
        [p[2],  p[3], -p[0], p[1]],
        [-p[1],  p[0],  p[3], p[2]],
        [-p[0], -p[1], -p[2], p[3]]
    ])
    
    # 计算新的四元数
    Q = M @ q
    
    # 归一化
    modQ = np.linalg.norm(Q)
    Q = Q / modQ
    
    # 如果标量部分为负，取反
    if Q[3] < 0:
        Q = -Q
    
    return Q

def qv_python():
    """
    Python implementation of qv.m
    
    读取文本文件中的数据，并使用qv_kinematics函数计算方位角和俯仰角
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial.transform import Rotation as R
    
    # 读取文本文件
    try:
        # 尝试使用pandas读取文件
        data1 = pd.read_csv('./12.12.1.txt', delim_whitespace=True, header=None)
        data1 = data1.values  # 转换为numpy数组
    except Exception as e:
        print(f"使用pandas读取文件失败: {e}")
        # 备选方案：使用numpy直接读取
        try:
            data1 = np.loadtxt('12.12.1.txt')
        except Exception as e2:
            print(f"使用numpy读取文件失败: {e2}")
            # 如果文件格式特殊，手动读取
            data1 = []
            with open('./12.12.1.txt', 'r') as f:
                for line in f:
                    values = [float(x) for x in line.strip().split()]
                    data1.append(values)
            data1 = np.array(data1)
    
    print("读取的数据:")
    print(data1)
    
    # 定义安装矩阵
    QV_1_ZH07_Z01_01 = np.array([
        [0.985381744558545, -0.170360844946125, -0.000429350982799],
        [0.170360844946125, 0.985381744558545, 0.000312413931025],
        [0.000370009792980, -0.000382227096880, 0.999999858314826]
    ])
    
    # 使用scipy创建旋转矩阵，相当于MATLAB中的rotz(-60)
    QV_2_ZH09_Z01_01 = R.from_euler('z', -60, degrees=True).as_matrix()
    
    # 初始化结果数组
    s = data1.shape[0]
    faip = np.zeros(s)
    thetap = np.zeros(s)
    
    # 对每行数据调用qv_kinematics函数
    for i in range(s):
        faip[i], thetap[i] = qv_kinematics(data1[i, 0], data1[i, 1], QV_2_ZH09_Z01_01)
    
    # 显示结果
    print("\n计算结果:")
    results = pd.DataFrame({
        'thetay': data1[:, 0],
        'thetax': data1[:, 1],
        '方位角(faip)': faip,
        '俯仰角(thetap)': thetap
    })
    print(results)
    
    return faip, thetap, data1