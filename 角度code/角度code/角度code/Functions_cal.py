# 完整实现qv.m和相关函数的Python版本

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# 添加MATLAB与Python实现的qv.m结果对比函数

def compare_qv_implementations():
    """
    比较Python实现的qv.m与MATLAB原始qv.m的结果
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matlab.engine
    from scipy.spatial.transform import Rotation as R
    
    print("开始比较Python和MATLAB的qv.m实现...")
    
    # 1. 获取Python实现的结果
    try:
        faip_py, thetap_py, data1_py = qv_python()
        print("Python实现执行成功!")
    except Exception as e:
        print(f"Python实现执行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 获取MATLAB原始实现的结果
    try:
        print("启动MATLAB引擎...")
        eng = matlab.engine.start_matlab()
        eng.addpath(r'd:\CODE\HealthSoftware\角度code\角度code\角度code')
        
        print("调用MATLAB的qv.m...")
        # 调用MATLAB的qv.m并获取结果
        # 注意：如果qv.m没有返回值，需要修改MATLAB代码或使用其他方式获取结果
        try:
            # 尝试直接调用qv.m
            faip_matlab, thetap_matlab = eng.qv(nargout=2)
            matlab_results_available = True
        except Exception as e1:
            print(f"直接调用MATLAB的qv.m失败: {e1}")
            print("尝试创建临时MATLAB函数来获取qv.m的结果...")
            
            # 创建临时MATLAB函数来运行qv.m并返回结果
            temp_matlab_code = """
            function [faip, thetap] = temp_qv_wrapper()
                % 运行qv.m并捕获结果
                run('qv.m');
                % 假设qv.m创建了faip和thetap变量
            end
            """
            
            try:
                # 写入临时文件
                with open('temp_qv_wrapper.m', 'w') as f:
                    f.write(temp_matlab_code)
                
                # 运行临时函数
                faip_matlab, thetap_matlab = eng.temp_qv_wrapper(nargout=2)
                matlab_results_available = True
            except Exception as e2:
                print(f"使用临时包装函数调用MATLAB的qv.m失败: {e2}")
                
                # 最后尝试：运行qv.m然后从MATLAB工作区获取变量
                try:
                    eng.eval("run('qv.m');", nargout=0)
                    faip_matlab = eng.workspace['faip']
                    thetap_matlab = eng.workspace['thetap']
                    matlab_results_available = True
                except Exception as e3:
                    print(f"从MATLAB工作区获取结果失败: {e3}")
                    matlab_results_available = False
        
        if matlab_results_available:
            # 转换MATLAB结果为numpy数组
            faip_matlab = np.array(faip_matlab).flatten()
            thetap_matlab = np.array(thetap_matlab).flatten()
            
            # 3. 比较结果
            print("\n结果比较:")
            print("-" * 80)
            print(f"{'数据点':<10} {'Python方位角':<15} {'MATLAB方位角':<15} {'差异':<10} {'Python俯仰角':<15} {'MATLAB俯仰角':<15} {'差异':<10}")
            print("-" * 80)
            
            for i in range(min(len(faip_py), len(faip_matlab))):
                faip_diff = abs(faip_py[i] - faip_matlab[i])
                thetap_diff = abs(thetap_py[i] - thetap_matlab[i])
                print(f"{i:<10} {faip_py[i]:<15.4f} {faip_matlab[i]:<15.4f} {faip_diff:<10.4f} {thetap_py[i]:<15.4f} {thetap_matlab[i]:<15.4f} {thetap_diff:<10.4f}")
            
            # 计算总体差异
            faip_mean_diff = np.mean(np.abs(faip_py[:len(faip_matlab)] - faip_matlab[:len(faip_py)]))
            thetap_mean_diff = np.mean(np.abs(thetap_py[:len(thetap_matlab)] - thetap_matlab[:len(thetap_py)]))
            
            print("\n总体差异统计:")
            print(f"方位角平均绝对差异: {faip_mean_diff:.4f}度")
            print(f"俯仰角平均绝对差异: {thetap_mean_diff:.4f}度")
            
            # 可视化比较
            plt.figure(figsize=(14, 10))
            
            # 方位角比较
            plt.subplot(2, 1, 1)
            plt.plot(faip_py[:len(faip_matlab)], 'r-o', label='Python方位角')
            plt.plot(faip_matlab[:len(faip_py)], 'b--x', label='MATLAB方位角')
            plt.title('方位角比较')
            plt.xlabel('数据点')
            plt.ylabel('角度 (度)')
            plt.grid(True)
            plt.legend()
            
            # 俯仰角比较
            plt.subplot(2, 1, 2)
            plt.plot(thetap_py[:len(thetap_matlab)], 'r-o', label='Python俯仰角')
            plt.plot(thetap_matlab[:len(thetap_py)], 'b--x', label='MATLAB俯仰角')
            plt.title('俯仰角比较')
            plt.xlabel('数据点')
            plt.ylabel('角度 (度)')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # 极坐标比较
            plt.figure(figsize=(12, 12))
            ax = plt.subplot(111, projection='polar')
            
            # 转换为弧度并调整方向
            theta_py = np.radians(faip_py[:len(faip_matlab)])
            r_py = 90 - thetap_py[:len(thetap_matlab)]
            
            theta_matlab = np.radians(faip_matlab[:len(faip_py)])
            r_matlab = 90 - thetap_matlab[:len(thetap_py)]
            
            ax.scatter(theta_py, r_py, c='red', s=50, alpha=0.7, label='Python结果')
            ax.scatter(theta_matlab, r_matlab, c='blue', s=50, alpha=0.7, marker='x', label='MATLAB结果')
            
            ax.set_title('方位角和俯仰角的极坐标比较', va='bottom')
            ax.set_theta_zero_location('N')  # 北方向为0度
            ax.set_theta_direction(-1)  # 顺时针方向
            ax.set_rlim(0, 90)
            ax.set_rticks([0, 15, 30, 45, 60, 75, 90])
            ax.set_rlabel_position(0)
            ax.grid(True)
            ax.legend(loc='upper right')
            
            plt.show()
        else:
            print("无法获取MATLAB的qv.m结果，跳过比较")
        
        # 清理
        eng.quit()
        print("MATLAB引擎已关闭")
        
    except Exception as e:
        print(f"MATLAB比较过程中出错: {e}")
        import traceback
        traceback.print_exc()


def compare_wuchafenxi0922():
    """
    比较 MATLAB 和 Python 实现的 wuchafenxi0922 函数的结果
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matlab.engine
    from Front_end.wuchafenxi0922 import wuchafenxi0922 as wuchafenxi0922_py

    print("开始比较 MATLAB 和 Python 的 wuchafenxi0922 实现...")

    # 准备输入数据
    USER_POS = np.array([313437, 4770653, 4208987])
    RV1=np.array([-67623.57813,4548867.5,5540375.5,38.06939697,5776.424316,-4717.989746]);
    RV2=np.array([-63341.4375,4908262,5226393.5,95.10617065	,5450.527832,-5090.785645]);
    RV3=np.array([-55522.63281,5246013,4889252.5,148.4838715,5100.333496,-5441.143066]);
    RV4=np.array([-44421.28125,5560615.5,4530453,197.7364655,4727.712891,-5767.272949]);
    RV5=np.array([-30302.62305,5850656.5,4151589,242.703537,4333.28125,-6067.784668]);
    RV6=np.array([-13462.03516,6114912.5,3754381.5,282.7574463,	3919.611816,-6341.501953]);
    # 更新 ALPHAX, ALPHAY, ALPHAZ
    ALPHAX = np.array([0.00083238, 0.00175312, 0.00076279, -0.00037315, 0.00154832, -0.00128882]) * np.pi / 180
    ALPHAY = np.array([-0.01101539, -0.00096647, 0.00370367, -0.0054522, -0.01022821, -0.00089305]) * np.pi / 180
    ALPHAZ = np.array([-0.00259131, 0.00285641, -0.00152585, -0.00001243, 0.00115596, -0.00061824]) * np.pi / 180
    # Define W values
    W_1 = [-0.0007902, 0.00033979, 0.0002078]
    W_2 = [-0.00164376, 0.00156558, 0.0001073]
    W_3 = [-0.00053028, 0.00036893, -0.00014961]
    W_4 = [-0.00019678, 0.0010008, -0.00027919]
    W_5 = [-0.00209627, 0.00168203, 0.00033462]
    W_6 = [-0.00314025, 0.00112048, 0.00085791]
    W = np.array([W_1, W_2, W_3, W_4, W_5, W_6])

    # 1. 执行 Python 实现
    try:
        theta_py, phi_py, rang_py = wuchafenxi0922_py(USER_POS, RV1, RV2, RV3, RV4, RV5, RV6, ALPHAX, ALPHAY, ALPHAZ, W)
        print("Python 实现执行成功!")
    except Exception as e:
        print(f"Python 实现执行失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 执行 MATLAB 实现
    try:
        print("启动 MATLAB 引擎...")
        eng = matlab.engine.start_matlab()
        eng.addpath(r'd:\CODE\HealthSoftware\角度code\角度code\角度code')

        print("调用 MATLAB 的 wuchafenxi0922.m...")
        theta_matlab, phi_matlab, rang_matlab = eng.wuchafenxi0922(nargout=3)

        # 转换 MATLAB 结果为 numpy 数组
        theta_matlab = np.array(theta_matlab).flatten()
        phi_matlab = np.array(phi_matlab).flatten()
        rang_matlab = np.array(rang_matlab).flatten()

        print("MATLAB 实现执行成功!")
    except Exception as e:
        print(f"MATLAB 实现执行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        eng.quit()
        print("MATLAB 引擎已关闭")

    # 3. 比较结果
    print("\n结果比较:")
    print("-" * 80)
    print(f"{'参数':<10} {'Python 结果':<15} {'MATLAB 结果':<15} {'差异':<10}")
    print("-" * 80)
    
    for i in range(len(theta_py)):
        theta_diff = np.abs(theta_py[i] - theta_matlab[i])
        phi_diff = np.abs(phi_py[i] - phi_matlab[i])
        rang_diff = np.abs(rang_py[i] - rang_matlab[i])
        
        print(f"theta[{i}]".ljust(10), f"{theta_py[i]:.4f}".ljust(15), f"{theta_matlab[i]:.4f}".ljust(15), f"{theta_diff:.4f}")
        print(f"phi[{i}]".ljust(10), f"{phi_py[i]:.4f}".ljust(15), f"{phi_matlab[i]:.4f}".ljust(15), f"{phi_diff:.4f}")
        print(f"rang[{i}]".ljust(10), f"{rang_py[i]:.4f}".ljust(15), f"{rang_matlab[i]:.4f}".ljust(15), f"{rang_diff:.4f}")
        print("-" * 80)

    # 计算总体差异
    theta_mean_diff = np.mean(np.abs(theta_py - theta_matlab))
    phi_mean_diff = np.mean(np.abs(phi_py - phi_matlab))
    rang_mean_diff = np.mean(np.abs(rang_py - rang_matlab))
    
    print("\n总体差异统计:")
    print(f"theta平均绝对差异: {theta_mean_diff:.4f}")
    print(f"phi平均绝对差异: {phi_mean_diff:.4f}")
    print(f"rang平均绝对差异: {rang_mean_diff:.4f}")

    plt.figure(figsize=(15, 5))
        
    plt.subplot(1, 3, 1)
    plt.plot(theta_py, 'r-o', label='Python')
    plt.plot(theta_matlab, 'b--x', label='MATLAB')
    plt.title('theta')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(phi_py, 'r-o', label='Python')
    plt.plot(phi_matlab, 'b--x', label='MATLAB')
    plt.title('phi')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(rang_py, 'r-o', label='Python')
    plt.plot(rang_matlab, 'b--x', label='MATLAB')
    plt.title('rang')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 执行比较函数
    compare_wuchafenxi0922()  

def qv_python():
    """
    Python implementation of qv.m
    
    读取文本文件中的数据，并使用qv_kinematics函数计算方位角和俯仰角
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
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
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 方位角
    plt.subplot(2, 1, 1)
    plt.plot(faip, 'r-o', label='方位角')
    plt.title('方位角 (Azimuth)')
    plt.xlabel('数据点')
    plt.ylabel('角度 (度)')
    plt.grid(True)
    plt.legend()
    
    # 俯仰角
    plt.subplot(2, 1, 2)
    plt.plot(thetap, 'b-o', label='俯仰角')
    plt.title('俯仰角 (Elevation)')
    plt.xlabel('数据点')
    plt.ylabel('角度 (度)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 极坐标可视化
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    # 转换为弧度并调整方向
    theta = np.radians(faip)
    r = 90 - thetap  # 将俯仰角转换为极坐标半径
    
    ax.scatter(theta, r, c='red', s=50, alpha=0.7)
    ax.set_title('方位角和俯仰角的极坐标表示', va='bottom')
    ax.set_theta_zero_location('N')  # 北方向为0度
    ax.set_theta_direction(-1)  # 顺时针方向
    ax.set_rlim(0, 90)
    ax.set_rticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_rlabel_position(0)
    ax.grid(True)
    
    plt.show()
    
    return faip, thetap, data1