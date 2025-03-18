import numpy as np
from scipy.spatial.transform import Rotation as R
from Functions import siyuanshuchengfa, OutExtra

def wuchafenxi0922(USER_POS, RV1, RV2, RV3, RV4, RV5, RV6, ALPHAX, ALPHAY, ALPHAZ, W):
    # 用户位置
    X, Y, Z = USER_POS

    # 做轨道插值
    k1 = (RV2 - RV1) / (64 * 1000)  # 64秒的线性插值率
    k2 = (RV3 - RV2) / (64 * 1000)
    k3 = (RV4 - RV3) / (64 * 1000)
    k4 = (RV5 - RV4) / (64 * 1000)
    k5 = (RV6 - RV5) / (64 * 1000)

    RV1 = RV1 - k1*2750
    RV2 = RV2 - k1*2750
    RV3 = RV3 - k2*2750
    RV4 = RV4 - k3*2750
    RV5 = RV5 - k4*2750
    RV6 = RV6 - k5*2750

    # 卫星位置
    data1 = np.array([RV1, RV2, RV3, RV4, RV5, RV6])

    # 计算四元数
    Q1 = np.zeros((6, 4))
    Q2 = np.zeros((6, 4))
    Q3 = np.zeros((6, 4))
    Q1_STEP = np.zeros((6, 4))
    Q2_STEP = np.zeros((6, 4))
    for i in range(6):
        Q1[i] = np.array([np.sin(ALPHAX[i]/2), 0, 0, np.cos(ALPHAX[i]/2)])  
        Q2[i] = np.array([0, np.sin(ALPHAY[i]/2), 0, np.cos(ALPHAY[i]/2)])
        Q3[i] = np.array([0, 0, np.sin(ALPHAZ[i]/2), np.cos(ALPHAZ[i]/2)])
        Q1_STEP[i] = siyuanshuchengfa(Q3[i], Q1[i])
        Q2_STEP[i] = siyuanshuchengfa(Q1_STEP[i], Q2[i])

    # 外推到整秒的四元数
    Q = np.array([OutExtra(Q2_STEP[i], W[i], -2.85) for i in range(6)])

    # 安装矩阵
    AZJZ = 1

    # 初始化结果数组
    theta = np.zeros(6)
    phi = np.zeros(6)
    rang = np.zeros(6)
    Bpos = np.zeros((6, 3))

    for i in range(6):
        RV = data1[i]
        Qe = Q[i]

        # 84坐标系转轨道系
        omg = 7.2921151467e-5
        r = RV[:3]
        v = RV[3:] + omg * np.array([-r[1], r[0], 0])
        R_matrix = np.zeros((3, 3))
        R_matrix[2] = -r / np.linalg.norm(r)
        R_matrix[1] = np.cross(v, r) / np.linalg.norm(np.cross(v, r))
        R_matrix[0] = np.cross(R_matrix[1], R_matrix[2])

        Porb = R_matrix @ (USER_POS - RV[:3])

        

        # 质心轨道到本体
        A = R.from_quat(Qe).as_matrix()
        # A保留的小数不同导致Psat结果也不同
        Psat = np.linalg.inv(A) @ Porb
        # Psat = A @ Porb
        Bpos[i,:] = np.dot(AZJZ, Psat).T

        # 计算方位角俯仰角和距离
        # 计算俯仰角
        theta[i] = np.degrees(np.arctan(np.sqrt(Bpos[i,0]**2 + Bpos[i,1]**2) / Bpos[i,2]))
        
        # 计算方位角
        phi[i] = np.degrees(np.arctan(Bpos[i,1] / Bpos[i,0]))
        
        # 计算距离
        rang[i] = np.sqrt(Bpos[i,0]**2 + Bpos[i,1]**2 + Bpos[i,2]**2)

        if theta[i] < 0:
            theta[i] = 180 + theta[i]

        if Bpos[i,0] < 0 and Bpos[i,1] > 0:  # 二象限
            phi[i] = 180 + phi[i]
        elif Bpos[i,0] < 0 and Bpos[i,1] < 0:  # 三象限
            phi[i] = phi[i] + 180
        elif Bpos[i,0] > 0 and Bpos[i,1] < 0:  # 四象限
            phi[i] = 360 + phi[i]

    return theta, phi, rang

# 示例使用
if __name__ == "__main__":
    USER_POS = np.array([313437, 4770653, 4208987])
    RV1 = np.array([-67623.57813, 4548867.5, 5540375.5, 38.06939697, 5776.424316, -4717.989746])
    RV2 = np.array([-63341.4375, 4908262, 5226393.5, 95.10617065, 5450.527832, -5090.785645])
    RV3 = np.array([-55522.63281, 5246013, 4889252.5, 148.4838715, 5100.333496, -5441.143066])
    RV4 = np.array([-44421.28125, 5560615.5, 4530453, 197.7364655, 4727.712891, -5767.272949])
    RV5 = np.array([-30302.62305, 5850656.5, 4151589, 242.703537, 4333.28125, -6067.784668])
    RV6 = np.array([-13462.03516, 6114912.5, 3754381.5, 282.7574463, 3919.611816, -6341.501953])
    
    ALPHAX = np.array([0.00083238, 0.00175312, 0.00076279, -0.00037315, 0.00154832, -0.00128882]) * np.pi / 180
    ALPHAY = np.array([-0.01101539, -0.00096647, 0.00370367, -0.0054522, -0.01022821, -0.00089305]) * np.pi / 180
    ALPHAZ = np.array([-0.00259131, 0.00285641, -0.00152585, -0.00001243, 0.00115596, -0.00061824]) * np.pi / 180
    
    W = np.array([
        [-0.0007902, 0.00033979, 0.0002078],
        [-0.00164376, 0.00156558, 0.0001073],
        [-0.00053028, 0.00036893, -0.00014961],
        [-0.00019678, 0.0010008, -0.00027919],
        [-0.00209627, 0.00168203, 0.00033462],
        [-0.00314025, 0.00112048, 0.00085791]
    ])

    theta, phi, rang = wuchafenxi0922(USER_POS, RV1, RV2, RV3, RV4, RV5, RV6, ALPHAX, ALPHAY, ALPHAZ, W)
    print("Theta (俯仰角):", theta)
    print("Phi (方位角):", phi)
    print("Range (距离):", rang)