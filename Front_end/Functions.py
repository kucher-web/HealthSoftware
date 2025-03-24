import pandas as pd
from sklearn.preprocessing import StandardScaler
import satellite
import numpy as np
from wuchafenxi0922 import wuchafenxi_py

# 处理快包数据
def template_filter(df, template_df):
    # 提取第一列的所有值
    values = template_df.iloc[:, 0].values.tolist()
    # 筛选df中包含values的列名
    filtered_columns = [col for col in df.columns if any(value in col for value in values)]
    filtered_df = df[['星上时间']+filtered_columns]
    return filtered_df

# 根据模板文件，合并快慢包数据
def template_based_filter(df_k, df_m, template_df):
    filtered_df1 = template_filter(df_k, template_df)
    filtered_df2 = template_filter(df_m, template_df)

    # 将两个df合并成一个df
    merged_df = pd.merge(filtered_df1, filtered_df2, on='星上时间', how='outer')
    merged_df
        
    return merged_df

# 是否删除源码数据
def DeleteYM(df):
    # 删除df数据框里面所有包含‘源码’的列
    df = df.loc[:, ~df.columns.str.contains('源码')]
    return df

# 筛选数据
def filter_data_angle(df):
    # 只保留与俯仰角和方位角计算相关的列
    # 删除只有一个非重复值的列
    columns = [col for col in df.columns if df[col].nunique() > 1]
    df = df[columns]
    # 把遥测代号和遥测名称合并，用于后续的图表展示
    df['遥测'] = df['遥测代号'] + '_' + df['遥测名称']
    df = df.drop(['遥测代号', '遥测名称','采集时间'], axis=1)
    # 只需要一部分数据，输入需要的遥测名称
    columns = ['TMK505', 'TMK506', 'TMK507', 'TMK508', 'TMK509', 'TMK510', 'TMZ148', 'TMZ150', 'TMZ151','TMZ152','TMZ153', 'TMO108', 'TMO109', 'TMO110', 'TMO111', 'TMO112', 'TMO113', 'TMO114', 'TMO115', 'TMZ067', 'TMZ068']
    # 查找遥测列中包含columns中的元素的行
    filtered_df = df[df['遥测'].notna()]  # First, filter out rows with NaN values in '遥测' column
    filtered_df = filtered_df[filtered_df['遥测'].str.contains('|'.join(columns))]

    # 新建一个表格，序列为星上时间，列为遥测名称，值为实际值
    df_pivot = filtered_df.pivot(index='星上时间', columns='遥测', values='实际值')
    # 重置索引
    df_pivot = df_pivot.reset_index()
    df_pivot
    return df_pivot

def calculate_angle(df_pivot):
    # 开始提取数据并进行相应计算
    # 1. 找到TMZ148出现的时间，找到对应的TMO110~115的值，作为TMO轨道数据
    # 只保留TMZ148不为空的行，以及下面一行
    notna_index = df_pivot[df_pivot['TMZ148_馈电指向时间_秒'].notna()].index
    df_calculate = df_pivot[df_pivot['TMZ148_馈电指向时间_秒'].notna()]

    # 找到TMZ148的下一行
    next_index = notna_index + 1
    df_calculate = pd.concat([df_calculate, df_pivot.loc[next_index]], axis=0)	# 把下一行加入到df_calculate中
    # 重置索引,按照星上时间排序
    df_calculate = df_calculate.sort_values(by='星上时间')
    df_calculate = df_calculate.reset_index(drop=True)
    
    # 开始计算
    # TMO110~115的值是轨道数据，提取6*3的数据
    # 6行数据，每行3个数据，不要空值
    TMO110 = df_calculate['TMO110_WGS84短时外推位置X'].dropna()
    TMO111 = df_calculate['TMO111_WGS84短时外推位置Y'].dropna()
    TMO112 = df_calculate['TMO112_WGS84短时外推位置Z'].dropna()
    TMO113 = df_calculate['TMO113_WGS84短时外推速度X'].dropna()
    TMO114 = df_calculate['TMO114_WGS84短时外推速度Y'].dropna()
    TMO115 = df_calculate['TMO115_WGS84短时外推速度Z'].dropna()
    TMO_data = np.array([TMO110, TMO111, TMO112, TMO113, TMO114, TMO115])
    TMO_data = TMO_data.T

    # TMK505~510的值是姿态数据，提取6*3的数据
    # 6行数据，每行3个数据，不要空值
    TMK505 = df_calculate['TMK505_轨道系姿态角EulerX'].dropna()
    TMK506 = df_calculate['TMK506_轨道系姿态角EulerY'].dropna()
    TMK507 = df_calculate['TMK507_轨道系姿态角EulerZ'].dropna()

    TMK508 = df_calculate['TMK508_轨道系角速度wbox'].dropna()
    TMK509 = df_calculate['TMK509_轨道系角速度wboy'].dropna()
    TMK510 = df_calculate['TMK510_轨道系角速度wboz'].dropna()
    # 6*3的数据
    ALPHA_data = np.array([TMK505, TMK506, TMK507])


    # W
    W_data = np.array([TMK508, TMK509, TMK510])
    W_data = W_data.T
    # 获取USER_POS数据
    USER_POS = np.array([313437, 4770653, 4208987])

    # 获取RV数据
    RV = TMO_data.astype(float)

    # 获取ALPHA数据,TMK505~507
    ALPHA = ALPHA_data.astype(float)

    # 获取W数据,TMK508~510
    W = W_data.astype(float)

    # 把数据带入进行计算
    theta ,phi , rang = wuchafenxi_py(USER_POS, RV, ALPHA, W)

    return theta, phi

    