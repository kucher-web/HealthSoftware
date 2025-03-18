import pandas as pd
from sklearn.preprocessing import StandardScaler
import satellite

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

# 俯仰角和方位角的计算
def calculate_angles(df, qv_option):
    # 这里是计算俯仰角和方位角的逻辑
    # 这只是一个示例，您需要根据实际的计算方法来实现这个函数
    if qv_option == "QV1":
        theta = 1;
        phi = 1;
    else:  # QV2
        theta = 2;
        phi = 2;
    
    return theta, phi

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
    