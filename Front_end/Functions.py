import pandas as pd
from sklearn.preprocessing import StandardScaler

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