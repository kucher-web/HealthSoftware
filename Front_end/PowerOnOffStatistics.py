import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    st.title("载管开关机时间统计")
    
    # 文件上传区域
    uploaded_files = st.file_uploader("请选择要分析的文件", accept_multiple_files=True, type=['txt', 'csv', 'xlsx'])
    
    if uploaded_files:
        st.write(f"已选择 {len(uploaded_files)} 个文件")
        
        # 创建分析按钮
        if st.button("开始分析"):
            # 创建结果表格
            results = []
            total_duration = 0
            
            for file in uploaded_files:
                try:
                    # 根据文件类型读取数据
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                    elif file.name.endswith('.xlsx'):
                        df = pd.read_excel(file)
                    else:
                        # 假设txt文件是CSV格式
                        df = pd.read_csv(file, sep='\t')

                    # 先对数据进行处理，需要根据时间戳排序
                    df = df.sort_values('timestamp')
                    
                    # 统计总时长和开关机时间
                    start_time = []
                    end_time = []
                    last_value = None
                    last_timestamp = None
                    
                    # 遍历表格，每一段开机中，第一个不为0的时间为开始时间，最后一个不为0的时间为结束时间
                    for index, row in df.iterrows():
                        current_value = row['TMY045-telemValue']
                        current_timestamp = row['timestamp']
                        
                        # 如果当前值不为0且上一个值为0，说明是开机开始
                        if current_value != 0 and (last_value == 0 or last_value is None):
                            start_time.append(current_timestamp)
                        
                        # 如果当前值为0且上一个值不为0，说明是关机，使用上一个时间戳作为结束时间
                        if current_value == 0 and last_value != 0 and last_value is not None:
                            end_time.append(last_timestamp)
                        
                        last_value = current_value
                        last_timestamp = current_timestamp
                    
                    # 如果最后一个状态是开机状态，需要添加结束时间
                    if last_value != 0 and len(start_time) > len(end_time):
                        end_time.append(last_timestamp)
                    
                    # 计算总时长和开关机次数
                    on_off_count = len(start_time)
                    total_duration = 0
                    
                    # 创建时间段详情表格
                    time_periods = []
                    for i in range(len(start_time)):
                        try:
                            # 尝试不同的时间格式
                            try:
                                start = datetime.strptime(start_time[i], '%Y-%m-%d %H:%M:%S.%f')
                            except ValueError:
                                start = datetime.strptime(start_time[i], '%Y-%m-%d %H:%M:%S')
                                
                            try:
                                end = datetime.strptime(end_time[i], '%Y-%m-%d %H:%M:%S.%f')
                            except ValueError:
                                end = datetime.strptime(end_time[i], '%Y-%m-%d %H:%M:%S')
                                
                            duration = (end - start).total_seconds()
                            total_duration += duration
                            
                            time_periods.append({
                                '开始时间': start_time[i],
                                '结束时间': end_time[i],
                                '持续时间(秒)': duration
                            })
                        except Exception as e:
                            st.warning(f"时间格式转换错误: {str(e)}")
                            st.warning(f"问题时间戳: 开始时间={start_time[i]}, 结束时间={end_time[i]}")
                    
                    # 将时间段按开始时间排序
                    time_periods_df = pd.DataFrame(time_periods)
                    if not time_periods_df.empty:
                        time_periods_df = time_periods_df.sort_values('开始时间')
                    
                    file_stats = {
                        '文件名': file.name,
                        '记录数': len(df),
                        '开关机次数': on_off_count,
                        '总工作时长(秒)': total_duration,
                        '开始时间列表': start_time,
                        '结束时间列表': end_time
                    }
                    results.append(file_stats)
                    
                    # 显示当前文件的时间段详情
                    st.subheader(f"文件 {file.name} 的时间段详情")
                    st.dataframe(time_periods_df)
                    
                except Exception as e:
                    st.error(f"处理文件 {file.name} 时出错: {str(e)}")
            
            # 显示汇总结果表格
            if results:
                st.subheader("文件汇总信息")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
            else:
                st.warning("没有生成任何结果") 