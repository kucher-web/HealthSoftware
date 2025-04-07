import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def parse_timestamp(timestamp_str):
    try:
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

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
            all_time_periods = []  # 用于存储所有文件的时间段
            
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
                    
                    # 找到所有时间戳突变点（15分钟以上）
                    time_gaps = []
                    last_timestamp = None
                    
                    for index, row in df.iterrows():
                        current_timestamp = row['timestamp']
                        if last_timestamp is not None:
                            try:
                                current_dt = parse_timestamp(current_timestamp)
                                last_dt = parse_timestamp(last_timestamp)
                                time_diff = (current_dt - last_dt).total_seconds()
                                
                                if time_diff > 900:  # 15分钟 = 900秒
                                    time_gaps.append({
                                        'start': last_timestamp,
                                        'end': current_timestamp,
                                        'gap_duration': time_diff
                                    })
                            except Exception as e:
                                st.warning(f"时间差计算错误: {str(e)}")
                        last_timestamp = current_timestamp
                    
                    # 根据时间戳突变点分割数据，并筛选出 TMY045_telemValue 不为0的时间段
                    time_periods = []
                    last_end = None
                    
                    for gap in time_gaps:
                        # 获取两个突变点之间的数据
                        if last_end is None:
                            period_data = df[df['timestamp'] <= gap['start']]
                        else:
                            period_data = df[(df['timestamp'] > last_end) & (df['timestamp'] <= gap['start'])]
                        
                        # 筛选出 TMY045_telemValue 不为0的时间段
                        powered_on_data = period_data[period_data['TMY045-telemValue'] != 0]
                        
                        if not powered_on_data.empty:
                            start_time = powered_on_data.iloc[0]['timestamp']
                            end_time = powered_on_data.iloc[-1]['timestamp']
                            
                            try:
                                start_dt = parse_timestamp(start_time)
                                end_dt = parse_timestamp(end_time)
                                duration = (end_dt - start_dt).total_seconds()
                                
                                time_periods.append({
                                    '文件名': file.name,
                                    '开始时间': start_time,
                                    '结束时间': end_time,
                                    '持续时间(秒)': duration
                                })
                            except Exception as e:
                                st.warning(f"时间格式转换错误: {str(e)}")
                        
                        last_end = gap['start']
                    
                    # 处理最后一个时间段
                    if last_end is not None:
                        period_data = df[df['timestamp'] > last_end]
                        powered_on_data = period_data[period_data['TMY045-telemValue'] != 0]
                        
                        if not powered_on_data.empty:
                            start_time = powered_on_data.iloc[0]['timestamp']
                            end_time = powered_on_data.iloc[-1]['timestamp']
                            
                            try:
                                start_dt = parse_timestamp(start_time)
                                end_dt = parse_timestamp(end_time)
                                duration = (end_dt - start_dt).total_seconds()
                                
                                time_periods.append({
                                    '文件名': file.name,
                                    '开始时间': start_time,
                                    '结束时间': end_time,
                                    '持续时间(秒)': duration
                                })
                            except Exception as e:
                                st.warning(f"时间格式转换错误: {str(e)}")
                    
                    # 计算总工作时长
                    total_duration = sum(period['持续时间(秒)'] for period in time_periods)
                    
                    # 将时间段添加到总列表中
                    all_time_periods.extend(time_periods)
                    
                    # 创建文件统计信息
                    file_stats = {
                        '文件名': file.name,
                        '记录数': len(df),
                        '开关机次数': len(time_periods),
                        '总工作时长(秒)': total_duration
                    }
                    results.append(file_stats)
                    
                    # 显示当前文件的时间段详情
                    st.subheader(f"文件 {file.name} 的时间段详情")
                    if time_periods:
                        time_periods_df = pd.DataFrame(time_periods)
                        time_periods_df = time_periods_df.sort_values('开始时间')
                        st.dataframe(time_periods_df)
                    else:
                        st.warning("没有找到有效的工作时间段")
                    
                except Exception as e:
                    st.error(f"处理文件 {file.name} 时出错: {str(e)}")
            
            # 显示所有文件的时间段合并表格
            if all_time_periods:
                st.subheader("所有文件时间段汇总")
                all_periods_df = pd.DataFrame(all_time_periods)
                all_periods_df = all_periods_df.sort_values('开始时间')
                st.dataframe(all_periods_df)
            
            # 显示汇总结果表格
            if results:
                st.subheader("文件汇总信息")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
            else:
                st.warning("没有生成任何结果") 