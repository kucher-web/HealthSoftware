from datetime import datetime, time, timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
import Functions as f

def main():
    st.title("俯仰角方位角计算")

    # 单一文件上传组件
    uploaded_data = st.file_uploader("请上传遥测量数据文件", type=['xlsx', 'xls','csv'], key='data_angle')

    if uploaded_data is not None:
        # 根据文件类型读取数据
        file_extension = uploaded_data.name.split('.')[-1].lower()
        if file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_data)
        elif file_extension == 'csv':
            df = pd.read_csv(uploaded_data)
        else:
            st.error("不支持的文件格式。请上传 .xlsx, .xls 或 .csv 文件。")
            return

        # 筛选数据
        df = f.filter_data_angle(df)

        # 将'星上时间'列转换为datetime类型
        df['星上时间'] = pd.to_datetime(df['星上时间'])

        # 获取星上时间列表
        time_list = df['星上时间'].tolist()
        total_points = len(time_list)
        
        st.subheader("选择数据范围")
        
         # 使用滑块选择数据范围
        selected_range = st.slider(
            "选择数据范围",
            min_value=0,
            max_value=total_points - 1,
            value=(0, total_points - 1)
        )

        # 计算选定的开始和结束时间
        start_datetime = time_list[selected_range[0]]
        end_datetime = time_list[selected_range[1]]

        # 显示选定的时间范围
        st.write(f"选定的开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"选定的结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 筛选时间范围内的数据
        mask = (df['星上时间'] >= start_datetime) & (df['星上时间'] <= end_datetime)
        df_filtered = df.loc[mask]

        # 展示数据
        st.subheader("数据预览")
        st.write(df_filtered)

        # 选择 QV1 或 QV2
        qv_option = st.selectbox("选择 QV1 或 QV2", ["QV1", "QV2"])

        if st.button("计算俯仰角和方位角"):
            # 这里添加俯仰角和方位角的计算逻辑
            # 假设我们有一个函数来计算俯仰角和方位角
            df_filtered['俯仰角'], df_filtered['方位角'] = f.calculate_angles(df_filtered, qv_option)

            # 创建图表
            fig = px.scatter(df_filtered, x='星上时间', y=['俯仰角', '方位角'], 
                             title=f"{qv_option} 俯仰角和方位角")
            st.plotly_chart(fig)

            # 显示计算结果表格
            st.subheader("计算结果")
            st.dataframe(df_filtered[['星上时间', '俯仰角', '方位角']])

    else:
        st.info("请上传遥测量数据文件")

if __name__ == "__main__":
		main()