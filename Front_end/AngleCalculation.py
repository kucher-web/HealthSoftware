import streamlit as st
import pandas as pd
import plotly.express as px
import Functions as f

def main():
		st.title("俯仰角方位角计算")

		# 文件上传组件
		uploaded_data_m = st.file_uploader("请上传遥测量慢包数据文件", type=['xlsx', 'xls'], key='data_m_angle')
		uploaded_data_k = st.file_uploader("请上传遥测量快包数据文件", type=['xlsx', 'xls'], key='data_k_angle')

		if uploaded_data_m is not None and uploaded_data_k is not None:
				# 读取Excel文件
				df_m = pd.read_excel(uploaded_data_m)
				df_k = pd.read_excel(uploaded_data_k)

				# 合并快慢包数据
				df = pd.concat([df_m, df_k], axis=0).sort_values('星上时间').reset_index(drop=True)

				# 时间范围选择
				min_date = df['星上时间'].min().date()
				max_date = df['星上时间'].max().date()
				start_date = st.date_input("选择开始日期", min_date)
				end_date = st.date_input("选择结束日期", max_date)

				# 确保选择的日期在数据范围内
				start_date = max(start_date, min_date)
				end_date = min(end_date, max_date)

				# 筛选时间范围内的数据
				mask = (df['星上时间'].dt.date >= start_date) & (df['星上时间'].dt.date <= end_date)
				df_filtered = df.loc[mask]

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
				st.info("请上传遥测量慢包和快包数据文件")

if __name__ == "__main__":
		main()