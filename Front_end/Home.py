import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import Functions as f

def main():

    # 设置页面标题和布局
    st.set_page_config(page_title="卫星遥测量监控系统", layout="wide")

    # 显示主标题
    st.title("卫星遥测量监控系统")

    # 创建文件上传功能
    # 模板文件上传
    uploaded_template = st.file_uploader("请上传模板文件（包含需要显示的列）", type=['xlsx', 'xls'], key='template')

    # 慢包数据文件上传
    uploaded_data_m = st.file_uploader("请上传遥测量慢包数据文件", type=['xlsx', 'xls'], key='data_m')

    # 快包数据文件上传
    uploaded_data_k = st.file_uploader("请上传遥测量快包数据文件", type=['xlsx', 'xls'], key='data_k')

    if uploaded_template is not None and uploaded_data_m is not None and uploaded_data_k is not None:
        # 读取Excel文件
        # 读取模板文件的第一个工作表获取需要显示的列
        template_df = pd.read_excel(uploaded_template,sheet_name="通信情况专项")
        # 读取慢包数据文件
        df_m = pd.read_excel(uploaded_data_m)

        # 读取快包数据文件
        df_k = pd.read_excel(uploaded_data_k)

        # 调用函数处理模板文件和快慢包数据文件
        filtered_df = f.template_based_filter(df_k,df_m,template_df)

        # 调用函数处理模板文件和快慢包数据文件
        filtered_df_k = f.template_filter(df_k,template_df)
        filtered_df_m = f.template_filter(df_m,template_df)

        # 添加显示控制开关
        show_raw_data = st.checkbox('显示源码', value=False)

        # 是否显示源码
        if show_raw_data:
            filtered_df = f.template_based_filter(df_k,df_m,template_df)
            filtered_df_k = f.template_filter(df_k,template_df)
            filtered_df_m = f.template_filter(df_m,template_df)
        else :
            filtered_df = f.DeleteYM(filtered_df)
            filtered_df_k = f.DeleteYM(filtered_df_k)
            filtered_df_m = f.DeleteYM(filtered_df_m)
            
        # 显示原始数据表格
        st.subheader("数据预览")
        st.dataframe(filtered_df)

        # 创建数据可视化部分
        st.subheader("数据可视化")
        
        # 添加列选择控件
        time_column = '星上时间'  # 统一时间列
        
        # 使用列布局并排显示
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**快包数据可视化**")
            # 固定X轴为星上时间
            if time_column not in filtered_df_k.columns:
                st.error(f"快包数据缺少必要时间列: {time_column}")
            else:
                y_columns_k = st.multiselect("选择快包遥测量(Y轴)", 
                                        options=[c for c in filtered_df_k.columns if c != time_column],
                                        default=[],
                                        key='y_cols_k')

        with col2:
            st.markdown("**慢包数据可视化**")
            # 固定X轴为星上时间
            if time_column not in filtered_df_m.columns:
                st.error(f"慢包数据缺少必要时间列: {time_column}")
            else:
                y_columns_m = st.multiselect("选择慢包遥测量(Y轴)", 
                                        options=[c for c in filtered_df_m.columns if c != time_column],
                                        default=[],
                                        key='y_cols_m')

        # 添加确认按钮（只保留一个）
        if st.button('确认更新图表'):
            with st.spinner('正在生成最新图表...'):
                # 快包数据图表
                if time_column in filtered_df_k.columns and y_columns_k:
                    st.subheader("快包数据可视化")
                    for i, y_col in enumerate(y_columns_k):
                        # 每行显示两个图表
                        cols = st.columns(2)
                        with cols[0]:
                            fig = px.line(filtered_df_k, x=time_column, y=y_col,
                                        title=f"快包 {y_col} 趋势图")
                            st.plotly_chart(fig)
                        with cols[1]:
                            fig = px.scatter(filtered_df_k, x=time_column, y=y_col,
                                        title=f"快包 {y_col} 散点图")
                            st.plotly_chart(fig)
                
                # 慢包数据图表
                if time_column in filtered_df_m.columns and y_columns_m:
                    st.subheader("慢包数据可视化")
                    for i, y_col in enumerate(y_columns_m):
                        cols = st.columns(2)
                        with cols[0]:
                            fig = px.line(filtered_df_m, x=time_column, y=y_col,
                                        title=f"慢包 {y_col} 趋势图")
                            st.plotly_chart(fig)
                        with cols[1]:
                            fig = px.scatter(filtered_df_m, x=time_column, y=y_col,
                                        title=f"慢包 {y_col} 散点图")
                            st.plotly_chart(fig)
                            
                if not y_columns_k and not y_columns_m:
                    st.warning("请至少选择一个遥测量进行可视化")
        else:
            st.info('请选择参数后点击按钮查看图表')
        
    else:
        st.info("请上传Excel文件以查看数据分析结果")

if __name__ == "__main__":
    main()