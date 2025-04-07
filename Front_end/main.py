import streamlit as st
from streamlit_option_menu import option_menu
import LookupData
import AngleCalculation
import PowerOnOffStatistics

def main():
    
    with st.sidebar:
        selected = option_menu(
            menu_title="主菜单",
            options=["首页", "数据分析", "俯仰角方位角计算", "载管开关机时间统计", "系统设置", "关于"],
            icons=["house", "graph-up", "calculator", "power", "gear", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "首页":
        st.title("欢迎使用卫星遥测量监控系统")
        st.write("这是一个用于监控和分析卫星遥测数据的综合平台。")
        st.write("请使用左侧菜单导航到不同的功能页面。")

        st.subheader("快速入口")
        if st.button("进入数据分析"):
            selected = "数据分析"
        if st.button("进入俯仰角方位角计算"):
            selected = "俯仰角方位角计算"
        if st.button("进入载管开关机时间统计"):
            selected = "载管开关机时间统计"

    if selected == "数据分析":
        LookupData.main()

    if selected == "俯仰角方位角计算":
        AngleCalculation.main()
        
    if selected == "载管开关机时间统计":
        PowerOnOffStatistics.main()
        
    if selected == "系统设置":
        st.title("系统设置")
        st.write("这里是系统设置页面，您可以在这里调整各种参数和选项。")

    if selected == "关于":
        st.title("关于本系统")
        st.write("卫星遥测量监控系统 v0.1")
        st.write("开发团队：KLC")
        st.write("联系方式：kulinchen@microsate.com")


if __name__ == "__main__":
    main()
