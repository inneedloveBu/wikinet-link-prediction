# main.py - Streamlit主应用
import streamlit as st

st.set_page_config(
    page_title="WikiNet - Wikipedia Path Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 侧边栏
with st.sidebar:
    st.title("🔗 WikiNet Explorer")
    st.markdown("Predict paths between Wikipedia pages using Graph Neural Networks")
    
    # 页面搜索
    page_a = st.text_input("Start Page", "Artificial intelligence")
    page_b = st.text_input("End Page", "Machine learning")
    
    # 参数设置
    depth = st.slider("Graph Depth", 1, 5, 2)
    model_type = st.selectbox(
        "GNN Model",
        ["GCN", "GAT", "GraphSAGE"]
    )
    
    if st.button("Predict Path", type="primary"):
        # 触发预测
        pass

# 主界面布局
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Knowledge Graph Visualization")
    # 显示图可视化
    # fig = app.visualize_subgraph(center_node, depth)
    # st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("📈 Prediction Results")
    
    # 显示预测结果
    st.metric("Connection Probability", "0.87")
    
    # 显示路径
    st.subheader("Shortest Path")
    st.write("1. Artificial intelligence")
    st.write("2. Neural network")
    st.write("3. Machine learning")
    
    # 下载选项
    if st.button("Export Graph as PNG"):
        st.success("Graph exported successfully!")