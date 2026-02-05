import streamlit as st
from auth.auth_utils import check_auth, logout
from auth.admin_panel import admin_panel_block

# ======================
# 页面设置 (必须在最前面)
# ======================
st.set_page_config(
    page_title="恩泽康泰-生信分析原型机",
    layout="wide"
)

# ======================
# 认证检查
# ======================
if check_auth():
    # 侧边栏显示用户信息和退出按钮
    st.sidebar.write(f"当前用户: **{st.session_state['username']}** ({st.session_state['role']})")
    logout()
    st.sidebar.divider()

    try:
        # ======================
        # 原有模块导入
        # ======================
        from modules.data_loader import load_expression, load_annotation
        from modules.gene_search import gene_search_block
        from modules.heatmap import heatmap_block
        from modules.pca import pca_block
        from modules.basic_plots import barplot_block, lineplot_block, violin_block

        # ======================
        # 新增模块导入
        # ======================
        from modules.diff_analysis import diff_block
        from modules.correlation import correlation_block
        from modules.plsda import plsda_block
        from modules.time_series import kmeans_time_block
        from modules.venn_plot import venn_block
        from modules.network import network_block
        from modules.diff_manager import diff_manager_block
        from modules.diff_report import diff_report_block

        st.title("🧬 恩泽康泰交互式生信数据可视化平台")

        # ======================
        # 菜单构建
        # ======================
        menu_options = [
            "🔥表达量热图",
            "🔭PCA分析",
            "📊常规柱状图",
            "🎻 Violin 图",
            "📈动态折线图",
            "🧪差异分析",
            "📐相关性分析",
            "🧭PLS-DA",
            "⏱序列分析",
            "🕸网络互作"
        ]
        
        # 如果是管理员，增加后台管理选项
        if st.session_state['role'] == 'admin':
            menu_options.append("🛠 用户管理后台")

        analysis_type = st.sidebar.selectbox(
            "选择功能模块",
            menu_options
        )
        
        # 记录模块切换日志
        if 'last_analysis_type' not in st.session_state or st.session_state['last_analysis_type'] != analysis_type:
            from auth.database import log_action
            log_action(st.session_state['username'], "切换模块", f"进入模块: {analysis_type}")
            st.session_state['last_analysis_type'] = analysis_type

        # ======================
        # 管理员后台逻辑
        # ======================
        if analysis_type == "🛠 用户管理后台":
            admin_panel_block()
        else:
            # ======================
            # 数据加载与生信分析逻辑
            # ======================
            df = load_expression()
            annotation_col = load_annotation()

            if df is not None:
                # 基因搜索
                df_show = gene_search_block(df)

                if analysis_type == "🔥表达量热图":
                    heatmap_block(df, df_show, annotation_col)

                elif analysis_type == "🔭PCA分析":
                    pca_block(df, annotation_col)

                elif analysis_type == "📊常规柱状图":
                    barplot_block(df, annotation_col)

                elif analysis_type == "🎻 Violin 图":
                    violin_block(df, annotation_col)

                elif analysis_type == "📈动态折线图":
                    lineplot_block(df, annotation_col)

                elif analysis_type == "🧪差异分析":
                    diff_block(df, df_show, annotation_col)

                elif analysis_type == "📐相关性分析":
                    correlation_block(df, df_show, annotation_col)

                elif analysis_type == "🧭PLS-DA":
                    plsda_block(df, annotation_col)

                elif analysis_type == "⏱序列分析":
                    kmeans_time_block(df)

                elif analysis_type == "🕸网络互作":
                    st.subheader("🕸 网络互作分析")
                    st.info("需要基因互作 edge 表（gene1, gene2）")
                    edge_file = st.file_uploader("上传网络文件", type=["csv"])
                    if edge_file:
                        import pandas as pd
                        edge_df = pd.read_csv(edge_file)
                        network_block(edge_df)
            else:
                st.info("请先上传表达矩阵以开始分析")

    except Exception as e:
        st.error("🚨 系统发生未预期错误，请检查输入数据或联系管理员")
        st.exception(e)
else:
    st.info("请登录以访问生信分析平台")
