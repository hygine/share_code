# modules/diff_manager.py
import streamlit as st


def diff_manager_block():
    st.subheader("🗂 差异分析结果管理")

    diff_keys = sorted(
        k.replace("diff_result_", "")
        for k in st.session_state.keys()
        if k.startswith("diff_result_")
    )

    if not diff_keys:
        st.info("当前没有缓存的差异分析结果")
        return

    selected = st.selectbox("选择差异分析结果", diff_keys)

    col1, col2, col3 = st.columns(3)

    # 重命名
    with col1:
        new_name = st.text_input("重命名该结果", value=selected)
        if st.button("✏️ 重命名"):
            if new_name != selected:
                st.session_state[f"diff_result_{new_name}"] = st.session_state.pop(
                    f"diff_result_{selected}"
                )
                st.session_state[f"sig_genes_{new_name}"] = st.session_state.pop(
                    f"sig_genes_{selected}"
                )
                st.success("重命名完成，请重新选择")
                st.experimental_rerun()

    # 删除
    with col2:
        if st.button("🗑 删除该结果"):
            st.session_state.pop(f"diff_result_{selected}", None)
            st.session_state.pop(f"sig_genes_{selected}", None)
            st.success("已删除")
            st.experimental_rerun()

    # 清空
    with col3:
        if st.button("🔥 清空所有差异结果"):
            for k in list(st.session_state.keys()):
                if k.startswith("diff_result_") or k.startswith("sig_genes_"):
                    del st.session_state[k]
            st.success("全部清空")
            st.experimental_rerun()
