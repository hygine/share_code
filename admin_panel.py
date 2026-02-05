import streamlit as st
import pandas as pd
from .database import get_all_users, add_user, delete_user, get_online_users, get_audit_logs, log_action

def admin_panel_block():
    st.header("🛠 管理员后台")
    
    tab1, tab2, tab3, tab4 = st.tabs(["用户列表", "新增用户", "在线用户", "操作记录"])
    
    with tab1:
        st.subheader("当前系统用户")
        users = get_all_users()
        if users:
            df_users = pd.DataFrame(users, columns=["ID", "用户名", "角色"])
            st.table(df_users)
            
            st.divider()
            st.subheader("删除用户")
            user_to_delete = st.selectbox("选择要删除的用户ID", [u[0] for u in users if u[1] != 'admin'])
            if st.button("确认删除"):
                # 记录删除操作
                target_username = next((u[1] for u in users if u[0] == user_to_delete), "Unknown")
                if delete_user(user_to_delete):
                    log_action(st.session_state['username'], "删除用户", f"删除了用户: {target_username}")
                    st.success(f"用户 {target_username} 已删除")
                    st.rerun()
                else:
                    st.error("删除失败")
        else:
            st.info("暂无用户数据")
            
    with tab2:
        st.subheader("创建新账号")
        with st.form("add_user_form"):
            new_username = st.text_input("用户名")
            new_password = st.text_input("密码", type="password")
            new_role = st.selectbox("角色", ["user", "admin"])
            submit = st.form_submit_button("创建用户")
            
            if submit:
                if new_username and new_password:
                    if add_user(new_username, new_password, new_role):
                        log_action(st.session_state['username'], "新增用户", f"创建了新用户: {new_username} ({new_role})")
                        st.success(f"用户 {new_username} 创建成功")
                        st.rerun()
                    else:
                        st.error("用户名已存在或创建失败")
                else:
                    st.warning("请填写完整信息")

    with tab3:
        st.subheader("实时在线用户 (最近5分钟内活跃)")
        online_users = get_online_users(minutes=5)
        if online_users:
            df_online = pd.DataFrame(online_users, columns=["用户名", "角色", "最后活动时间"])
            st.table(df_online)
        else:
            st.info("当前无在线用户")
        
        if st.button("刷新在线状态"):
            st.rerun()

    with tab4:
        st.subheader("系统操作日志")
        logs = get_audit_logs(limit=100)
        if logs:
            df_logs = pd.DataFrame(logs, columns=["用户名", "操作类型", "详情", "时间"])
            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("暂无操作记录")
        
        if st.button("刷新日志"):
            st.rerun()
