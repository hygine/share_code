import streamlit as st
from .database import verify_user, update_last_active, log_action

def login_page():
    st.subheader("用户登录")
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submit = st.form_submit_button("登录")
        
        if submit:
            role = verify_user(username, password)
            if role:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['role'] = role
                update_last_active(username)
                log_action(username, "登录", "用户成功登录系统")
                st.success(f"欢迎回来, {username}!")
                st.rerun()
            else:
                st.error("用户名或密码错误")

def logout():
    if st.sidebar.button("退出登录"):
        username = st.session_state.get('username')
        if username:
            log_action(username, "登出", "用户主动退出登录")
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['role'] = None
        st.rerun()

def check_auth():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if not st.session_state['logged_in']:
        login_page()
        return False
    
    # 每次请求更新最后活动时间
    update_last_active(st.session_state['username'])
    return True
