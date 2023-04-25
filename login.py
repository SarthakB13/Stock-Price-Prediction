import streamlit as st
import sqlite3
import subprocess

conn = sqlite3.connect('database.db')
c = conn.cursor()

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)')

def add_user(username, password):
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()

def login(username, password):
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    return user

create_table()
def main():
    st.title("Login Page")
    menu = ['Login', 'SignUp']
    choice = st.sidebar.selectbox("Login", menu)
    if choice == 'Login':
        st.subheader("Please login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            user = login(username, password)
            if user:
                subprocess.run(["streamlit", "run", "web_app.py"])
            else:
                st.error("Invalid username or password")

    elif choice == 'SignUp':
        st.subheader("Create new account")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type='password')
        if st.button("Sign Up"):
            add_user(new_username, new_password)
            st.success("New account created!")
    
if __name__ == '__main__':
    main()
#    if webbapp == True:
#        import web_app.py
