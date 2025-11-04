import streamlit as st

def step_1():
    st.title("Step 1: Credentials")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    account = st.text_input("Account")
    warehouse = st.text_input("Warehouse")
    
    return username, password, account, warehouse

def step_2():
    st.title("Step 2: Table Details")
    schema = st.text_input("Schema")
    table = st.text_input("Table Name")
    
    return schema, table

def step_3():
    st.title("Step 3: Upload CSV")
    csv_file = st.file_uploader("Choose a CSV file")
    
def reset():
    username, password, account, warehouse = "", "", "", ""
    schema, table = "", ""
    csv_file = None

def main():
    st.sidebar.title("Form Wizard")
    current_step = st.sidebar.selectbox("Step", ["Step 1", "Step 2", "Step 3"])
    
    
    if current_step == "Step 1":
        username, password, account, warehouse = step_1()        
    
    elif current_step == "Step 2":
        schema, table = step_2()
        
    
    elif current_step == "Step 3":
        csv_file = step_3()
        next_step_button = st.sidebar.button("Submit")
    
    
    if st.sidebar.button("Reset", key="reset"):
        username, password, account, warehouse = "", "", "", ""
        schema, table = "", ""
        csv_file = None

if __name__ == "__main__":
    main()