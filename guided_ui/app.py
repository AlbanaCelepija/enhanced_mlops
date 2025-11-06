import os
import streamlit as st
from streamlit_option_menu import option_menu

def step_1(selected_aspect: str = "Baseline"):
    st.title("📊 Stage 1: Data preparation")
    option_menu(
            "Operations",
            ["Data profiling", "Data validation", "Data preprocessing", "Data documentation"],
            menu_icon="sliders",
            default_index=0,
            orientation="horizontal"
        )
    with st.expander("Operation name 1"):
        tab1, tab2, tab3 = st.tabs(["Documentation", "Code", "Metadata"])
        with tab1:
            st.write("This is the Documentation tab")
            code = st.text_area("Documentation")
        with tab2:
            st.write("This is the Code tab")
            code = st.text_area("Code implementation")
        with tab3:
            st.write("This is the Metadata tab")
    with st.expander("Operation name 2"):
        tab1, tab2, tab3 = st.tabs(["Documentation", "Code", "Metadata"])
        with tab1:
            st.write("This is the Documentation tab")
            doc = st.text_input("Doc")
        with tab2:
            st.write("This is the Code tab")
        with tab3:
            st.write("This is the Metadata tab")
        
    
def step_2(selected_aspect: str = "Baseline"):
    st.title("Stage 2: Modeling")
    rows = ["Model A", "Model B"]
    column_labels = ["Accuracy", "Precision", "Recall", "Latency"]

    for row in rows:
        with st.expander(f"{row} Details"):
            cols = st.columns(len(column_labels))
            for i, col in enumerate(cols):
                with col:
                    st.metric(label=column_labels[i], value=f"{80 + i}%")
                    st.button(f"Show {column_labels[i]}", key=f"{row}_{i}")
    schema = st.text_input("Schema")
    table = st.text_input("Table Name")
    
    return schema, table

def step_3(selected_aspect: str = "Baseline"):
    st.title("⚙️ Stage 3: Operationalization")    
    option_menu(
            "AI lifecycle stages",
            ["Data preparation", "Modeling", "Operationalisation"],
            icons=["graph-up", "tools", "speedometer2"],
            menu_icon="sliders",
            default_index=0,
            orientation="horizontal"
        )
    
    #csv_file = st.file_uploader("Choose ")


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("AI product development lifecycle")
    filenames = os.listdir("../framework/use_cases")
    current_product = st.sidebar.selectbox("AI Products list", filenames)    
    with st.sidebar:
        current_step = option_menu(
            "AI lifecycle stages",
            ["Data preparation", "Modeling", "Operationalisation"],
            icons=["graph-up", "tools", "speedometer2"],
            menu_icon="sliders",
            default_index=0,
            orientation="vertical"
        )
        selected_aspect = option_menu(
            "AI system requirements",
            ["Baseline", "Fairness", "Robustness"],
            icons=["house", "list-task", "gear"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical"
        )
    if selected_aspect == "Baseline":
        st.write("📊 Overview content")
    elif selected_aspect == "Fairness":
        st.write("📄 Details content")
    elif selected_aspect == "Robustness":
        st.write("⚙️ Settings content")
        
    if current_step == "Data preparation":
        step_1(selected_aspect)     
    
    elif current_step == "Modeling":
        step_2(selected_aspect)
        
    elif current_step == "Operationalisation":
        step_3(selected_aspect)
    
    

if __name__ == "__main__":
    main()