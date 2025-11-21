import os
#import yaml
import streamlit as st
from streamlit_option_menu import option_menu

USE_CASES_FOLDER = "../framework/library/use_cases"
use_cases_list = os.listdir(USE_CASES_FOLDER)
    
def step_1(current_step: str = "Data preparation", selected_aspect: str = "Baseline", aipc_config_path: str = "aipc.yaml"):
    st.title("📊 Stage 1: Data preparation")
    option_menu(
            "Operations",
            ["Data profiling", "Data validation", "Data preprocessing", "Data documentation"],
            menu_icon="sliders",
            default_index=0,
            orientation="horizontal"
        )
    #with open(aipc_config_path, "r") as yaml_file:
    #    aipc_configs = yaml.safe_load(yaml_file)
    
    with st.expander("Operation name 1", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Documentation", "Code", "Metadata"])
        with tab1:
            st.write("This is the Documentation tab")
            code = st.text_area("Documentation", key="doc_1")
        with tab2:
            st.write("This is the Code tab")
            code = st.text_area("Code implementation", key="code_1")
        with tab3:
            st.write("This is the Metadata tab")
            metadata = st.text_area("Metadata information", key="meta_1")
            
        if st.button("Run operation", key="run_op_1"):
            st.write("Running operation procedure...")
            
    with st.expander("Operation name 2"):
        tab1, tab2, tab3 = st.tabs(["Documentation", "Code", "Metadata"])
        with tab1:
            st.write("This is the Documentation tab")
            code = st.text_area("Documentation", key="doc_2")
        with tab2:
            st.write("This is the Code tab")
            code = st.text_area("Code implementation", key="code_2")
        with tab3:
            st.write("This is the Metadata tab")
            metadata = st.text_area("Metadata information", key="meta_2")
            
        if st.button("Run operation", key="run_op_2"):
            st.write("Running operation procedure...")
    
        
    
def step_2(current_step: str = "Modeling", selected_aspect: str = "Baseline", aipc_config_path: str = "aipc.yaml"):
    st.title("🛠️ Stage 2: Modeling")
    option_menu(
            "Operations",
            ["Feature Engineering", "Model training", "Model evaluation", "Model validation", "Model and product documentation"],
            menu_icon="sliders",
            default_index=0,
            orientation="horizontal"
    )
    with st.expander("Operation name 1", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Documentation", "Code", "Metadata"])
        with tab1:
            st.write("This is the Documentation tab")
            code = st.text_area("Documentation")
        with tab2:
            st.write("This is the Code tab")
            code = st.text_area("Code implementation")
        with tab3:
            st.write("This is the Metadata tab")
            schema = st.text_input("Schema")
            column_labels = ["Accuracy", "Precision", "Recall", "Latency"]
            cols = st.columns(len(column_labels))
            for i, col in enumerate(cols):
                with col:
                    st.metric(label=column_labels[i], value=f"{80 + i}%")
                    st.button(f"Show {column_labels[i]}", key=f"{i}")

def step_3(current_step: str = "Operationalization", selected_aspect: str = "Baseline", aipc_config_path: str = "aipc.yaml"):
    st.title("⚙️ Stage 3: Operationalization")    
    option_menu(
            "Operations",
            ["Model deployment", "Model monitoring", "Production data monitoring", 
             "System monitoring", "Pre-inference transformations", "Post-inference transformations"],
            menu_icon="sliders",
            default_index=0,
            orientation="horizontal"
        )  
    with st.expander("Operation name 1", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Documentation", "Code", "Metadata"])
        with tab1:
            st.write("This is the Documentation tab")
            code = st.text_area("Documentation")
        with tab2:
            st.write("This is the Code tab")
            code = st.text_area("Code implementation")
        with tab3:
            st.write("This is the Metadata tab")  
    #csv_file = st.file_uploader("Choose ")


def main():      
    st.set_page_config(layout="wide")
    st.sidebar.title("AI product development lifecycle")   
    if st.sidebar.button("Start new AI product", key="new_ai_product"):
        pass #st.sidebar.write("Creating new AI product project ...") 
    current_product = st.sidebar.selectbox("AI Products list", use_cases_list)    
    st.write(os.path.join(USE_CASES_FOLDER, current_product, "src", "data_preparation.py")) 
         
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
    if current_step == "Data preparation":
        step_1(current_step, selected_aspect)     
    
    elif current_step == "Modeling":
        step_2(current_step, selected_aspect)
        
    elif current_step == "Operationalisation":
        step_3(current_step, selected_aspect)
    
    

if __name__ == "__main__":
    main()