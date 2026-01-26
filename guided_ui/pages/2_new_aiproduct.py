# streamlit: page_name = "New AI Product"
import os
import yaml
import pandas as pd
import streamlit as st
from utils import populate_stages

st.set_page_config(layout="wide", page_title="New AI Product")  # , page_icon="📊"
st.title("Generate a new AI Product: guide through the steps")
# st.sidebar.header("DataFrame Demo")

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
parent_folder = os.path.dirname(parent_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/library/config/pipeline_definitions.yaml"
)

# Step 1: definition of the necessary operations (active/inactive ops)
with open(pipeline_definitions_folder, "r") as yaml_file:
    pipeline_configs = yaml.safe_load(yaml_file)
    requirements_dimensions = list(map(lambda x: x.capitalize(), pipeline_configs["requirements_dimensions"]))

def lifecycle_stages():
    ai_prod_name = st.text_area("AI product name")
    ai_prod_desc = st.text_area("AI product description")
    st.write("AI system's stages and operations")
    populate_stages(pipeline_configs, createview=True)
    

# Step 2: definition of the requirements dimensions to be satisfied according to the AI product design objectives
def show_new_prod_skeleton():
    st.session_state["page"] = "new_prod"
    checklist_requirements = st.multiselect(
        "AI system's requirements dimensions",
        requirements_dimensions,
        ["Baseline", "Robustness"],
    )


# Step 3: specification of the data artifacts and the AI product objectives (classification, clustering, information extraction)
# subtasks:
#   - load data artifact from corresponding dh project
#   - load data characteristics into the prompt context in order to facilitate the planning of the operations
#data_atifact = pd.read_csv("artifacts/data/data.csv") # TODO


# Step 4: CoT prompting to plan the operations of a new AI product
# subtasks:
#   - select the right open source toolkits to use for implementing each operation according to the pre-defined requirements
#   - generate code snippets based on the selected toolkits for each operation

# Step 5: generate the new AI product folder structure with the necessary code files and configuration files
def generate_prod_acton():
    if st.button("Create AI product skeleton", key=f"generate_product"):
        st.success("Operation completed successfully!")
        new_prod_folder = os.path.join(parent_folder, "framework/library/use_cases/new_prod")
        os.makedirs(new_prod_folder, exist_ok=True)
        
    

if __name__ == "__main__":
    lifecycle_stages()
    show_new_prod_skeleton()
    generate_prod_acton()
    
