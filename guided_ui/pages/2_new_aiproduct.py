# streamlit: page_name = "New AI Product"
import os
import yaml
import streamlit as st
from utils import populate_stages

st.set_page_config(page_title="New AI Product")  # , page_icon="📊"
st.title("Second Page")
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
populate_stages(pipeline_configs, createview=True)


# Step 2: definition of the requirements dimensions to be satisfied according to the AI product design objectives
def show_new_prod_skeleton():
    st.session_state["page"] = "new_prod"
    checklist_requirements = st.multiselect(
        "Requirements dimensions",
        ["Baseline", "Fairness", "Robustness"],
        ["Baseline", "Robustness"],
    )


# Step 3: specification of the data artifacts and the AI product objectives (classification, clustering, information extraction)
# subtasks:
#   - load data artifact from corresponding dh project


# Step 4: CoT prompting to plan the operations of a new AI product
# subtasks:
#   - select the right open source toolkits to use for implementing each operation according to the pre-defined requirements
#   - generate code snippets based on the selected toolkits for each operation

# Step 5: generate the new AI product folder structure with the necessary code files and configuration files


def generate_new_prod():
    st.session_state["page"] = "new_prod"
    st.title("Second Page")
    # st.warning("No data found. Please go back to the main page.")


if __name__ == "__main__":
    show_new_prod_skeleton()
