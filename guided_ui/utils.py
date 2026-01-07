import os
import yaml
import streamlit as st

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/library/config/pipeline_definitions.yaml"
)


def session_state_params(current_product, current_framework):
    if "current_product" not in st.session_state:
        st.session_state.current_product = current_product
    if "current_framework" not in st.session_state:
        st.session_state.current_framework = current_framework


def get_pipeline_operations():
    with open(pipeline_definitions_folder, "r") as yaml_file:
        pipeline_configs = yaml.safe_load(yaml_file)
    pipeline_ai_operations = pipeline_configs["ai_operations"]
    pipeline_data_operations = [
        elem["operations"]
        for elem in pipeline_ai_operations
        if elem["stage"] == "data_preparation"
    ]
    pipeline_model_operations = [
        elem["operations"]
        for elem in pipeline_ai_operations
        if elem["stage"] == "modelling"
    ]
    pipeline_operationalisation_ops = [
        elem["operations"]
        for elem in pipeline_ai_operations
        if elem["stage"] == "operationalization"
    ]
    all_pipeline_operations = (
        pipeline_data_operations[0]
        + pipeline_model_operations[0]
        + pipeline_operationalisation_ops[0]
    )
    return all_pipeline_operations


def populate_stages(pipeline_configs, createview=False):
    colors = ["#dbeafe", "#bfdbfe", "#93c5fd"]
    cols = st.columns(3)
    for i, stage in enumerate(pipeline_configs["ai_operations"]):
        with cols[i]:
            for j, operation in enumerate(stage["operations"]):
                op_type = list(operation.keys())[0]
                checkbox = ""
                if createview:
                    checkbox = f"""<input type="checkbox" id="{op_type}" name="{op_type}" value="{op_type}" checked enabled>"""
                st.markdown(
                    f"""
                    <div style="
                        margin-left:{j*40}px;
                        height:50px;
                        width:{450 - j*40}px;
                        background-color:#dbeafe;
                        border-radius:10px;
                        padding:10px;
                    ">
                        {checkbox}{op_type.upper().replace("_", " ")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
