import os
import yaml
import pandas as pd
import streamlit as st

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/library/config/pipeline_definitions.yaml"
)
USE_CASES_FOLDER = os.path.join(parent_folder, "framework/library/use_cases")


def show_dashboard(current_product, current_framework):
    st.title(
        "Dashboard: AI Product Operations by Requirements Dimension", anchor="dashboard"
    )
    product_config_file = os.path.join(
        USE_CASES_FOLDER, current_product, "metadata", f"aipc_{current_framework}.yaml"
    )
    with open(product_config_file, "r") as yaml_file:
        aipc_configs = yaml.safe_load(yaml_file)
    
    with open(pipeline_definitions_folder, "r") as yaml_file:
        pipeline_configs = yaml.safe_load(yaml_file)
        
    requirements_dimensions = pipeline_configs["requirements_dimensions"]
    populate_stages(pipeline_configs)
    for requirement in requirements_dimensions:
        status = read_operations_status(pipeline_configs, aipc_configs, requirement)             
        show_operations_status(pipeline_configs, requirement, status)

def show_operations_status(pipeline_configs, requirement, status):
    with st.expander(f"{requirement.upper()} OPERATIONS", expanded=True):
        cols_req = st.columns(3)
        for i, stage in enumerate(pipeline_configs["ai_operations"]):
            with cols_req[i]:
                for j, operation in enumerate(stage["operations"]):
                    op_type = list(operation.keys())[0]
                    declared_operations = list(
                        filter(lambda x: x["operation_type"] == op_type, status)
                    )[0]["passed_operations"]
                    passed = len(declared_operations) > 0
                    st.markdown(
                            f"""
                            <div style="
                                margin-left:{j*40}px;
                                height:50px;
                                width:{450 - j*40}px;
                                background-color:{'#e6f4ea' if passed else '#FFE5E5'};
                                border-radius:10px;
                                padding:10px;
                            ">
                                {'; -  '.join(declared_operations) if passed else "✗"}
                            </div>
                            """,
                            unsafe_allow_html=True
                    ) 
def read_operations_status(pipeline_configs, aipc_configs, requirement):
    status = []
    for i, stage in enumerate(pipeline_configs["ai_operations"]):
        for j, operation in enumerate(stage["operations"]):
            op_type = list(operation.keys())[0]
            passed_operations = [
                op["id"]
                for op in aipc_configs["operations"]
                if op["requirement_dimension"] == requirement and op["type"] == op_type
            ]
            status.append(
                {
                    "operation_type": op_type,
                    "passed_operations": passed_operations,
                }
            )  
    return status
def populate_stages(pipeline_configs):
    colors = ["#dbeafe", "#bfdbfe", "#93c5fd"]
    cols = st.columns(3)
    for i, stage in enumerate(pipeline_configs["ai_operations"]):
        with cols[i]:
            for j, operation in enumerate(stage["operations"]):
                op_type = list(operation.keys())[0]
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
                        {op_type.upper().replace("_", " ")}
                    </div>
                    """,
                    unsafe_allow_html=True
                )   
                
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
