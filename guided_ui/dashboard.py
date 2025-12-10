import os
import yaml
import streamlit as st

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/library/config/pipeline_definitions.yaml"
)
USE_CASES_FOLDER = os.path.join(parent_folder, "framework/library/use_cases")


def show_dashboard(current_product, current_framework):
    st.title(
        "Dashboard: Summary of the defined AI product operations for each requirement dimension"
    )
    product_config_file = os.path.join(
        USE_CASES_FOLDER, current_product, "metadata", f"aipc_{current_framework}.yaml"
    )
    with open(product_config_file, "r") as yaml_file:
        aipc_configs = yaml.safe_load(yaml_file)
    
    with open(pipeline_definitions_folder, "r") as yaml_file:
        pipeline_configs = yaml.safe_load(yaml_file)
    requirements_dimensions = pipeline_configs["requirements_dimensions"]
    for req in requirements_dimensions:
        st.write(req.upper())
        status = []
        for stage in pipeline_configs["ai_operations"]:
            for operation in stage["operations"]:
                op_type = list(operation.keys())[0]
                passed_operations = [
                    op["id"]
                    for op in aipc_configs["operations"]
                    if op["requirement_dimension"] == req and op["type"] == op_type
                ]
                status.append(
                    {
                        "operation_type": op_type,
                        "passed_operations": passed_operations,
                    }
                )
        all_pipeline_operations = get_pipeline_operations()
        cols = st.columns(len(all_pipeline_operations))
        for col, pipeline_op in zip(cols, all_pipeline_operations):
            op_type = list(pipeline_op.keys())[0]
            declared_operations = list(
                filter(lambda x: x["operation_type"] == op_type, status)
            )[0]["passed_operations"]
            # st.write(declared_operations)

            passed = len(declared_operations) > 0
            color = "red" if passed else "green"
            col.button(
                f"{len(declared_operations)}" if passed else "✗",
                key=f"{col}_{color}",
                help=op_type,
                type="primary" if passed else "secondary",
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