import os
import sys
import yaml
import inspect
import streamlit as st
import importlib.util
from streamlit_option_menu import option_menu
from streamlit_elements import mui, elements
from library.src.artifact_types import Data, Configuration, Report

from dashboard import show_dashboard

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
pipeline_definitions_folder = os.path.join(
    parent_folder, "framework/library/config/pipeline_definitions.yaml"
)
USE_CASES_FOLDER = os.path.join(parent_folder, "framework/library/use_cases")
use_cases_list = os.listdir(USE_CASES_FOLDER)
platform_list = ["local", "dh"]


def import_from_path(module_name, file_path):
    """Import a module given its name and file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def stages_settings(stage):
    if stage == "Data preparation":
        st.title("📊 Stage 1: Data preparation")
        selected_operation = option_menu(
            "Operations",
            [
                "Data profiling",
                "Data validation",
                "Data preprocessing",
                "Data documentation",
            ],
            menu_icon="sliders",
            default_index=0,
            orientation="horizontal",
        )
        selected_operation = selected_operation.replace(" ", "_").lower()
        step_operations_module = "data_preparation.py"
        return selected_operation, step_operations_module
    elif stage == "Modeling":
        st.title("🛠️ Stage 2: Modeling")
        selected_operation = option_menu(
            "Operations",
            [
                "Feature engineering",
                "Model training",
                "Model evaluation",
                "Model validation",
                "Model documentation",
            ],
            menu_icon="sliders",
            default_index=0,
            orientation="horizontal",
        )
        selected_operation = selected_operation.replace(" ", "_").lower()
        step_operations_module = "modelling.py"
        return selected_operation, step_operations_module
    elif stage == "Operationalisation":
        st.title("⚙️ Stage 3: Operationalization")
        selected_operation = option_menu(
            "Operations",
            [
                "Model deployment",
                "Model monitoring",
                "Production data monitoring",
                "System monitoring",
                "Pre inference transformations",
                "Post inference transformations",
            ],
            menu_icon="sliders",
            default_index=0,
            orientation="horizontal",
        )
        selected_operation = selected_operation.replace(" ", "_").lower()
        step_operations_module = "operationalization.py"
        return selected_operation, step_operations_module


def show_stage(
    current_step: str = "Data preparation",
    selected_aspect: str = "Baseline",
    current_product: str = "tabular",
    current_framework: str = "local",
):
    selected_operation, step_operations_module = stages_settings(current_step)
    product_config_file = os.path.join(
        USE_CASES_FOLDER, current_product, "metadata", f"aipc_{current_framework}.yaml"
    )
    with open(product_config_file, "r") as yaml_file:
        aipc_configs = yaml.safe_load(yaml_file)
        # operations
        operations = aipc_configs["operations"]
        data_operations = [
            elem
            for elem in operations
            if elem["stage"] == "data_preparation"
            and elem["requirement_dimension"] == selected_aspect.lower()
        ]
        model_operations = [
            elem
            for elem in operations
            if elem["stage"] == "modelling"
            and elem["requirement_dimension"] == selected_aspect.lower()
        ]
        operationalisation_ops = [
            elem
            for elem in operations
            if elem["stage"] == "operationalization"
            and elem["requirement_dimension"] == selected_aspect.lower()
        ]
        # artifacts
        artifacts = aipc_configs["artifacts"]
        data_artifacts = artifacts["data"]
        model_artifacts = artifacts["model"]
        configuration_artifacts = artifacts["configuration"]

        if current_step == "Data preparation":
            populate_frames(
                data_operations,
                selected_operation,
                step_operations_module,
                current_product,
                current_framework,
                model_artifacts,
                data_artifacts,
                configuration_artifacts,
            )
        elif current_step == "Modeling":
            populate_frames(
                model_operations,
                selected_operation,
                step_operations_module,
                current_product,
                current_framework,
                model_artifacts,
                data_artifacts,
                configuration_artifacts,
            )
        elif current_step == "Operationalisation":
            populate_frames(
                operationalisation_ops,
                selected_operation,
                step_operations_module,
                current_product,
                current_framework,
                model_artifacts,
                data_artifacts,
                configuration_artifacts,
            )
        # visualize data artifacts
        populate_data_artifacts(data_artifacts)


def populate_frames(
    operations,
    selected_data_operation,
    step_operations_module,
    current_product,
    current_framework,
    model_artifacts,
    data_artifacts,
    configuration_artifacts,
):
    st.write(selected_data_operation)
    for ind, operation in enumerate(operations):
        if operation["type"] == selected_data_operation:
            operation_id = operation["id"]
            operation_name = operation["name"]
            specs = operation["implementation"]["spec"]
            method_name = specs["method_name"]
            inputs = specs["inputs"]
            outputs = specs["outputs"]

            with st.expander(
                f"Operation {operation_id}: {operation_name}", expanded=True
            ):
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    ["Documentation", "Code", "Metadata", "Input", "Output"]
                )
                with tab1:
                    st.write("This is the Documentation tab")
                    code = st.text_area(
                        "Documentation", key=f"doc_{ind}", value=operation["name"]
                    )
                with tab2:
                    st.write("This is the Code tab")
                    method_content = load_method_content(
                        method_name,
                        current_product,
                        current_framework,
                        step_operations_module,
                    )
                    code = st.text_area(
                        "Code implementation", key=f"code_{ind}", value=method_content
                    )
                with tab3:
                    st.write("This is the Metadata tab")
                    metadata = st.text_area(
                        "Metadata information", key=f"meta_{ind}", value=specs
                    )
                with tab4:
                    st.write("This is the Input tab")
                    inputs = st.text_area("Inputs", key=f"input_{ind}", value=inputs)
                with tab5:
                    st.write("This is the Output tab")
                    outputs = st.text_area("Inputs", key=f"output_{ind}", value=outputs)

                if st.button("Run operation", key=f"run_op_{ind}"):
                    run_data_operation(
                        operation,
                        data_artifacts,
                        model_artifacts,
                        configuration_artifacts,
                        current_product,
                        current_framework,
                    )


def populate_data_artifacts(data_artifacts):
    with st.container():
        for artifact in data_artifacts:
            st.write(artifact)


def load_method_content(
    method_name,
    current_product,
    current_framework="local",
    step_operations_module="data_preparation.py",
):
    product_operations_file = os.path.join(
        USE_CASES_FOLDER,
        current_product,
        "src",
        f"{current_framework}_platform",
        step_operations_module,
    )
    curr_module = import_from_path("curr_module", product_operations_file)
    func = getattr(curr_module, method_name)
    source_text = inspect.getsource(func)
    return source_text


def run_data_operation(
    operation,
    data_artifacts,
    model_artifacts,
    config_artifacts,
    current_product,
    current_framework="local",
):
    product_operations_file = os.path.join(
        USE_CASES_FOLDER,
        current_product,
        "src",
        f"{current_framework}_platform",
        "data_preparation.py",
    )
    curr_module = import_from_path("curr_module", product_operations_file)

    specs = operation["implementation"]["spec"]
    method_name = specs["method_name"]
    inputs = specs["inputs"]
    outputs = specs["outputs"]
    input_vars = {}
    for my_input in inputs:
        input_name = list(my_input.values())
        input_artifact = [art for art in data_artifacts if art["name"] == input_name[0]]
        if len(input_artifact) > 0:
            artifact_vars = {
                var_name: var_value
                for var_name, var_value in input_artifact[0].items()
                if var_name != "name"
            }
            input_vars.update({"data": Data(**artifact_vars)})
        input_artifact = [
            art for art in config_artifacts if art["name"] == input_name[0]
        ]
        if len(input_artifact) > 0:
            artifact_vars = {
                var_name: var_value
                for var_name, var_value in input_artifact[0].items()
                if var_name != "name"
            }
            input_vars.update({"config": Configuration(**artifact_vars)})
    print(input_vars)

    func = getattr(curr_module, method_name)
    result = func(**input_vars)
    # method_name = globals()[method_name]
    # result = method_name(**input_vars)


def session_state_test():
    st.title("Counter Example")
    if "count" not in st.session_state:
        st.session_state.count = 0
    increment = st.button("Increment")
    if increment:
        st.session_state.count += 1
    st.write("Count = ", st.session_state.count)


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("AI product development lifecycle")
    if st.sidebar.button("Start new AI product", key="new_ai_product"):
        pass  # st.sidebar.write("Creating new AI product project ...")
    current_framework = st.sidebar.selectbox("Tool governance platform", platform_list)
    current_product = st.sidebar.selectbox("AI Products list", use_cases_list)
    # session_state_test()

    with st.sidebar:
        current_step = option_menu(
            "AI lifecycle stages",
            ["Data preparation", "Modeling", "Operationalisation"],
            icons=["graph-up", "tools", "speedometer2"],
            menu_icon="sliders",
            default_index=0,
            orientation="vertical",
        )
        selected_aspect = option_menu(
            "AI system requirements",
            ["Baseline", "Fairness", "Robustness"],
            icons=["house", "list-task", "gear"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
        )
    if st.button("Summary Dashboard", key="dashboard"):
        # settings = st.Page("dashboard.py", title="Summary Dashboard")
        show_dashboard(current_product, current_framework)
    else:
        show_stage(current_step, selected_aspect, current_product, current_framework)


if __name__ == "__main__":
    main()
