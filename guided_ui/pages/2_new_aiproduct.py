# streamlit: page_name = "New AI Product"
import os
import yaml
import asyncio
import pandas as pd
import streamlit as st
from utils import populate_stages
from dotenv import load_dotenv, find_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

_ = load_dotenv(find_dotenv())

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
    return ai_prod_name, ai_prod_desc
    

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
async def run_mcp_query(user_input):   
    # Model
    model = ChatOpenAI(model="gpt-4o-mini") #"gpt-5"

    # MCP Client via HTTP
    client = MultiServerMCPClient(
        {
            "mlops_tai_engineers": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8000/mcp"  
            },
            "file_system": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8080/mcp"  
            }
        }
    )
    tools = await client.get_tools()
    print(tools)
    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools)

    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    async def call_model(state: MessagesState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # LangGraph pipeline
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue)
    builder.add_edge("tools", "call_model")

    graph = builder.compile()
    result = await graph.ainvoke({"messages": [{"role": "user", "content": user_input}]})

    # Extract last message text
    last_msg = result["messages"][-1].content
    return last_msg if isinstance(last_msg, str) else str(last_msg)



# Step 5: generate the new AI product folder structure with the necessary code files and configuration files
def generate_prod_action(ai_prod_desc):
    #I need to generate code for data drift detection, put the code snippet generated inside the file data_drift.py inside the specified folder

    if st.button("Create AI product skeleton", key=f"generate_product"):        
        new_prod_folder = os.path.join(parent_folder, "framework/library/use_cases/new_prod")
        folder = f"FOLDER: {new_prod_folder}"
        os.makedirs(new_prod_folder, exist_ok=True)
        with st.spinner("Thinking..."):
            answer = asyncio.run(run_mcp_query(ai_prod_desc + folder))
            st.success("Operation completed successfully!")
            st.success(answer)               
        
    

if __name__ == "__main__":
    ai_prod_name, ai_prod_desc = lifecycle_stages()
    show_new_prod_skeleton()
    generate_prod_action(ai_prod_desc)
    
