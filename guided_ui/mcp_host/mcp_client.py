import os
import opik
import asyncio
from typing import TypedDict, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import StateGraph, END

from langchain_ollama import ChatOllama
from opik.integrations.langchain import OpikTracer


os.environ["OPIK_PROJECT_NAME"] = "ai_engineers_agents_project"
opik.configure(use_local=True)


def get_chat_model():
    # Using Ollama
    model = (
        ChatOllama(
            model="llama3.2:1b",
            temperature=0.7,
        )
        .bind_tools(tools)
        .with_config({"callbacks": [opik_tracer]})
    )
    return model


class GraphState(TypedDict):
    messages: List[BaseMessage]


def agent_node(state: GraphState):
    model = get_chat_model()
    response = model.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


def should_continue(state: GraphState):
    last_msg = state["messages"][-1]
    print(state)
    return "tools" if last_msg.tool_calls else "end"

async def chat_loop(tools):
    """Run an interactive chat loop"""
    print("\nMCP Chatbot Started!")
    print("Type your queries or 'quit' to exit.")
    
    while True:
        try:
            query = input("\nQuery: ").strip()
    
            if query.lower() == 'quit':
                break
                
            await process_query(query, tools)
            print("\n")
                
        except Exception as e:
            print(f"\nError: {str(e)}")

async def process_query(query: str, tools):
    ############################################################ alternative 2:
    model = ChatOllama(
        model="llama3.2:1b",
        temperature=0.7,
    ).bind_tools(tools)

    def agent_node(state: GraphState):
        response = model.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    tool_node = ToolNode(tools)

    builder = StateGraph(GraphState)
    # Define the nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    # Define the flow
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")
    graph = builder.compile()

    opik_tracer = OpikTracer(
        graph=graph.get_graph(xray=True), tags=["langchain", "ollama", "mcp"]
    )
    config = {
        "callbacks": [opik_tracer],
    }

    result = await graph.ainvoke(
        input={
            "messages": [
                HumanMessage(
                    content=query
                )
            ]
        },
        config=config,
    )
    print(result["messages"])
async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["../mcp_servers/mcp_server_string_tools.py"],  # Update this path
    )
    # launch the server as a subprocess
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # initialize the communication between the client and the server
            await session.initialize()
            session_tools = await session.list_tools()
            print(session_tools)
            tools = await load_mcp_tools(session)
            await chat_loop(tools)

            ############################################################# alternative 1:
            """"
            agent = create_react_agent(model, tools)          
            # Try out the tools via natural language
            msg1 = {"messages": "Reverse the string 'hello world'"}
            msg2 = {
                "messages": "How many words are in the sentence 'Model Context Protocol is powerful'?"
            }

            # TODO: apply query rewriting and save the optimised query

            res1 = await agent.ainvoke(msg1)
            # print("Reversed string result:", res1)
            for m in res1["messages"]:
                m.pretty_print()
            res2 = await agent.ainvoke(msg2)
            # print("Word count result:", res2)
            for m in res2["messages"]:
                m.pretty_print()
            """
            



if __name__ == "__main__":
    asyncio.run(main())
