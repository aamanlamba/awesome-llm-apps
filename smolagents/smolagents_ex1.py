# Test smolagents - using Huggingface smplagents 
# Model: Huggingface free inference model HfApiModel
from datetime import datetime
import streamlit as st
from smolagents import ToolCallingAgent, CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
import logging
from transformers import tool
import requests
# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create custom tool 

### Currently this fails with an
### unexpected keyword argument 'sanitize_inputs_outputs'
@tool
def check_model_exists(model_name: str) -> str:
    """
    Tool to check if specific model exists in repository

    Args:
        model_name: The model to be searched for in the local repository
    """
    return "Model exists" #dummy return
class SmolAgentApp:
    def __init__(self):
        try:
            # use Ollama and Llama3.2
            # To-do: Add drop-down list of ollama models to select 
            model = LiteLLMModel(
                model_id="ollama_chat/llama3.2",
                api_base="http://localhost:11434/api/chat",

            )
            #use ToolCalling Agent to invoke agents using JSON
            self.toolCallingAgent = ToolCallingAgent(tools=[check_model_exists], 
                                                     model=model,
                                                     add_base_tools=True,
                          )

            #Use CodeAgent to invoke agents using python code

            self.codeAgent = CodeAgent(tools=[check_model_exists],
                                model=model, 
                                additional_authorized_imports=['requests','math', 'bs4'],
                                add_base_tools=True,
                                )
        except Exception as e:
            logger.error(f"Error initiating tools: {str(e)}")
            st.error(f"Error initiating tools: {str(e)}")

    def process_query(self,query:str) -> str:
        try:
            #to-do - when should codeAgent vs toolCallingAgent be used?
            response = self.toolCallingAgent.run(query)
            return response
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            st.error(f"Error executing query: {str(e)}")
    
    def getAgentlogs(self) -> str:
        logs = self.toolCallingAgent.logs
        return logs

    def testTools(self):
        try:                    
            #test tools
            logger.info(self.toolCallingAgent.run("How many countries are in the continent of Asia?"))

            logger.info(self.codeAgent.run("Which is the smallest country in Asia?"))
        except Exception as e:
            logger.error(f"Error invoking tools: {str(e)}")

from enum import Enum 
# Define the MessageRole enum 
class MessageRole(Enum): 
    SYSTEM = 'system' 
    USER = 'user' 
    ASSISTANT = 'assistant'


def main():
    """Streamlit application"""
    st.set_page_config(
        page_title="SmolAgent Example App",
        layout="wide",
        page_icon="🌐",
    )
    st.title("SmolAgent Example App 🌐")
    #initialize session state
    if('chat_history' not in st.session_state):
        st.session_state.chat_history=[]
    if('smolagent_app' not in st.session_state):
        st.session_state.smolagent_app = SmolAgentApp()
    # check if Ollama models exist
    model_name="Llama3.1"
    modelQuery = f"Does {model_name} exist in the local ollama repository?"
    #Temporarily commenting the next lines due to tool invocation error
    #result = st.session_state.smolagent_app.process_query(modelQuery)
    #check_model_exists("your_model_name")
    #logger.info(f"{modelQuery} -> {result}")
    #st.text(result)
    query = st.text_input("Enter query:",type="default")
    if query:
        # process query and return results
        with st.spinner():
            response = st.session_state.smolagent_app.process_query(query)
            #st.caption(response)
            #read agent logs
            agentLogs = st.session_state.smolagent_app.getAgentlogs()
            logger.info(agentLogs)
            # Print questions and answers from agent logs
            #qa_dict = extract_qa(agentLogs)

            # Print in the format Question - Answer
            #for question, answer in qa_dict.items():
            logger.info(f"Question: {query}\nAnswer: {response}\n")
            st.caption(f"Question: {query}\nAnswer: {response}\n")


            #add to history
            st.session_state.chat_history.append(
                {"user":query,
                "agent":response,
                "timestamp":datetime.now().strftime("%H:%M:%S")}
            )
            st.session_state.query=""
    col1, col2 = st.columns([1,5])
    # display history
    st.subheader("Search history")
    for chat in reversed(st.session_state.chat_history):
        with st.container():
            col1, col2 = st.columns([6,1])
            with col1:
                st.markdown(f"**Question** {chat['user']}")
                st.markdown(f"**Answer** {chat['agent']}")
            with col2:
                st.caption(f"⌚ {chat['timestamp']}")
            st.markdown("-----")

if __name__=="__main__":
    main()