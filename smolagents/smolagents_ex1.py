# Test smolagents - using Huggingface smplagents 
# Model: Huggingface free inference model HfApiModel
from datetime import datetime
import streamlit as st
from smolagents import ToolCallingAgent, CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
import logging
# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            # TO-DO: Resolve Adding any web search tool gives the error
            # ERROR:__main__:Error initiating tools: "Error: tool 'web_search' already exists in the toolbox."
            self.toolCallingAgent = ToolCallingAgent(tools=[], 
                                                     model=model,
                                                     #add_base_tools=True,
                          )

            #Use CodeAgent to invoke agents using python code

            self.codeAgent = CodeAgent(tools=[DuckDuckGoSearchTool],
                                model=model, 
                                additional_authorized_imports=['requests','math', 'bs4'],
                                #add_base_tools=True,
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

    def testTools(self):
        try:                    
            #test tools
            logger.info(self.toolCallingAgent.run("How many countries are in the continent of Asia?"))

            logger.info(self.codeAgent.run("Which is the smallest country in Asia?"))
        except Exception as e:
            logger.error(f"Error invoking tools: {str(e)}")

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
    
    query = st.text_input("Enter query:",type="default")
    if query:
        # process query and return results
        with st.spinner():
            response = st.session_state.smolagent_app.process_query(query)
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
                st.markdown(f"**User** {chat['user']}")
                st.markdown(f"**Agent** {chat['agent']}")
            with col2:
                st.caption(f"⌚ {chat['timestamp']}")
            st.markdown("-----")

if __name__=="__main__":
    main()