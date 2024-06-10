import streamlit as st
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import load_documents_and_build_index

# Verify API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("API key not found. Please check your environment variable configuration.")
else:
    st.write(f"API Key: {api_key}")
    
# Streamlit page setup
st.set_page_config(page_title="Query Interface", layout="wide")
st.title('Query Interface for Document and Data Analysis')

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Prepare data path and load data
data_path = os.path.join(base_dir, "data", "test_data.csv")
if os.path.exists(data_path):
    data_df = pd.read_csv(data_path)
else:
    st.error(f"Data file not found at {data_path}")
    st.stop()

# Set up the data query engine
data_query_engine = PandasQueryEngine(df=data_df, verbose=True, instruction_str=instruction_str)
data_query_engine.update_prompts({"pandas_prompt": new_prompt})

# Load the PDF document and create the query engine
pdf_file_path = os.path.join(base_dir, "data", "power-bi-fundamentals-compressed.pdf")
index_name = "PowerBI_Fundamentals_Index"
rebuild = True

try:
    pdf_query_engine = load_documents_and_build_index(pdf_file_path, index_name, rebuild)
except Exception as e:
    st.error(f"An error occurred while loading the PDF document: {e}")
    st.stop()

# Configure tools for the agent
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=data_query_engine,
        metadata=ToolMetadata(
            name="test_data",
            description="This gives information about blocks on chain",
        ),
    ),
    QueryEngineTool(
        query_engine=pdf_query_engine,
        metadata=ToolMetadata(
            name="doc_data",
            description="This gives information about Power BI Fundamentals Documentation",
        ),
    ),
]

# Initialize the AI model and agent with GPT-4 and longer responses
llm = OpenAI(api_key=api_key, model="gpt-4-turbo", max_tokens=4096)
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

# User input for querying
user_query = st.text_input("Enter your query (type 'q' to stop):")
if user_query:
    if user_query == 'q':
        st.write("Exiting query interface.")
    else:
        with st.spinner('Processing your query...'):
            try:
                result = agent.query(user_query)
                # Use a text area to display long responses
                st.text_area("Result:", value=result.response, height=500)
            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")
