import os
import glob
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in a .env file and restart the app.")
    st.stop()

st.title("Data Explorer Agent")

# File upload
uploaded = st.file_uploader("Upload your diabetes CSV data", type="csv")
if not uploaded:
    st.info("Please upload a CSV file to get started.")
    st.stop()

# Read DataFrame
df = pd.read_csv(uploaded)
st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

# Instantiate the LLM and agent once
llm = ChatOpenAI(
    model="gpt-4-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True
)

# Define tabs and default prompts
tab_names = ["Data Quality", "Descriptive Stats", "Exploratory Plots", "Insights"]
tabs = st.tabs(tab_names)

default_prompts = {
    "Data Quality": (
        "1. Check for missing values, zeros in key columns, and duplicates.\n"
        "2. Summarize how many issues you found in each category."
    ),
    "Descriptive Stats": (
        "Compute the mean, median, and standard deviation for all numeric features, and list the top 3 features by variance."
    ),
    "Exploratory Plots": (
        "1. Generate a histogram of `Glucose` faceted by `Outcome` and capture the figure in `fig1`.\n"
        "2. Generate a boxplot of `BMI` grouped by `Outcome` and capture the figure in `fig2`.\n"
        "3. Save `fig1` to 'fig1.png' and `fig2` to 'fig2.png' using `fig.savefig()`.\n"
        "Do not call plt.show() or use Streamlit-specific display functions."
    ),
    "Insights": (
        "Provide 3 key insights about the dataset based on your analyses and saved plots."
    )
}

# Loop through each tab
for name, tab in zip(tab_names, tabs):
    with tab:
        st.header(name)
        prompt = st.text_area(
            label=f"{name} Prompt",
            value=default_prompts[name],
            key=name
        )
        if st.button(f"Run: {name}", key=f"btn_{name}"):
            with st.spinner("Running agent..."):
                try:
                    # Clear previous images
                    for f in glob.glob("fig*.png"):
                        os.remove(f)

                    # Run agent; it will save figures to disk
                    output = agent.run(prompt)
                    st.markdown("**Agent Response:**")
                    st.write(output)

                    # Display saved images
                    for img_file in sorted(glob.glob("fig*.png")):
                        st.image(img_file, caption=img_file)
                except Exception as e:
                    st.error(f"Error: {e}")

# Sidebar instructions
with st.sidebar:
    st.markdown("---")
    st.markdown("**Instructions**")
    st.markdown(
        "1. Upload your CSV dataset.\n"
        "2. Select a tab and tweak the prompt as needed.\n"
        "3. Click 'Run' to generate and display plots saved as images.\n"
        "4. Explore different questions to uncover insights!"
    )
