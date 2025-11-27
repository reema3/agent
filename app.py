from openai import OpenAI
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import kagglehub


st.set_page_config(
    page_title = 'Conversation AI - BI Dashboard',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

@st.cache_resource
def init_conenctions():
    "Setup OpenAI"
    llm_model="openai/gpt-oss-20b:groq"
    api_key=os.environ.get('HF_TOKEN')
    base_url="https://router.huggingface.co/v1"

    client = OpenAI(
    base_url=base_url,
    api_key=api_key)

    return client,llm_model

@st.cache_data(ttl=300) #cache for 5 min
def load_data():   
    "Load and prepare data"
    dataset_dir = kagglehub.dataset_download("rohitsahoo/sales-forecasting")
    file_path = os.path.join(dataset_dir, "train.csv")
    df = pd.read_csv(file_path)
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    return df

def get_df_info(df):
    "Get metadata about Dataframe to help OpenAI understand it"

    def convert_date_to_string(date_val):
        if pd.isna(date_val):
            return "N/A"
        elif hasattr(date_val,'strftime'):
            return date_val.strftime('%Y-%m-%d')
        else:
            return str(date_val)
        
    date_range = 'N/A'

    if 'Order Date' in df.columns:
        min_date = df['Order Date'].min()
        max_date = df['Order Date'].max()
        date_range = f"{convert_date_to_string(min_date)} to {convert_date_to_string(max_date)}"

    return {
        'columns' : df.columns.tolist(),
        'countries' : df['Country'].unique().tolist() if 'Country' in df.columns else [],
        'products' : df['Product Name'].unique().tolist() if 'Product Name' in df.columns else [],
        'categories' : df['Category'].unique().tolist() if 'Category' in df.columns else [],
        'date_range' : date_range,
        'shape' : df.shape,
        'metric' : ['Sales']

    }

def generate_pandas_code(client, llm_model,question,df_info):

    prompt = f"""
    You have a pandas a Dataframe called 'df' with these columns: {df_info['columns']}
    Countries available: {df_info['countries']}
    Products available: {df_info['products']}
    Categories available: {df_info['categories']}
    Date range: {df_info['date_range']}

    The user asks: "{question}"

    Write python code using pandas to answer this question.
    The code should:
    1. Start with result = 
    2. Use df as the Dataframe variable
    3. Return a dataframe or Series
    4. Be a single line if possible

    Examples:
    - "Total Sales" -> result = df['Sales'].sum()
    - Sales by Country -> result = df.groupby('Country')['Sales'].sum()
    - Average Sales for Furniture -> df[df['Category']=='Furniture']['Sales'].mean()

    Return only the code, no explanation. 

    """

    response = client.chat.completions.create(
        model = llm_model,
        messages = [
            {"role":"system", "content":"You are a pandas expert.Return only valid python code"},
            {"role":"user","content":prompt}
        ]
    )

    code = response.choices[0].message.content.strip()

    #Cleanup the code
    if '```python' in code:
        code = code.replace('```python','').replace('```','')
    code =  code.strip('"').strip('"').strip()
    if not code.startswith('result ='):
        code = 'result = '+ code
    code = code.strip()

    return code

def execute_pandas_code(code,df):
    try:
        namespace = {'df':df, 'pd':pd}
        exec(code,namespace)
        return namespace.get('result'), None
    except Exception as e:
        return None, str(e)


def main():
    st.title("Conversation AI - BI Dashboard")
    client, llm_model = init_conenctions()

    try:
        df =load_data()
        df_info = get_df_info(df)

        with st.sidebar:
            st.header("AI Assistant")

            user_question = st.text_area(
                "Ask me anything about your data:",
                placeholder = "e.g., What are total sales in United States?",
                height =100
            )

            if st.button("Get Answer",type = "primary"):
                if user_question:
                    with st.spinner("Thinking..."):
                        code = generate_pandas_code(client, llm_model,user_question,df_info)

                        result, error  = execute_pandas_code(code,df)

                        if error:
                            st.error(f"Error: {error}")
                            st.code(code, language = 'python')
                        else:
                            st.success("Answer: ")

                            with st.expander("Generated Code"):
                                st.code(code,language='python')

                            st.write(result)
            
            # st.sidebar.write(df.columns)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


if __name__ == "__main__":
    main()

