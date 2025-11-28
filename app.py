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
    page_title = 'Python Assistant - talk to your data',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

@st.cache_resource
def init_conenctions():
    "Setup OpenAI"
    llm_model="openai/gpt-oss-20b:groq"
    api_key=os.environ["HF_TOKEN"]
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
    # df['Order Month'] = df['Order Date'].dt.to_period('M').astype(str)
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

def user_wants_chart(question):

    question_lower = question.lower()

    chart_keywords = [
        'chart','plot','graph','visualize','visualization',
        'show me a chart', 'show chart', 'plot this','graph this',
        'pie chart','bar chart','line chart','trend','show me visually',
        'visual','diagram'
    ]

    return any(keyword in question_lower for keyword in chart_keywords)

def generate_chart_from_result(result, question):
    # Keywords to determine chart type
    trend_keywords = ['trend', 'over time','daily','monthly','timeline','preogression']
    comparison_keywords = ['compare', 'vs','versus', 'comparison']
    distribution_keywords = ['by region','by product','breakdown', 'distribution']

    question_lower = question.lower()

    # Check if result is suitable for charting
    if result is None or isinstance(result, (int, float, str)):
        return None
    
    try:
        # CASE 1: Time series data (has dates in index)
        if isinstance(result, pd.Series) and pd.api.types.is_datetime64_any_dtype(result.index):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result.index,
                y=result.values,
                mode=' lines+markers',
                name='Value',
                line=dict(color='#667eea', width=2)
                ))
            fig.update_layout(
                title="Trend Analysis",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400
                )
            return fig
        
         # CASE 2: Time series data (has dates in index)
        elif isinstance(result, pd.Series) and len(result)>1:
            # Check if it's better as pie or bar
            if any(word in question_lower for word in distribution_keywords) and len(result)<=10:
                # Pie chart for distribution
                fig = px.pie(
                    values=result.values,
                    names=result.index,
                    title="Distribution Analysis"
                )
            else:
                # Bar chart for comparisón
                fig = px.bar(
                    x=result.index,
                    y=result.values,
                    title="Comparison Analysis", 
                    color=result.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    xaxis_title=result.index.name or "Category",
                    yaxis_title="Value",
                    height=400
                )
            return fig
        
        # CASE 3: DataFrame time series
        elif isinstance(result, pd. DataFrame) and 'Order Date' in result. columns:
            # Line chart for time series
            value_cols = [col for col in result.columns if col != 'Order Date']
            fig = go.Figure()
            for col in value_cols:
                fig.add_trace(go.Scatter(
                    x=result ['Order Date'],
                    y=result[col],
                    mode= 'lines+markers',
                    name=col
                ))
                fig.update_layout(
                    title="Trend Analysis",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=400
                )
            return fig
        

    except Exception as e:
        print(f"Chart generation error {e}")
        return None

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
    st.markdown(
        """
        <style>
        /* Style for the metric cards */
        [data-testid="stMetric"] {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #1f77b4; /* Streamlit Blue accent */
            transition: all 0.3s ease;
        }
        [data-testid="stMetric"]:hover {
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        /* Style for the KPI value to be larger and bold */
        [data-testid="stMetricValue"] {
            font-size: 2.5em !important;
            font-weight: 700 !important;
            color: #1f77b4; 
        }
        [data-testid="stMetricLabel"] {
            font-size: 1.1em;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1><font color='#1f77b4'>Conversational BI Dashboard</font></h1>", unsafe_allow_html=True)


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

                            # Only generate chart if user asks for it
                            if user_wants_chart(user_question):
                                chart = generate_chart_from_result(result, user_question)
                                if chart:
                                   st.plotly_chart(chart, use_container_width=True)
                                else:
                                    st.info(" Unable to generate a chart for this data type. Try a different query.")
            
            # st.sidebar.write(df.columns)
            st.markdown("---")
            st.caption("Example Questions")
            st.markdown(
                """
                - Total Sales in 2017?
                - Sales per Category?
                - Average Sales for Furniture?
                - Which country has the highest sales?
                """
            )


            # --- Main Dashboard Layout ---
        st.subheader("Key Performance Indicators")
        
        # 1. KPI Cards using columns for better layout (styling via CSS injection above)
        total_sales = df['Sales'].sum()
        total_orders = df['Order ID'].nunique()
        
        kpi_col1, kpi_col2 = st.columns(2)

        with kpi_col1:
            st.metric(label="Total Sales (USD)", value=f"${total_sales:,.2f}")
        
        with kpi_col2:
            st.metric(label="Total Orders", value=f"{total_orders:,}")


        # 2. Charts Section Layout
        st.markdown("---")
        st.subheader("Sales and Performance Deep Dive")

        # Create two columns for the charts
        chart_col1, chart_col2 = st.columns(2)


        # Chart 1: Sales Over Time Chart (Left Column)
        with chart_col1:
            st.caption("Monthly Sales Trend")

            # Aggregate sales by month
            df['Order Month'] = df['Order Date'].dt.to_period('M').astype(str)
            monthly_sales = df.groupby('Order Month')['Sales'].sum().reset_index()
            monthly_sales['Order Month'] = pd.to_datetime(monthly_sales['Order Month']) 

            fig_time = px.line(
                monthly_sales, 
                x='Order Month', 
                y='Sales', 
                title='Total Sales Over Time',
                markers=True
            )
            
            fig_time.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Order Month",
                yaxis_title="Total Sales",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig_time.update_traces(line_color='#1f77b4', line_width=2) # Blue color for the line

            st.plotly_chart(fig_time, use_container_width=True)


        # Chart 2: Sales by Product Category (Right Column)
        with chart_col2:
            st.caption("Sales Distribution by Product Category")
            
            # Aggregate sales by segment
            segment_sales = df.groupby('Category')['Sales'].sum().reset_index()

            fig_segment = px.pie(
                segment_sales,
                names='Category',
                values='Sales',
                title='Sales by Category',
                hole=.3 # Creates a donut chart
            )
            
            fig_segment.update_traces(textposition='inside', textinfo='percent+label')
            fig_segment.update_layout(showlegend=True, margin=dict(l=20, r=20, t=40, b=20))

            st.plotly_chart(fig_segment, use_container_width=True)


        # 3. Data Preview (moved into an expander for cleanliness)
        st.markdown("---")
        with st.expander("Show Raw Data Preview"):
            st.dataframe(df.head())


    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


if __name__ == "__main__":
    main()

