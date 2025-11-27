from openai import OpenAI
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title = 'Conversation AI - BI Dashboard',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)