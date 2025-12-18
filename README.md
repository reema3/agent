<h1>Conversational Data Insights App</h1>

**Talk to your data in plain English — no SQL, no Python, no Excel formulas.**

This project is my attempt to build a lightweight **LLM-powered analytics assistant** where users can explore a dataset using natural language. Instead of writing queries or code, the user simply asks a question like:

“Show me the top 5 products by sales.”

“Why did sales drop in 2017?”

“Plot month-wise sales trend.”

…and the app generates the analysis, charts, and explanations automatically.

<h2>🚀 Key Features </h2>

<h3>🔹 Natural-Language to Data Analysis</h3>

- Users ask questions in plain English
- The app interprets intent and runs the appropriate data operations
- No need to write code or queries

<h3> 🔹 Business Insight Generation </h3>

- Raw outputs are converted into **business-friendly insights**
- Focus on *what the numbers mean*, not just the numbers themselves

<h3>🔹 Insight Memory using Vector Database </h3>

- User questions and generated insights are stored as embeddings
- When a similar question is asked again, the system can:
  - Reuse prior insights
  - Improve consistency and response quality
- Demonstrates a simple **RAG-style feedback loop**

<h3>🔹 Streamlit UI</h3>

A fast, clean web interface built with Streamlit that runs locally or can be deployed anywhere.

<h3>🔹 LLM-Powered Reasoning</h3>

Uses an LLM backend to:

- Understand user intent
- Validate and structure the query
- Generate Python code on the fly
- Execute that code safely on the dataset

<h3>🔹 Visual Analytics</h3>

Supports:
- Bar charts
- Line charts
- Time series
- Category breakdowns

(Using matplotlib / plotly.)

<h3>🔹 Open-Source Sample Dataset</h3>

App uses the Superstore Sales dataset, a popular public dataset used in Tableau demos, Kaggle notebooks, and BI case studies.

<h2>🛠️ Tech Stack</h2>

• **Python 3**

• **Streamlit** — UI

• **Pandas** — Data handling

• **Matplotlib / Plotly** — Visualizations

• **OpenAI (or any LLM of choice)** — Natural-language understanding

• **Vector Database** (e.g. FAISS / Chroma) – Insight memory

• **Virtual Environment** (venv)


<h2>📦 Installation</h2>
  
	1.	Clone the repo

  git clone <your-repo-url>
  
  cd your-repo-name
  
	2.	Create and activate virtual env

  python3 -m venv myenv
  
  source myenv/bin/activate
  
	3.	Install dependencies

  pip install -r requirements.txt

	4.	Run the app

streamlit run app.py


<h2>🧠 How It Works</h2>

	1.	User enters a natural-language query
  
	2.	LLM interprets the intent → generates safe Python code
  
	3.	Code is executed on the dataset
  
	4.	Results and visualizations are rendered back to Streamlit

This approach allows:

	•	Rewriting business questions into analytical tasks
  
	•	Faster exploratory analysis without manual coding

<h2> Screenshot </h2>
<img width="2850" height="1612" alt="image" src="https://github.com/user-attachments/assets/a0da5453-faf0-400e-95aa-40e96344db6d" />
https://analyticsassistant.streamlit.app/

<h2> 🎯 Why I Built This</h2>

I wanted to explore how LLMs can make analytics more accessible — especially for people who can interpret insights but aren’t comfortable writing SQL or Python.

This project helped me understand:

	•	Prompt engineering
  
	•	Code-generation agents
	
	•	Explore how vector memory can improve analytical systems
  
	•	Building simple data apps end-to-end
