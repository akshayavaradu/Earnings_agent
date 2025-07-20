from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine
import re
import pandas as pd
import json
from typing import List, Type, Literal 
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import json
from langchain.vectorstores import FAISS
import os
from glob import glob
from IPython.display import HTML, Markdown, Image 
from pydantic import BaseModel
import json
import os
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

dic = r'C:\Users\Akshaya V\git\Earnings research\Earnings_agent\PublicReportResearch-main'
full_path = os.path.join(dic, 'docs')
question="Extract the net revenue of citi bank"
llm_model_name="qwen3:4b"
embedding_model="nomic-embed-text"
sec_excel_path=r'C:/Users/Akshaya V/git/Earnings research/Earnings_agent/PublicReportResearch-main/50_metrics.xlsx'
sec_excel=pd.read_excel(r'C:/Users/Akshaya V/git/Earnings research/Earnings_agent/PublicReportResearch-main/50_metrics.xlsx')
bank_name_mapping = {'AMERICAN EXPRESS COMPANY': 'American Express',
    'Bank of America Corporation': 'Bank of America',
    'CAPITAL\xa0ONE\xa0FINANCIAL\xa0CORP': 'Capital One',
    'Citigroup\xa0Inc': 'Citi',
    'Fifth Third Bancorp': 'Fifth Third',
    'Huntington Bancshares Incorporated': 'Huntington Bank',
    'JPMorgan Chase & Co': 'JPMorgan Chase',
    'KeyCorp': 'KeyBank',
    'NORTHERN TRUST CORPORATION': 'Northern Trust',
    'PNC Financial Services Group, Inc.': 'PNC Bank',
    "People's United Financial, Inc.": 'Peoples United',
    'SCHWAB CHARLES CORP': 'Charles Schwab',
    'STATE STREET CORPORATION': 'State Street',
    'TEGNA INC.': 'Tegna',
    'THE BANK OF NEW YORK MELLON CORPORATION': 'BNY Mellon',
    'TRUIST FINANCIAL CORPORATION': 'Truist',
    'The Goldman Sachs Group, Inc.': 'Goldman Sachs',
    'US BANCORP \\DE\\': 'US Bancorp',
    'WELLS FARGO & COMPANY/MN': 'Wells Fargo'}

sec_excel['CompanyName']=sec_excel['CompanyName'].replace(bank_name_mapping)
allowed_metrics: List[str] = sec_excel.columns[2:].unique().tolist()
allowed_banks: List[str] = sec_excel['CompanyName'].unique().tolist()

class ParsedRequest_intent(BaseModel):
    intent: str                 
 

def intent(
    llm_model_name: str,
    user_input: str,
    schema: Type[BaseModel]
    
) -> BaseModel:
    system_prompt = """You are a expert in classifying questions into 2 categories. The 2 categories are exact question , needs_clarification. A question is marked as exact if it has concrete details in 3 categories - company name, quarter & Year , metrics to be analysed. It is needs_clarification if the user uses words like analyse, in detail , research, elaborate or if the user doesn't provide specific metrics or company name or year & quarter to be analysed . give the output in JSON format srtictly . JSON has one key and it is called intent . For example ) {"intent": "exact"}"""
    llm = ChatOllama(model=llm_model_name)

    messages = [
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=user_input.strip())
    ]
    response = llm.invoke(messages)
    return response.content.strip()

def vague(model, question, allowed_banks, allowed_metrics):
    prompt = f"""
You are an expert in rewriting vague finance-related questions. Your sole task is to **rephrase the user's question** by **expanding it explicitly** along three dimensions:
- Company names (banks)
- Financial metrics
- Quarters and Years

Important:
- DO NOT provide an answer, explanation, or rationale.
- ONLY return the **rewritten question** in plain text.

Defaults (if the user doesn’t specify):
1. Company: Wells Fargo
2. Quarters: 1Q2025, 4Q2024, 3Q2024
3. Metrics: NetIncome, EarningsPerShare, TotalRevenue
4. Add important metrics for senior leadership (e.g., ReturnOnEquity, ROA, CET1Ratio)

Rules:
- Use only official bank names from: {json.dumps(allowed_banks)}
- Convert quarters to format: "1Q2025"
- Use only metrics from: {json.dumps(allowed_metrics)}

Example:
User: Extract the net revenue of citi bank in 2025Q1  
Output: Extract TotalRevenue, NetIncome, EarningsPerShare, and ReturnOnEquity for Citigroup Inc in 1Q2025, 4Q2024, and 3Q2024.

Respond ONLY with the rewritten version of the user’s question.
Here is the question:  {question}
    """

    llm = ChatOllama(model=model)
   
    response = llm.invoke(prompt)
    return response.content.strip()

class ParsedRequest(BaseModel):
                 
    banks: List[str]            # Must match allowed_banks
    quarters: List[str]         # e.g., "1Q2025", "4Q2024"
    metrics: List[str]          # Must match allowed_metrics


def parse_with_chatollama(
        llm_model_name: str,
        user_input: str,
        schema: Type[BaseModel],
        allowed_banks: List[str],
        allowed_metrics: List[str]
    ) -> BaseModel:
    system_prompt = f"""
You are a financial assistant. Your task is to extract structured information from user input and return it in the following JSON format:

{{
  
  "banks": [valid bank names],
  "quarters": ["1Q2025", "4Q2024", "3Q2024"],
  "metrics": [valid metric keys]
}}

Rules:
- Map any abbreviation or alias to official bank names from this list: {json.dumps(allowed_banks)}
- Extract all mentioned quarters in "1Q2025" format. Include the previous 2 quarters for each.
- Extract only metrics listed here: {json.dumps(allowed_metrics)}.
- Output only the JSON structure as shown above, no explanation or markdown.
"""

    llm = ChatOllama(model=llm_model_name)

    messages = [
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=user_input.strip())
    ]
    response = llm.invoke(messages)
    return response.content.strip()

class ParsedRequest(BaseModel):
    intent: str
    banks: List[str]
    quarters: List[str]
    metrics: List[str]

def is_natural_language(text):
    return bool(re.search(r"[A-Za-z]{4,}.*\.", text)) and not is_table_like(text)

def is_table_like(text):
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return False

    table_like = 0
    for line in lines:
        tokens = line.strip().split()
        num_tokens = len(tokens)
        numbers = len([t for t in tokens if re.fullmatch(r"[\d,.%$]+", t)])
        symbols = len([t for t in tokens if re.fullmatch(r"[\d,.%$O/(U)-]+", t)])

        if num_tokens >= 3 and numbers / num_tokens > 0.5:
            table_like += 1
        elif len(re.findall(r"\$\s?\d", line)) > 1:
            table_like += 1
        elif len(re.findall(r"\d{2,},", line)) > 1:
            table_like += 1

    return table_like / len(lines) > 0.4

def extract_text_excluding_tables(pdf_path):
    final_text = []
    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, (LTTextBox, LTTextLine)):
                text = element.get_text().strip()
                if text and is_natural_language(text):
                    final_text.append(text)
    return "\n\n".join(final_text).strip()

def parse_filename_metadata(filename: str):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_")
    bank_map = {
        "jpm": "JP Morgan Chase",
        "boa": "Bank of America",
        "citi": "Citigroup",
        "gs": "Goldman Sachs",
        "ms": "Morgan Stanley",
    }
    bank_code = parts[0].lower()
    quarter = parts[1].upper() if len(parts) > 1 else "UNKNOWN"
    bank = bank_map.get(bank_code, bank_code.upper())
    return bank, quarter

def create_chunks_with_ollama(text: str, metadata: dict = None):
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    chunker = SemanticChunker(embeddings=embedder, min_chunk_size=2000)
    doc = Document(page_content=text, metadata=metadata or {})
    return chunker.split_documents([doc])

def search_chunks(chunks, parsed_query: ParsedRequest, top_k=5, index_path="faiss_index"):
    embedder = OllamaEmbeddings(model="nomic-embed-text")

    filter_dict = {
        "bank": parsed_query.banks,
        "quarter": parsed_query.quarters
    }

    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embedder)
        vectorstore.save_local(index_path)

    query_text = (
        f"Find information about {', '.join(parsed_query.metrics)} "
        f"for banks like {', '.join(parsed_query.banks)} "
        f"during quarters such as {', '.join(parsed_query.quarters)}"
    )

    return vectorstore.similarity_search(query_text, k=top_k, filter=filter_dict)

def rerank_with_chatollama(chunks, parsed_query: ParsedRequest):
    llm = ChatOllama(model="llama3.2:3b")
    context = "\n\n".join([chunk.page_content for chunk in chunks])
    prompt = (
        f"You are a financial analyst assistant.\n\n"
        f"Query:\n{parsed_query.model_dump_json(indent=2)}\n\n"
        f"Extracted text:\n{context}\n\n"
        f"Please summarize the relevant details in a table or paragraph based on the query intent."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

all_chunks = []

for pdf in os.listdir(full_path):
    if not pdf.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(full_path, pdf)
    print(f" Processing: {pdf}")

    bank, quarter = parse_filename_metadata(pdf)
    clean_text = extract_text_excluding_tables(pdf_path)

    tagged_text = f"<{bank} {quarter}>\n{clean_text}\n</{bank} {quarter}>"
    metadata = {"source": pdf, "bank": bank, "quarter": quarter}
    chunks = create_chunks_with_ollama(tagged_text, metadata)
    all_chunks.extend(chunks)

print(f"\nTotal Chunks Created: {len(all_chunks)}")
for i, chunk in enumerate(all_chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print("Metadata:", chunk.metadata)
    print("Content preview:", chunk.page_content[:300], "...\n")

class FinancialDataInput(BaseModel):
    banks: List[str]
    quarters: List[str]
    metrics: List[str]

def convert_to_qtr(date_str):
    date = pd.to_datetime(date_str)
    quarter = (date.month - 1) // 3 + 1
    return f"{quarter}Q{date.year}"

@tool
def extract_bank_metrics(input_data: FinancialDataInput) -> dict:
    excel_path = sec_excel_path
    df = pd.read_excel(excel_path)
    bank_name_mapping = {'AMERICAN EXPRESS COMPANY': 'American Express',
    'Bank of America Corporation': 'Bank of America',
    'CAPITAL\xa0ONE\xa0FINANCIAL\xa0CORP': 'Capital One',
    'Citigroup\xa0Inc': 'Citi',
    'Fifth Third Bancorp': 'Fifth Third',
    'Huntington Bancshares Incorporated': 'Huntington Bank',
    'JPMorgan Chase & Co': 'JPMorgan Chase',
    'KeyCorp': 'KeyBank',
    'NORTHERN TRUST CORPORATION': 'Northern Trust',
    'PNC Financial Services Group, Inc.': 'PNC Bank',
    "People's United Financial, Inc.": 'Peoples United',
    'SCHWAB CHARLES CORP': 'Charles Schwab',
    'STATE STREET CORPORATION': 'State Street',
    'TEGNA INC.': 'Tegna',
    'THE BANK OF NEW YORK MELLON CORPORATION': 'BNY Mellon',
    'TRUIST FINANCIAL CORPORATION': 'Truist',
    'The Goldman Sachs Group, Inc.': 'Goldman Sachs',
    'US BANCORP \\DE\\': 'US Bancorp',
    'WELLS FARGO & COMPANY/MN': 'Wells Fargo'}

    df['CompanyName']=df['CompanyName'].replace(bank_name_mapping)
    df['Datetime']=pd.to_datetime(df['Datetime'],format="%Y-%m-%d")
    df['Quarter']=df['Datetime'].apply(convert_to_qtr)
    print(df.head())

    banks = input_data.banks
    quarters = input_data.quarters
    metrics = input_data.metrics

    result = df.loc[(df['CompanyName'].str.upper().isin([b.upper() for b in banks])) & (df['Quarter'].isin(quarters).values)]
    if result.empty:
        return {"message":"No data found"}
    print(result) 
    response={"CompanyName":result['CompanyName'].values,
              "Quarter":quarters,}
    for metric in metrics:
        response[metric]=result.iloc[0].get(metric,"metric not found")
    result=result[['CompanyName','Datetime']+metrics]
    result['Datetime']=result['Datetime'].apply(convert_to_qtr)
    response=result.to_json(orient='records')
    return response

class ChartInput(BaseModel):
    data: List[dict]
    chart_type: Literal['bar', 'line']
    title: str
    x_axis: str
    y_axis: List[str]

@tool
def create_chart(input_data: ChartInput) -> str:
    df = pd.DataFrame(input_data.data)
    
    if df.empty:
        return "No data provided to generate chart."

    plt.figure(figsize=(10, 6))
    
    if input_data.chart_type == 'bar':
        df.plot(x=input_data.x_axis, y=input_data.y_axis, kind='bar')
    elif input_data.chart_type == 'line':
        df.plot(x=input_data.x_axis, y=input_data.y_axis, kind='line')

    plt.title(input_data.title)
    plt.xlabel(input_data.x_axis)
    plt.ylabel(', '.join(input_data.y_axis))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    chart_path = f"{input_data.title.replace(' ', '_')}.png"
    plt.savefig(chart_path)
    plt.close()
    
    return f"Chart saved to {chart_path}"

def generate_detailed_html_report(llm_model_name: str, table_output: dict, extracted_text: str,question:str,intent:str) -> str:
    llm = ChatOllama(model=llm_model_name)
    exact_prompt=f""" You are provided with a json{table_output}. Convert that to html format by adding tags and display it as a table 
User question :{question}"""
    prompt = f"""
You are a professional financial analyst creating an internal report for Wells Fargo senior leadership.
Your response must contain three clearly formatted parts in HTML TAgs with embedded CSS to make it visually appealing and boardroom-ready.

### Instructions:
1. **Part 1**: Present the input JSON request as a table and add HTML Table tags (no changes in values).
2. **Part 2**: Use the extracted text to create a detailed narrative summary based on the below question asked. Structure this as a qualitative analysis that highlights trends, anomalies, risks, and growth areas.
3. **Part 3**: Ensure the writing style is impressive to senior executives — use formal, insightful, and concise language.

### Formatting Rules:
- Use <table>, <thead>, <tbody>, <tr>, <th>, <td> for tables
- Use <p>, <h2>, <h3> for textual sections
- Use inline CSS to match Wells Fargo branding (deep red: #b31b1b, gold: #ffd700, and professional fonts)
- Make layout visually appealing (padding, borders, alternating row colors, aligned text)
- Output ONLY valid HTML tags (no Markdown or commentary)

---

JSON Request:
```json
{table_output}
```

Extracted Report Text:
{extracted_text}

User question :
{question}
"""
    if intent=="exact":
        final_question=exact_prompt
    else:
        final_question=prompt
    response = llm.invoke([HumanMessage(content=final_question.strip())])
    return response.content

result = intent(
    llm_model_name=llm_model_name,
    user_input=question,
    schema=ParsedRequest_intent
)
intent1 = result.split("</think>")[-1].strip()
intent=(json.loads(intent1))
print("\n intetnt:", intent)

if (intent['intent']!="exact"):
    result = vague(
    model=llm_model_name,
    question=question,
    allowed_banks=allowed_banks,
    allowed_metrics=allowed_metrics
    )
    question = result.split("</think>")[-1].strip()
    print("\n question:",question)

result = parse_with_chatollama(
    llm_model_name=llm_model_name,
    user_input=question,
    schema=ParsedRequest,
    allowed_banks=allowed_banks,
    allowed_metrics=allowed_metrics
)
response = result.split("</think>")[-1].strip()
print("\n Pydantic schema: ",response)
input_params=(json.loads(response))
print(input_params)


intent_ext=intent['intent']

structured_query = {**intent_ext, **input_params}
print("\n structured query:",structured_query)

structured_query_json = json.dumps(structured_query, indent=2)
print("\n structured query:",structured_query_json)

parsed_query = ParsedRequest.model_validate(json.loads(structured_query_json))

index_path = "faiss_index"
embedder = OllamaEmbeddings(model=embedding_model)

if os.path.exists(index_path):
    print("Loading FAISS index from disk...")
    vectorstore = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
else:
    print("No FAISS index found. Creating new one from all_chunks...")
    vectorstore = FAISS.from_documents(all_chunks, embedder)
    vectorstore.save_local(index_path)
    print("Saved FAISS index to disk.")

query_text = (
    f"Find information about {', '.join(parsed_query.metrics)} "
    f"for banks like {', '.join(parsed_query.banks)} "
    f"during quarters such as {', '.join(parsed_query.quarters)}"
)

filter_dict = {
    "bank": parsed_query.banks,
    "quarter": parsed_query.quarters
}

matched_chunks = vectorstore.similarity_search(query_text, k=5, filter=filter_dict)

final_ans = ""
for i, chunk in enumerate(matched_chunks):
    metadata = chunk.metadata
    print(f"\n--- Matched Chunk {i+1} ---")
    print(f" Source: {metadata.get('source', 'N/A')}")
    print(f" Bank: {metadata.get('bank', 'Unknown')}")
    print(f" Quarter: {metadata.get('quarter', 'Unknown')}")
    print(" Content Preview:\n", chunk.page_content[:300], "...\n")
    
    final_ans += (
        f"<div class='chunk'>\n"
        f"<p><strong>Source:</strong> {metadata.get('source', 'N/A')}</p>\n"
        f"<p><strong>Bank:</strong> {metadata.get('bank', 'Unknown')}</p>\n"
        f"<p><strong>Quarter:</strong> {metadata.get('quarter', 'Unknown')}</p>\n"
        f"<p>{chunk.page_content}</p>\n"
        f"</div>\n\n"
    )

print(" Final Combined Answer with Metadata:\n")
print(final_ans)


excel_Data = extract_bank_metrics.invoke({"input_data":input_params})
table_output=json.dumps(excel_Data, indent=2)
print(" \n Table output",table_output)


intent=intent['intent']


html_output = generate_detailed_html_report(llm_model_name, table_output, final_ans, question,intent)

display(HTML(html_output))

if excel_Data and excel_Data != '{"message":"No data found"}':
    chart_input = {
        "data": json.loads(excel_Data),
        "chart_type": "bar",
        "title": f"Financial Metrics for {', '.join(input_params['banks'])}",
        "x_axis": "Datetime",
        "y_axis": input_params['metrics']
    }
    chart_path = create_chart.invoke({"input_data": chart_input})
    print(chart_path)
    display(Image(filename=chart_path.split(' ')[-1]))

print("Script finished successfully!")
