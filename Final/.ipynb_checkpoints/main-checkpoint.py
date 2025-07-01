# Constants and input parameters
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


# From cell 6

result = parse_with_chatollama(
    llm_model_name=llm_model_name,  # Replace with your loaded Ollama model name
    user_input=question,
    schema=ParsedRequest,
    allowed_banks=allowed_banks,
    allowed_metrics=allowed_metrics
)
response = result.split("</think>")[-1].strip()
print(response)

# response=(result.model_dump_json())
input_params=(json.loads(response))
print(input_params)

# From cell 8
embedder = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)


# From cell 9
import json
import os
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
#intent="exact"
structured_query={'intent': 'needs_clarification', 'banks': ['Citi'], 'quarters': ['1Q2025', '4Q2024', '3Q2024'], 'metrics': ['TotalRevenue', 'NetIncome', 'EarningsPerShare', 'ReturnOnEquity']}
# Prepare structured query
#print(intent, input_params)
#structured_query = {**intent, **input_params}
print(structured_query)

structured_query_json = json.dumps(structured_query, indent=2)
print(structured_query_json)

parsed_query = ParsedRequest.model_validate(json.loads(structured_query_json))

# Load or create FAISS index
index_path = "faiss_index"
embedder = OllamaEmbeddings(model="nomic-embed-text")

if os.path.exists(index_path):
    print("🔁 Loading FAISS index from disk...")
    vectorstore = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
else:
    print("⚙️ No FAISS index found. Creating new one from all_chunks...")
    vectorstore = FAISS.from_documents(all_chunks, embedder)
    vectorstore.save_local(index_path)
    print("✅ Saved FAISS index to disk.")

# Build semantic search query
query_text = (
    f"Find information about {', '.join(parsed_query.metrics)} "
    f"for banks like {', '.join(parsed_query.banks)} "
    f"during quarters such as {', '.join(parsed_query.quarters)}"
)

matched_chunks = vectorstore.similarity_search(query_text, k=5)

# Output matched content with metadata
final_ans = ""
for i, chunk in enumerate(matched_chunks):
    metadata = chunk.metadata
    print(f"\n--- Matched Chunk {i+1} ---")
    print(f"📄 Source: {metadata.get('source', 'N/A')}")
    print(f"🏦 Bank: {metadata.get('bank', 'Unknown')}")
    print(f"📅 Quarter: {metadata.get('quarter', 'Unknown')}")
    print("🧠 Content Preview:\n", chunk.page_content[:300], "...\n")
    
    final_ans += (
        f"<div class='chunk'>\n"
        f"<p><strong>Source:</strong> {metadata.get('source', 'N/A')}</p>\n"
        f"<p><strong>Bank:</strong> {metadata.get('bank', 'Unknown')}</p>\n"
        f"<p><strong>Quarter:</strong> {metadata.get('quarter', 'Unknown')}</p>\n"
        f"<p>{chunk.page_content}</p>\n"
        f"</div>\n\n"
    )

print("✅ Final Combined Answer with Metadata:\n")
print(final_ans)


# From cell 11
# structured_input = FinancialDataInput(**input_params)

# 🔧 Call your tool (directly)
excel_Data = extract_bank_metrics.invoke({"input_data":input_params})
#structured_input)
print(json.dumps(excel_Data, indent=2))


# From cell 12
# input_json = structured_query_json
# input1=(json.loads(structured_query_json))
# print(type(input1))
# print(type(input1.get('banks')))
# excel_path = sec_excel_path
# output = get_financial_data(excel_path, input1)

# # Pretty print
# table_output=json.dumps(output, indent=2)


# From cell 13
table_output=json.dumps(excel_Data, indent=2)

# From cell 15
# Example parsed_query dict
input_params={'banks': ['Citi'], 'quarters': ['1Q2025', '4Q2024', '3Q2024'], 'metrics': ['TotalRevenue', 'NetIncome', 'EarningsPerShare', 'ReturnOnEquity']}
parsed_query = input_params

# Example: Create a string of matched chunk content


# Call function
html_output = generate_detailed_html_report(llm_model_name, table_output, final_ans, question)

# Optionally write to HTML file
display(HTML(html_output))

# From cell 16
(html_output.split("</think>")[-1].strip())

# From cell 17
html_output

# From cell 18
