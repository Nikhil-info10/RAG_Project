# IIT PATNA: InnoCorp Knowledge Assistant

## 1. What This Project Does

This project is a document-grounded knowledge assistant for InnoCorp Solutions. It supports:

- PDF, TXT, DOCX, XLSX, and XLS document ingestion
- Chroma vector database storage
- Hybrid retrieval using semantic vector search and BM25 keyword search
- Conversational memory for follow-up questions
- Source citations in answers
- Hallucination handling for unsupported information
- Web-search fallback for questions not answered by internal documents
- Streamlit chat interface
- MCP server tools for compatible AI clients

## 2. Files to Share

Share the project folder with this structure:

```text
Project root/
|-- streamlit_app.py
|-- retrieval_and_rag.py
|-- ingest.py
|-- requirements.txt
|-- Code/
|   |-- config.json
|   |-- requirements.txt              (optional duplicate setup file)
|-- Data/
|   |-- company and policy PDF files
|   |-- employee Excel files
|-- TEAM_SETUP_GUIDE.md
|-- mcp_server.py
|-- .vscode/
|   |-- mcp.json
```

The most important files are:

- `streamlit_app.py`: Streamlit user interface
- `retrieval_and_rag.py`: retrieval, employee lookup, memory, citations, and web fallback
- `ingest.py`: loads files from `Data/` into Chroma
- `requirements.txt`: recommended dependency file
- `Code/config.json`: provider and model configuration
- `Data/`: internal PDFs and Excel workbooks used as the knowledge base
- `mcp_server.py`: MCP stdio server exposing internal search, employee lookup, and web search
- `.vscode/mcp.json`: VS Code configuration for launching the local MCP server

## 3. Do Not Share These Items

Do not share secrets or local machine state:

- `Code/.env`: contains API keys and must be created separately on each laptop
- `.venv/`: each team member should create their own virtual environment
- `__pycache__/`: generated Python cache files
- `~$*.xlsx`: temporary Excel lock files

The `chroma_db/`, `ingest_manifest.json`, and `chroma_provenance.json` files are generated local data. They may be shared, but it is more reliable for each team member to run ingestion locally because stored metadata can contain paths from the original laptop.

## 4. Requirements

Each team member needs:

- Windows 10 or Windows 11
- Python 3.10 or newer; Python 3.12 is recommended
- Internet access for package installation and the first local embedding-model download
- An OpenAI API key if LLM answers and web-search summaries are required
- Access permission for the internal documents in `Data/`

## 5. First-Time Setup on Windows

Open PowerShell in the project root.

### Create a virtual environment

```powershell
py -3.12 -m venv .venv
```

If Python 3.12 is not installed, use the available supported Python version:

```powershell
py -m venv .venv
```

### Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in a PowerShell window opened for your user account:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Install dependencies

Use the root requirements file:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 6. Configure Environment Variables

Create a new file named `Code/.env` on each laptop. Never copy API keys into the guide or commit them to source control.

Minimum configuration for LLM answers:

```dotenv
OPENAI_API_KEY=put-your-key-here
```

Optional Bing search configuration:

```dotenv
BING_API_KEY=put-your-bing-key-here
```

The application also supports `GOOGLE_API_KEY` or `GEMINI_API_KEY` for compatible embedding configurations, but the current `Code/config.json` uses local Hugging Face embeddings:

```json
{
    "provider": "local",
    "models": {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash"
    }
}
```

The `ddgs` package provides the web-search fallback when a Bing key is not configured. Internet access is still required.

## 7. Ingest the Internal Documents

Place approved PDFs and Excel files in `Data/`, then run:

```powershell
python ingest.py
```

The ingestion script:

- Scans `Data/`
- Reads PDFs, TXT, DOCX, XLSX, and XLS files
- Converts Excel rows into searchable text
- Splits documents into chunks
- Creates or updates `chroma_db/`
- Records processed file times in `ingest_manifest.json`

If a file is changed and needs reprocessing, run `python ingest.py` again. If the manifest is copied from another laptop and prevents reprocessing, delete `ingest_manifest.json` and run ingestion again.

## 8. Run the Streamlit Application

From the project root, with the virtual environment active:

```powershell
python -m streamlit run streamlit_app.py
```

Open the URL shown by Streamlit, normally:

```text
http://localhost:8501
```

## 9. Use the MCP Server

The project includes a local stdio MCP server for compatible AI clients. It exposes:

- `search_internal_documents`: hybrid Chroma vector and BM25 search
- `lookup_employee`: employee lookup from the Excel database
- `search_web`: public web search

The VS Code configuration is stored in `.vscode/mcp.json`. After installing dependencies, reload VS Code to discover the server. To run it directly:

```powershell
python mcp_server.py
```

The stdio server waits for an MCP client and normally does not display a normal command prompt.

To use another Streamlit port:

```powershell
python -m streamlit run streamlit_app.py --server.port 8502
```

## 10. Optional CLI Mode

The original terminal interface can still be run with:

```powershell
python retrieval_and_rag.py
```

The Streamlit interface is recommended for normal use.

## 11. Quick Verification Checklist

After setup, test these questions:

1. `What is the leave policy?`
2. `What about carry-forward?`
3. `How many sick leave days has Alex taken?`
4. `Is Alex an employee of InnoCorp?`
5. `What benefits is Alex eligible for?`
6. `What is the latest iPhone 17 Pro specification?`
7. `What information is available about a person who is not in the employee database?`

Confirm that:

- Internal questions cite the relevant PDF or Excel source.
- Follow-up questions use conversation history.
- Employee facts come from the employee workbook.
- Unrelated questions do not return employee leave data.
- Questions absent from internal documents can use web search when internet and an LLM key are available.
- The sidebar Clear conversation button removes the current chat history.

## 12. Common Problems

### `ModuleNotFoundError`

Confirm the virtual environment is active, then install dependencies again:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Chroma database is empty

Run:

```powershell
python ingest.py
```

### Excel data is missing

Check that the workbook is inside `Data/`, has an `.xlsx` or `.xls` extension, and rerun ingestion. Do not use the temporary `~$` Excel lock file.

### No LLM provider is configured

Create `Code/.env` and add:

```dotenv
OPENAI_API_KEY=put-your-key-here
```

Restart Streamlit after changing `.env`.

### Web search returns no answer

Check internet access. The web fallback requires a working network connection and an LLM API key to summarize search results. A Bing key can improve reliability, but is optional because `ddgs` is also supported.

### The browser shows old behavior

Stop the running Streamlit process with `Ctrl+C`, start it again, and use the Clear conversation button. Streamlit keeps session history until it is cleared or the process is restarted.

## 13. Security Notes

- Keep `Code/.env` private.
- Do not put API keys in Python files, screenshots, chat messages, or documentation.
- Share only approved internal documents in `Data/`.
- Treat employee records and policy documents as sensitive company data.
- Use source citations to verify answers before relying on them.
