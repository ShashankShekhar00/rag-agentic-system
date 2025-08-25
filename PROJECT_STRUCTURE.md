# Deep Research AI Agentic System - Clean Project Structure

## 📁 Project Overview
A production-ready RAG (Retrieval-Augmented Generation) system for medical document analysis, specifically designed for heart disease research and question answering.

## 🗂️ Clean Project Structure

```
deep-research-ai-agent/
├── 📄 Core Application Files
│   ├── main.py                    # Main application with interactive menu
│   ├── interactive_rag.py         # Interactive query interface with file output
│   ├── query_to_file.py          # Simple script for saving queries to files
│   └── EXAMPLE_QUESTIONS.md       # Sample questions for testing
│
├── ⚙️ Configuration & Dependencies
│   ├── .env                      # Environment variables (API keys)
│   ├── pyproject.toml           # Project configuration and dependencies
│   ├── requirements.txt         # Python package requirements
│   └── uv.lock                  # Locked dependency versions
│
├── 🧠 Source Code
│   └── src/
│       ├── agents/              # AI agents for analysis
│       │   ├── __init__.py
│       │   ├── drafting_agent.py    # AI analysis and report generation
│       │   └── research_agent.py    # Research coordination
│       │
│       ├── config/              # Configuration management
│       │   ├── __init__.py
│       │   └── settings.py          # Application settings and API keys
│       │
│       ├── models/              # Data models
│       │   ├── __init__.py
│       │   └── tree.py              # Tree structure for organized output
│       │
│       ├── tools/               # Core functionality tools
│       │   ├── __init__.py
│       │   ├── analysis_tools.py    # Content analysis utilities
│       │   ├── decorators.py        # Function decorators
│       │   ├── document_tools.py    # PDF processing and chunking
│       │   ├── rag_tools.py         # RAG search and retrieval
│       │   ├── search_tools.py      # Web search integration
│       │   └── vector_tools.py      # Weaviate vector database operations
│       │
│       ├── utils/               # Utility functions
│       │   └── __init__.py
│       │
│       └── workflows/           # Main workflows
│           ├── rag_workflow.py      # Primary RAG analysis workflow
│           └── research_workflow.py  # Web research workflow
│
├── 📊 Output & Reports
│   └── reports/                 # Generated analysis reports (cleaned)
│
└── 🔧 Development Environment
    ├── .git/                    # Git repository
    ├── .venv/                   # Python virtual environment
    └── README.md                # Project documentation
```

## 🚀 Core Features
- **Medical Document Analysis**: Upload PDFs and get intelligent answers about heart disease
- **Vector Database**: Weaviate integration for semantic document search
- **AI Integration**: Google Gemini and OpenAI support with enhanced local fallback
- **Multiple Interfaces**: Command-line, interactive menu, and file output options
- **Structured Output**: Save comprehensive analysis reports to timestamped text files

## 📱 Usage Options

### 1. Main Application (Interactive Menu)
```bash
uv run python main.py
```

### 2. Interactive Query Interface
```bash
uv run python interactive_rag.py
```

### 3. Simple File Output Script
```bash
uv run python query_to_file.py
```

## 🗃️ Document Support
- **Supported**: PDF files (medical documents, research papers)
- **Sample Data**: Heart disease documents (healthyheart.pdf, Heart Disease-Full Text.pdf)
- **Processing**: Automatic chunking and vector embedding

## 🔑 Required Setup
1. Weaviate database running (via Docker Compose)
2. API keys configured in `.env`:
   - `GOOGLE_API_KEY` (primary)
   - `OPENAI_API_KEY` (fallback)
   - `TAVILY_API_KEY` (web search)

## 🧹 Cleanup Summary
**Removed Files:**
- 25+ test files (`test_*.py`)
- 5+ debug files (`debug_*.py`)
- Temporary output files
- Old report files
- Python cache files (`__pycache__`)
- Duplicate directories and examples
- Development artifacts

**Kept Files:**
- Core application logic
- Essential configuration
- Production-ready interfaces
- Clean documentation
- Required dependencies

This is now a clean, production-ready medical AI research system! 🎉
