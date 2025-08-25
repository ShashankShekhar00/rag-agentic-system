# Deep Research AI Agentic System

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20AI-orange.svg)
![Weaviate](https://img.shields.io/badge/Weaviate-Vector%20DB-red.svg)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI%20Model-yellow.svg)

## 🔬 Overview

An advanced AI-powered medical research system using RAG (Retrieval-Augmented Generation) for intelligent document analysis. This system specializes in heart disease research and provides evidence-based medical insights through interactive query interfaces.

##  Features

- **🤖 Agentic AI Architecture**: Built with LangGraph for sophisticated AI agent workflows
- **📚 Advanced RAG System**: Retrieval-Augmented Generation for accurate medical document analysis
- **🔍 Vector Database**: Weaviate integration for efficient document embedding and retrieval
- **🧠 Multi-Model AI**: Google Gemini integration with OpenAI fallback support
- **💻 Interactive Interfaces**: Multiple query modes including terminal and file-based interactions
- **📊 Comprehensive Analysis**: In-depth medical research with source citations and evidence tracking
- **🐳 Docker Support**: Containerized deployment with Docker Compose
- **📈 Professional Reports**: Automated generation of detailed analysis reports

##  Architecture

The system follows a modular, agent-based architecture:

```
src/
├── agents/          # AI agents for research and drafting
├── workflows/       # LangGraph workflow definitions
├── tools/          # Specialized tools for document processing
├── models/         # Data models and tree structures
├── config/         # Configuration management
└── utils/          # Utility functions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- uv package manager
- Google Gemini API key
- Docker (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ShashanShekhar00/rag-agentic-system.git
cd rag-agentic-system
```

2. **Install dependencies**
```bash
uv sync
```

3. **Set up environment variables**
```bash
# Add your API keys to environment variables
export GOOGLE_API_KEY="your_gemini_api_key"
export OPENAI_API_KEY="your_openai_api_key"  # Optional fallback
```

4. **Start Weaviate database**
```bash
docker-compose up -d
```

5. **Run the application**
```bash
uv run python main.py
```

## 📱 Usage

### Interactive Mode
```bash
uv run python main.py
```

Choose from the main menu:
- **RAG Mode**: Interactive question-answering with document retrieval
- **Research Mode**: Comprehensive research report generation
- **Document Management**: Upload and manage medical documents

### Query-to-File Mode
```bash
uv run python interactive_rag.py
```

Select from 15+ predefined medical questions or ask custom queries. Results are automatically saved to timestamped files.

### Direct Query Mode
```bash
uv run python query_to_file.py "Your medical question here"
```

## 🔧 Configuration

Key configuration options in `src/config/settings.py`:

- **Vector Database**: Weaviate connection settings
- **AI Models**: Model selection and parameters
- **Document Processing**: Chunk size and overlap settings
- **Analysis Depth**: Research thoroughness levels

## 📊 Sample Queries

The system excels at medical research queries such as:

- "What are the early warning signs of heart disease?"
- "Compare different treatment approaches for cardiovascular conditions"
- "Analyze the relationship between lifestyle factors and heart health"
- "What are the latest research findings on heart disease prevention?"

## 🗂 Document Support

Currently optimized for:
- Heart disease research papers
- Medical guidelines and protocols
- Clinical study reports
- Health education materials

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **LangGraph**: For the agentic AI framework
- **Weaviate**: For vector database capabilities
- **Google Gemini**: For advanced language model support
- **Medical Research Community**: For providing valuable datasets

##  Contact

**Shashank Shekhar** - [GitHub](https://github.com/ShashanShekhar00)

Project Link: [https://github.com/ShashanShekhar00/rag-agentic-system](https://github.com/ShashanShekhar00/rag-agentic-system)
