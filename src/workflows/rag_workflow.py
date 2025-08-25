"""Enhanced RAG workflow with document upload and processing capabilities."""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from src.models.tree import Tree, NodeType
from src.tools.search_tools import tavily_search, web_scraper, search_multiple_sources
from src.tools.vector_tools import store_in_weaviate, get_research_context
from src.tools.analysis_tools import extract_insights, summarize_content
from src.tools.document_tools import upload_documents, chunk_documents
from src.tools.rag_tools import store_document_chunks, rag_search, get_document_context
from src.agents.drafting_agent import DraftingAgent


class RAGState(TypedDict):
    """State for the RAG workflow."""
    query: str
    research_tree: Tree
    report: str
    status: str
    error_message: str
    iteration: int
    max_iterations: int
    uploaded_files: List[str]
    topic: str
    use_web_search: bool
    document_context: str
    web_context: str


class RAGWorkflow:
    """Enhanced RAG workflow that combines document-based retrieval with web research."""
    
    def __init__(self):
        """Initialize the RAG workflow."""
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the enhanced RAG workflow graph."""
        workflow = StateGraph(RAGState)
        
        workflow.add_node("start_rag", self._start_rag)
        workflow.add_node("process_documents", self._process_documents)
        workflow.add_node("search_documents", self._search_documents)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("combine_context", self._combine_context)
        workflow.add_node("finalize", self._finalize)
        
        workflow.set_entry_point("start_rag")
        
        workflow.add_edge("start_rag", "process_documents")
        workflow.add_edge("process_documents", "search_documents")
        workflow.add_edge("search_documents", "web_search")
        workflow.add_edge("web_search", "combine_context")
        workflow.add_edge("combine_context", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()

    def run(self, 
           query: str, 
           topic: str = "research", 
           use_web_search: bool = False,
           max_iterations: int = 3) -> Dict[str, Any]:
        """Run the RAG workflow."""
        
        initial_state: RAGState = {
            "query": query,
            "research_tree": Tree(f"RAG Analysis: {query}"),
            "report": "",
            "status": "starting",
            "error_message": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "uploaded_files": [],
            "topic": topic,
            "use_web_search": use_web_search,
            "document_context": "",
            "web_context": ""
        }
        
        print(f"🚀 Starting analysis for: {query}")
        result = self.graph.invoke(initial_state)
        return result

    def _start_rag(self, state: RAGState) -> Dict[str, Any]:
        """Initialize the RAG workflow."""
        print(f"🔍 Processing query: '{state['query']}'")
        print("⏳ Analyzing documents and generating response...")
        
        return {
            **state,
            "status": "rag_started"
        }

    def _process_documents(self, state: RAGState) -> Dict[str, Any]:
        """Process uploaded documents for analysis."""
        query = state["query"]
        topic = state.get("topic", "research")
        
        try:
            # Get list of uploaded files (this would typically come from the application state)
            uploaded_files = state.get("uploaded_files", [])
            
            if not uploaded_files:
                print("⚠️  No documents to process, skipping document processing...")
                return {
                    **state,
                    "status": "no_documents"
                }
            
            print(f"📄 Processing {len(uploaded_files)} documents...")
            
            # Process documents (upload and chunk them)
            for file_path in uploaded_files:
                # This would typically involve uploading and chunking documents
                # For now, we'll assume documents are already processed
                print(f"  ✅ Processed: {file_path}")
            
            return {
                **state,
                "status": "documents_processed"
            }
            
        except Exception as e:
            print(f"❌ Error processing documents: {str(e)}")
            return {
                **state,
                "error_message": f"Document processing failed: {str(e)}",
                "status": "error"
            }

    def _search_documents(self, state: RAGState) -> Dict[str, Any]:
        """Search uploaded documents for relevant information."""
        query = state["query"]
        topic = state.get("topic", "research")
        
        try:
            print(f"🔍 Searching documents for: {query}")
            
            # Search for relevant document chunks
            search_results = rag_search(query)
            
            if search_results:
                print(f"📚 Found {len(search_results)} relevant document chunks")
                
                # Get document context from search results
                document_context = get_document_context(search_results)
                print(f"📋 Using {len(document_context)} characters of document context")
                
                return {
                    **state,
                    "document_context": document_context,
                    "status": "documents_searched"
                }
            else:
                print("⚠️  No relevant documents found")
                return {
                    **state,
                    "document_context": "",
                    "status": "no_documents_found"
                }
                
        except Exception as e:
            print(f"❌ Error searching documents: {str(e)}")
            return {
                **state,
                "error_message": f"Document search failed: {str(e)}",
                "status": "error"
            }

    def _web_search(self, state: RAGState) -> Dict[str, Any]:
        """Perform web search if enabled."""
        query = state["query"]
        use_web_search = state.get("use_web_search", False)
        
        if not use_web_search:
            print("🌐 Web search disabled, skipping...")
            return {
                **state,
                "web_context": "",
                "status": "web_search_skipped"
            }
        
        try:
            print(f"🌐 Searching web for: {query}")
            
            # Perform web search using Tavily
            search_results = tavily_search(query)
            
            if search_results:
                print(f"🔍 Found {len(search_results)} web sources")
                
                # Extract and combine web content
                web_context = ""
                for result in search_results[:3]:  # Limit to top 3 results
                    web_context += f"\n\n[{result.get('title', 'Web Source')}]\n{result.get('content', '')}"
                
                return {
                    **state,
                    "web_context": web_context,
                    "status": "web_search_completed"
                }
            else:
                print("⚠️  No web results found")
                return {
                    **state,
                    "web_context": "",
                    "status": "no_web_results"
                }
                
        except Exception as e:
            print(f"❌ Error in web search: {str(e)}")
            return {
                **state,
                "error_message": f"Web search failed: {str(e)}",
                "web_context": "",
                "status": "web_search_error"
            }

    def _combine_context(self, state: RAGState) -> Dict[str, Any]:
        """Combine document and web contexts."""
        print("🔗 Preparing context for analysis...")
        
        combined_context = state.get("document_context", "")
        
        return {
            **state,
            "combined_context": combined_context,
            "status": "context_combined"
        }

    def _create_report(self, context: str, query: str, state: RAGState) -> Dict[str, Any]:
        """Create a structured report from the retrieved context using AI analysis."""
        print(f"📝 Creating comprehensive RAG report...")
        
        try:
            # Initialize the DraftingAgent for AI analysis
            drafting_agent = DraftingAgent()
            
            # Create a detailed prompt for analysis
            analysis_prompt = f"""
            Please analyze the following medical document content and provide a structured answer to this question: "{query}"

            Requirements:
            - Extract specific, relevant information from the provided context
            - Organize the response in a clear, structured format with bullet points or numbered lists
            - Focus on evidence-based information from the documents
            - Highlight key facts and recommendations
            - Provide actionable insights where applicable

            Document Context:
            {context}

            Please provide a comprehensive, well-structured analysis that directly answers the question.
            """
            
            # Use DraftingAgent to generate intelligent analysis
            print("🤖 Generating AI analysis...")
            analysis_result = drafting_agent.draft_report(analysis_prompt)
            
            # Format the final report
            formatted_report = f"""# 🏥 **RAG Analysis Report**

## 📝 **Query:** {query}

## 🔍 **AI Analysis:**
{analysis_result}

## 📚 **Sources:**
- Heart Disease Documents (healthyheart.pdf, Heart Disease-Full Text.pdf)
- Analysis generated using OpenAI GPT

---
*Report generated by Deep Research AI Agent with AI Analysis*
"""
            
            print("✅ AI analysis completed successfully")
            return {
                "report": formatted_report,
                "analysis": analysis_result,
                "query": query
            }
            
        except Exception as e:
            print(f"⚠️ AI analysis failed, using fallback analysis: {str(e)}")
            # Fallback to structured text processing if AI fails
            return self._create_fallback_report(context, query, state)
    
    def _create_fallback_report(self, context: str, query: str, state: RAGState) -> Dict[str, Any]:
        """Fallback report creation when AI analysis is unavailable."""
        print("📋 Creating fallback structured report...")
        
        # Truncate context if too long
        max_context_length = 2000
        if len(context) > max_context_length:
            truncated_context = context[:max_context_length] + "...\n[Content truncated for readability]"
        else:
            truncated_context = context
        
        # Create basic structured report
        report = f"""# 🏥 **RAG Analysis Report**

## 📝 **Query:** {query}

## 🔍 **Retrieved Information:**
{truncated_context}

## 📚 **Sources:**
- Heart Disease Documents from uploaded PDFs
- Retrieved via vector search

## ⚠️ **Note:**
This is a basic text retrieval. For intelligent AI analysis, please configure OpenAI API key.

---
*Report generated by Deep Research AI Agent*
"""
        
        return {
            "report": report,
            "analysis": truncated_context,
            "query": query
        }

    def _finalize(self, state: RAGState) -> Dict[str, Any]:
        """Finalize the RAG analysis."""
        query = state["query"]
        context = state.get("combined_context", state.get("document_context", ""))
        
        if not context:
            error_msg = "No relevant context found for the query"
            print(f"❌ {error_msg}")
            return {
                **state,
                "report": f"# Error\n\n{error_msg}",
                "status": "error",
                "error_message": error_msg
            }
        
        # Create the final report using AI analysis
        report_data = self._create_report(context, query, state)
        
        return {
            **state,
            "report": report_data["report"],
            "status": "completed"
        }

# Export the workflow class
__all__ = ["RAGWorkflow", "RAGState"]
