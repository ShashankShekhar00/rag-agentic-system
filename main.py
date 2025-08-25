"""
Deep Research AI Agentic System with RAG Capabilities

A comprehensive AI research system that combines:
- Document-based RAG (Retrieval-Augmented Generation)  
- Web research using Tavily API
- LangGraph workflows for intelligent processing
- Vector database storage with Weaviate

Author: Deep Research AI Team
Version: 1.0.0
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.workflows.research_workflow import ResearchWorkflow
from src.workflows.rag_workflow import RAGWorkflow
from src.tools.rag_tools import list_uploaded_documents
from src.tools.document_tools import upload_documents
from src.config.settings import TAVILY_API_KEY, WEAVIATE_URL


class DeepResearchAI:
    """Main application class for Deep Research AI system."""
    
    def __init__(self):
        """Initialize the Deep Research AI system."""
        self.research_workflow: Optional[ResearchWorkflow] = None
        self.rag_workflow: Optional[RAGWorkflow] = None
        self._initialize_workflows()
    
    def _initialize_workflows(self) -> None:
        """Initialize research and RAG workflows."""
        try:
            self.research_workflow = ResearchWorkflow()
            self.rag_workflow = RAGWorkflow()
            print("✅ Research and RAG workflows initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize workflows: {e}")
            raise
    
    def display_header(self) -> None:
        """Display application header and configuration status."""
        print("🤖 Deep Research AI System with RAG Capabilities")
        print("=" * 60)
        
        # Configuration status
        print(f"📊 Weaviate URL: {WEAVIATE_URL}")
        print(f"🔑 Tavily API: {'Configured' if TAVILY_API_KEY else 'Not configured'}")
        
        if not TAVILY_API_KEY:
            print("⚠️  Warning: TAVILY_API_KEY not set. Web research may not work.")
        
        print()
    
    def show_uploaded_documents(self) -> None:
        """Display information about uploaded documents."""
        try:
            documents = list_uploaded_documents.invoke({"topic": ""})
            
            if documents and len(documents) > 0 and "error" not in documents[0]:
                print(f"📚 Uploaded Documents ({len(documents)}):")
                for i, doc in enumerate(documents, 1):
                    topic = doc.get('topic', 'general')
                    chunk_count = doc.get('total_chunks', 0)
                    print(f"  {i}. {doc['file_name']} (Topic: {topic}, Chunks: {chunk_count})")
            else:
                print("📚 No documents uploaded yet")
        except Exception as e:
            print(f"📚 Documents status: Unable to check ({str(e)[:50]}...)")
        
        print()
    
    def _show_main_menu(self) -> str:
        """Display main menu and return user choice."""
        print("=" * 60)
        print("🎯 SELECT MODE:")
        print("1. 📄 RAG Mode - Ask questions about uploaded documents")
        print("2. 🔍 Research Mode - Web-based research")
        print("3. 🔀 Hybrid Mode - Combine documents + web research")
        print("4. 📁 Document Management")
        print("5. 💡 Examples & Help")
        print("6. 🚪 Exit")
        print("=" * 60)
        
        return input("\nSelect mode (1-6): ").strip()
    
    def _rag_mode(self) -> None:
        """RAG mode for document-based Q&A."""
        print("\n📄 RAG Mode - Document-based Question Answering")
        print("-" * 50)
        
        # Check if documents exist
        try:
            docs = list_uploaded_documents.invoke({"topic": ""})
            if not docs or len(docs) == 0 or "error" in docs[0]:
                print("⚠️  No documents found. Please upload documents first in Document Management mode.")
                return
        except Exception as e:
            print(f"⚠️  Could not check documents: {e}")
            return
        
        print("Available documents:")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc['file_name']} (Topic: {doc.get('topic', 'General')})")
        
        # Get query
        query = input("\n❓ Enter your question about the uploaded documents: ").strip()
        if not query:
            print("❌ No query provided!")
            return
        
        # Optional topic filter
        topic = input("🏷️  Filter by topic (press Enter for all documents): ").strip()
        
        print(f"\n🔍 Processing query: '{query}'")
        if topic:
            print(f"🏷️  Filtering by topic: '{topic}'")
        
        print("⏳ Analyzing documents and generating response...")
        
        try:
            # Run RAG workflow
            result = self.rag_workflow.run(query, topic if topic else "")
            
            if result.get("status") == "completed" and result.get("report"):
                # Save report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rag_analysis_{timestamp}.md"
                filepath = Path("reports") / filename
                
                # Create reports directory if it doesn't exist
                filepath.parent.mkdir(exist_ok=True)
                
                # Save the report
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result["report"])
                
                print(f"\n✅ Analysis completed!")
                print(f"📋 Report saved to: {filepath}")
                print(f"📊 Report length: {len(result['report'])} characters")
                
                # Show preview
                print(f"\n📄 Report Preview:")
                print("-" * 40)
                lines = result["report"].split('\n')[:15]
                for line in lines:
                    print(line)
                if len(result["report"].split('\n')) > 15:
                    print("... (truncated)")
                    
            else:
                error_msg = result.get("error_message", "Unknown error occurred")
                print(f"❌ Analysis failed: {error_msg}")
                print("Please try again or check your query.")
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
    
    def _research_mode(self) -> None:
        """Research mode for web-based research."""
        print("\n🔍 Research Mode - Web-based Research")
        print("-" * 40)
        
        if not TAVILY_API_KEY:
            print("❌ Tavily API key not configured. Web research not available.")
            return
        
        query = input("❓ Enter your research question: ").strip()
        if not query:
            print("❌ No query provided!")
            return
        
        print(f"\n🔍 Researching: '{query}'")
        print("⏳ Searching web sources and analyzing...")
        
        try:
            result = self.research_workflow.run(query)
            
            if result.get("success"):
                # Save report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"research_report_{timestamp}.md"
                filepath = Path("reports") / filename
                
                filepath.parent.mkdir(exist_ok=True)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result["report"])
                
                print(f"\n✅ Research completed!")
                print(f"📋 Report saved to: {filepath}")
                print(f"📊 Report length: {len(result['report'])} characters")
                
                # Show preview
                print(f"\n📄 Report Preview:")
                print("-" * 40)
                lines = result["report"].split('\n')[:15]
                for line in lines:
                    print(line)
                if len(result["report"].split('\n')) > 15:
                    print("... (truncated)")
                    
            else:
                print("❌ Research failed. Please try again.")
                
        except Exception as e:
            print(f"❌ Error during research: {e}")
    
    def _hybrid_mode(self) -> None:
        """Hybrid mode combining documents and web research."""
        print("\n🔀 Hybrid Mode - Documents + Web Research")
        print("-" * 45)
        print("This mode combines your uploaded documents with web research for comprehensive analysis.")
        print("Note: Hybrid mode is currently under development.")
        print("Please use RAG mode for documents or Research mode for web research.")
    
    def _document_management(self) -> None:
        """Document management mode."""
        print("\n📁 Document Management")
        print("-" * 25)
        
        while True:
            print("\n📁 Document Management Options:")
            print("1. 📤 Upload Document")
            print("2. 📋 List Documents")
            print("3. 🔙 Back to Main Menu")
            
            choice = input("Select option (1-3): ").strip()
            
            if choice == "1":
                self._upload_document()
            elif choice == "2":
                self._list_documents()
            elif choice == "3":
                break
            else:
                print("❌ Invalid choice. Please select 1-3.")
    
    def _upload_document(self) -> None:
        """Upload a document to the system."""
        file_path = input("📄 Enter file path: ").strip()
        if not file_path:
            print("❌ No file path provided!")
            return
        
        if not Path(file_path).exists():
            print("❌ File not found!")
            return
        
        topic = input("🏷️  Enter topic (optional): ").strip() or "general"
        
        print(f"⏳ Uploading and processing: {file_path}")
        
        try:
            result = upload_documents.invoke({
                "file_paths": [file_path],
                "topic": topic
            })
            
            if "error" not in result:
                print(f"✅ Document uploaded successfully!")
                print(f"📄 File: {result.get('file_name', file_path)}")
                print(f"🏷️  Topic: {result.get('topic', topic)}")
                print(f"📊 Chunks created: {result.get('chunks_created', 'Unknown')}")
            else:
                print(f"❌ Upload failed: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error uploading document: {e}")
    
    def _list_documents(self) -> None:
        """List all uploaded documents."""
        try:
            documents = list_uploaded_documents.invoke({"topic": ""})
            
            if documents and len(documents) > 0 and "error" not in documents[0]:
                print(f"\n📚 Uploaded Documents ({len(documents)}):")
                print("-" * 40)
                for i, doc in enumerate(documents, 1):
                    topic = doc.get('topic', 'general')
                    chunk_count = doc.get('total_chunks', 0)
                    print(f"{i:2d}. {doc['file_name']}")
                    print(f"    Topic: {topic}")
                    print(f"    Chunks: {chunk_count}")
                    print()
            else:
                print("📚 No documents uploaded yet")
                
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
    
    def _show_help(self) -> None:
        """Show examples and help information."""
        print("\n💡 Examples & Help")
        print("-" * 20)
        print()
        print("🔍 Example Questions for RAG Mode:")
        print("- What are the main risk factors for heart disease?")
        print("- How can exercise help prevent heart disease?")
        print("- What foods are good for heart health?")
        print("- How does smoking affect the heart?")
        print()
        print("🌐 Example Questions for Research Mode:")
        print("- Latest developments in AI technology")
        print("- Climate change impact on agriculture")
        print("- Best practices for software development")
        print()
        print("📋 Supported File Types:")
        print("- PDF documents (.pdf)")
        print("- Text files (.txt)")
        print("- More formats coming soon!")
        print()
        print("💡 Tips:")
        print("- Use specific, focused questions for better results")
        print("- Upload relevant documents before using RAG mode")
        print("- Check the reports/ folder for saved analyses")
    
    def run(self) -> None:
        """Run the main application loop."""
        self.display_header()
        self.show_uploaded_documents()
        
        while True:
            try:
                choice = self._show_main_menu()
                
                if choice == '1':
                    self._rag_mode()
                elif choice == '2':
                    self._research_mode()
                elif choice == '3':
                    self._hybrid_mode()
                elif choice == '4':
                    self._document_management()
                elif choice == '5':
                    self._show_help()
                elif choice == '6':
                    print("👋 Thank you for using Deep Research AI!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                print("Please try again.")


def main():
    """Main entry point for the application."""
    try:
        app = DeepResearchAI()
        app.run()
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
