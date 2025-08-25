#!/usr/bin/env python3
"""
Generic RAG Query Script with File Output
Save any RAG analysis to a text file instead of just terminal output
"""

import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.workflows.rag_workflow import RAGWorkflow

def save_rag_analysis_to_file(query: str, filename_prefix: str = "rag_analysis"):
    """
    Run RAG analysis and save output to a text file
    
    Args:
        query: The question to ask
        filename_prefix: Prefix for the output filename
    """
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean filename prefix (remove special characters)
    clean_prefix = "".join(c for c in filename_prefix if c.isalnum() or c in ['_', '-'])
    output_file = f"{clean_prefix}_{timestamp}.txt"
    
    print(f"🧪 Testing Query: '{query}'")
    print(f"📁 Output will be saved to: {output_file}")
    print("=" * 70)
    
    try:
        # Initialize workflow
        rag_workflow = RAGWorkflow()
        print("🚀 Initializing RAG workflow...")
        
        print(f"🔍 Processing query: {query}")
        
        # Execute RAG workflow
        result = rag_workflow.run(
            query=query,
            topic="",
            use_web_search=False
        )
        
        print("\n✅ RAG Workflow Results:")
        print("-" * 50)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Report available: {'Yes' if result.get('report') else 'No'}")
        
        if result.get("report"):
            report = result["report"]
            print(f"Report length: {len(report)} characters")
            
            # Save to text file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("RAG ANALYSIS REPORT\n")
                f.write("=" * 70 + "\n")
                f.write(f"Query: {query}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Status: {result.get('status', 'unknown')}\n")
                f.write(f"Report Length: {len(report)} characters\n")
                f.write("=" * 70 + "\n\n")
                f.write(report)
                f.write("\n\n" + "=" * 70 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 70 + "\n")
            
            print(f"\n✅ Report saved successfully to: {output_file}")
            print(f"📊 File size: {os.path.getsize(output_file)} bytes")
            
            # Show preview of saved content
            print("\n📄 File Preview (first 500 characters):")
            print("-" * 50)
            with open(output_file, 'r', encoding='utf-8') as f:
                preview = f.read(500)
                print(preview)
                if len(preview) == 500:
                    print("... (truncated)")
                    
            return output_file
                    
        else:
            error_msg = "❌ No report generated"
            print(error_msg)
            
            # Save error to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("RAG ANALYSIS ERROR REPORT\n")
                f.write("=" * 70 + "\n")
                f.write(f"Query: {query}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Status: {result.get('status', 'unknown')}\n")
                f.write("=" * 70 + "\n\n")
                f.write(error_msg + "\n")
                f.write(f"Full result: {result}\n")
            
            print(f"❌ Error details saved to: {output_file}")
            return output_file
            
    except Exception as e:
        error_msg = f"❌ Error during analysis: {e}"
        print(error_msg)
        
        # Save exception to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("RAG ANALYSIS EXCEPTION REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Query: {query}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write(error_msg + "\n\n")
            
            import traceback
            f.write("FULL TRACEBACK:\n")
            f.write("-" * 30 + "\n")
            f.write(traceback.format_exc())
        
        print(f"❌ Exception details saved to: {output_file}")
        return output_file

def main():
    """Main function - modify the query here to test different questions"""
    
    # Example questions from EXAMPLE_QUESTIONS.md
    example_queries = [
        "What are the main risk factors for heart disease?",
        "How can I prevent heart disease?", 
        "What are the symptoms of heart disease?",
        "What causes heart disease?",
        "What lifestyle changes help reduce heart disease risk?",
        "How does exercise affect heart health?",
        "What dietary changes can improve heart health?",
        "How does smoking affect the heart?",
        "What foods are good for heart health?",
        "How much exercise is recommended for heart health?"
    ]
    
    print("🔍 Available Example Queries:")
    for i, query in enumerate(example_queries, 1):
        print(f"  {i}. {query}")
    
    print("\n📝 Current Query Configuration:")
    
    # MODIFY THIS LINE TO TEST DIFFERENT QUESTIONS
    selected_query = "What lifestyle changes help reduce heart disease risk?"
    
    print(f"📋 Selected: {selected_query}")
    print("\n" + "=" * 70)
    
    # Run the analysis and save to file
    output_file = save_rag_analysis_to_file(
        query=selected_query,
        filename_prefix="lifestyle_changes"
    )
    
    print(f"\n🎉 Analysis complete! Check the file: {output_file}")

if __name__ == "__main__":
    main()
