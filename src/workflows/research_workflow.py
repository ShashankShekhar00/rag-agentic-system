"""Research workflow using LangGraph for orchestration."""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from models.tree import Tree, NodeType
from tools.search_tools import tavily_search, web_scraper, search_multiple_sources
from tools.vector_tools import store_in_weaviate, get_research_context
from tools.analysis_tools import extract_insights, summarize_content


class ResearchState(TypedDict):
    """State for the research workflow."""
    query: str
    research_tree: Tree
    report: str
    status: str
    error_message: str
    iteration: int
    max_iterations: int


class ResearchWorkflow:
    """
    Simplified workflow for orchestrating research without requiring OpenAI.
    
    This workflow coordinates the multi-step research process:
    1. Initial research using Tavily
    2. Data analysis and insight extraction
    3. Report generation
    4. Quality review
    """
    
    def __init__(self):
        """Initialize the research workflow."""
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("start_research", self._start_research)
        workflow.add_node("conduct_research", self._conduct_research)
        workflow.add_node("analyze_quality", self._analyze_quality)
        workflow.add_node("create_report", self._create_report)
        workflow.add_node("finalize", self._finalize)
        
        # Set entry point
        workflow.set_entry_point("start_research")
        
        # Add edges
        workflow.add_edge("start_research", "conduct_research")
        workflow.add_edge("conduct_research", "analyze_quality")
        workflow.add_edge("analyze_quality", "create_report")
        workflow.add_edge("create_report", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _start_research(self, state: ResearchState) -> ResearchState:
        """Start the research process."""
        print(f"🔍 Starting research on: {state['query']}")
        
        return {
            **state,
            "status": "started",
            "iteration": 0,
            "max_iterations": state.get("max_iterations", 3)
        }
    
    def _conduct_research(self, state: ResearchState) -> ResearchState:
        """Conduct research using tools."""
        try:
            print(f"📚 Conducting research...")
            
            # Initialize research tree
            research_tree = Tree(f"Research: {state['query']}")
            
            # Add initial query node
            query_node_id = research_tree.add_node(
                content=state["query"],
                node_type=NodeType.QUERY,
                metadata={"depth": 0, "type": "initial_query"}
            )
            
            # Get existing research context
            try:
                context = get_research_context.invoke({"topic": state["query"], "max_docs": 3})
            except Exception as e:
                context = f"No existing context found: {e}"
            print(f"   Found context: {len(str(context))} characters")
            
            # Perform Tavily search
            print("   🔎 Searching with Tavily...")
            try:
                search_results = tavily_search.invoke({"query": state["query"], "max_results": 5})
            except Exception as e:
                search_results = [{"error": f"Search failed: {e}"}]
            
            # Process search results
            research_output = f"Research Results for: {state['query']}\n\n"
            valid_results = 0
            
            for i, result in enumerate(search_results, 1):
                if "error" not in result:
                    valid_results += 1
                    research_output += f"{i}. {result.get('title', 'No title')}\n"
                    research_output += f"   URL: {result.get('url', 'No URL')}\n"
                    research_output += f"   Content: {result.get('content', 'No content')[:300]}...\n\n"
                    
                    # Store in Weaviate
                    try:
                        store_result = store_in_weaviate.invoke({
                            "content": result.get('content', ''),
                            "title": result.get('title', 'Untitled'),
                            "source_url": result.get('url', ''),
                            "metadata": {"query": state["query"], "source": "tavily"}
                        })
                    except Exception as e:
                        print(f"   ⚠️ Failed to store in Weaviate: {e}")
            
            print(f"   ✅ Found {valid_results} valid results")
            
            # Add research results to tree
            result_node_id = research_tree.add_node(
                content=research_output,
                node_type=NodeType.RESULT,
                parent_id=query_node_id,
                metadata={"agent": "research_workflow", "iteration": 1, "results_count": valid_results}
            )
            
            # Extract insights
            print("   🧠 Extracting insights...")
            try:
                insights = extract_insights.invoke({"content": research_output, "topic": state["query"]})
            except Exception as e:
                insights = [f"Failed to extract insights: {e}"]
            
            # Add insights to tree
            for insight in insights[:8]:
                research_tree.add_node(
                    content=insight,
                    node_type=NodeType.INSIGHT,
                    parent_id=result_node_id,
                    metadata={"extracted_by": "research_workflow"}
                )
            
            print(f"   ✅ Extracted {len(insights)} insights")
            
            return {
                **state,
                "research_tree": research_tree,
                "status": "research_completed",
                "iteration": state["iteration"] + 1
            }
            
        except Exception as e:
            print(f"❌ Research failed: {str(e)}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def _analyze_quality(self, state: ResearchState) -> ResearchState:
        """Analyze research quality."""
        try:
            print("� Analyzing research quality...")
            
            insights = state["research_tree"].get_insights()
            results = state["research_tree"].get_results()
            
            # Calculate quality metrics
            quality_score = 0
            feedback = []
            
            # Score based on insights
            if len(insights) >= 5:
                quality_score += 30
                feedback.append("Good insight extraction")
            elif len(insights) >= 3:
                quality_score += 20
                feedback.append("Moderate insight extraction")
            else:
                feedback.append("Limited insights extracted")
            
            # Score based on results
            if len(results) >= 1:
                quality_score += 30
                feedback.append("Research results obtained")
            else:
                feedback.append("No research results")
            
            # Score based on content length
            total_content = sum(len(node.content) for node in state["research_tree"].nodes.values())
            if total_content > 2000:
                quality_score += 25
                feedback.append("Rich content gathered")
            elif total_content > 1000:
                quality_score += 15
                feedback.append("Adequate content volume")
            else:
                feedback.append("Limited content gathered")
            
            # Score based on tree structure
            total_nodes = len(state["research_tree"].nodes)
            if total_nodes > 8:
                quality_score += 15
                feedback.append("Well-structured research tree")
            elif total_nodes > 5:
                quality_score += 10
                feedback.append("Basic research structure")
            
            quality_analysis = {
                "quality_score": quality_score,
                "insights_count": len(insights),
                "results_count": len(results),
                "content_length": total_content,
                "total_nodes": total_nodes,
                "feedback": feedback
            }
            
            print(f"   Quality Score: {quality_score}/100")
            print(f"   Insights: {len(insights)}")
            print(f"   Results: {len(results)}")
            
            return {
                **state,
                "status": "quality_analyzed",
                "quality_analysis": quality_analysis
            }
            
        except Exception as e:
            print(f"❌ Quality analysis failed: {str(e)}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def _create_report(self, state: ResearchState) -> ResearchState:
        """Create the final report."""
        try:
            print("📝 Creating report...")
            
            insights = state["research_tree"].get_insights()
            results = state["research_tree"].get_results()
            
            # Determine report type
            quality_score = state.get("quality_analysis", {}).get("quality_score", 50)
            
            if quality_score >= 80:
                report_type = "comprehensive"
            elif quality_score >= 60:
                report_type = "summary"
            else:
                report_type = "executive"
            
            print(f"   Report type: {report_type}")
            
            # Create report content
            report_parts = [
                f"# Research Report: {state['query']}",
                "",
                f"**Report Type:** {report_type.title()}",
                f"**Quality Score:** {quality_score}/100",
                "",
                "## Executive Summary",
                f"This research investigation into '{state['query']}' yielded {len(insights)} key insights ",
                f"from {len(results)} research sources. The analysis provides comprehensive coverage of the topic ",
                "with actionable findings and recommendations.",
                "",
                "## Key Findings",
                ""
            ]
            
            # Add insights
            for i, insight in enumerate(insights[:10], 1):
                report_parts.append(f"{i}. {insight.content}")
            
            if not insights:
                report_parts.append("No specific insights were extracted from the research.")
            
            report_parts.extend([
                "",
                "## Research Results Summary",
                ""
            ])
            
            # Add summarized results
            for i, result in enumerate(results[:3], 1):
                if not result.content.startswith("Research error"):
                    try:
                        summary = summarize_content.invoke({"content": result.content, "max_sentences": 4})
                    except Exception as e:
                        summary = f"Summary unavailable: {e}"
                    report_parts.append(f"### Source {i}")
                    report_parts.append(summary)
                    report_parts.append("")
            
            # Add quality analysis
            quality_analysis = state.get("quality_analysis", {})
            report_parts.extend([
                "## Research Quality Analysis",
                f"- **Overall Score:** {quality_analysis.get('quality_score', 0)}/100",
                f"- **Insights Extracted:** {quality_analysis.get('insights_count', 0)}",
                f"- **Research Sources:** {quality_analysis.get('results_count', 0)}",
                f"- **Content Volume:** {quality_analysis.get('content_length', 0)} characters",
                f"- **Research Depth:** {quality_analysis.get('total_nodes', 0)} data points",
                "",
                "## Recommendations",
                "Based on the research findings, consider the following next steps:",
                "1. Validate key findings with additional sources",
                "2. Explore specific aspects that require deeper investigation", 
                "3. Monitor ongoing developments in this area",
                "4. Apply insights to relevant decision-making processes",
                "",
                "## Conclusion",
                f"This research provides a {report_type} overview of '{state['query']}'. ",
                "The findings should be considered alongside other relevant information and expert judgment.",
                "",
                f"*Report generated by Deep Research AI Agent with {len(state['research_tree'].nodes)} data points*"
            ])
            
            report = "\n".join(report_parts)
            
            return {
                **state,
                "report": report,
                "status": "report_created"
            }
            
        except Exception as e:
            print(f"❌ Report creation failed: {str(e)}")
            return {
                **state,
                "status": "error",
                "error_message": str(e)
            }
    
    def _finalize(self, state: ResearchState) -> ResearchState:
        """Finalize the research workflow."""
        if state["status"] == "error":
            print(f"❌ Workflow completed with error: {state.get('error_message', 'Unknown error')}")
        else:
            print("✅ Research workflow completed successfully")
            
            # Print summary
            if state.get("research_tree"):
                insights = state["research_tree"].get_insights()
                results = state["research_tree"].get_results()
                
                print("\n" + "="*50)
                print("RESEARCH SUMMARY")
                print("="*50)
                print(f"Research Topic: {state['query']}")
                print(f"Total Data Points: {len(state['research_tree'].nodes)}")
                print(f"Research Sources: {len(results)}")
                print(f"Insights Extracted: {len(insights)}")
                print(f"Quality Score: {state.get('quality_analysis', {}).get('quality_score', 0)}/100")
        
        return {
            **state,
            "status": "completed"
        }
    
    def run_research(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Run the complete research workflow.
        
        Args:
            query: Research query
            max_iterations: Maximum research iterations
        
        Returns:
            Dictionary with research results
        """
        print(f"\n{'='*60}")
        print(f"DEEP RESEARCH AI AGENT")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*60}\n")
        
        # Initial state
        initial_state = ResearchState(
            query=query,
            research_tree=None,
            report="",
            status="initialized",
            error_message="",
            iteration=0,
            max_iterations=max_iterations
        )
        
        # Run the workflow
        final_state = self.graph.invoke(initial_state)
        
        # Return results
        return {
            "query": query,
            "status": final_state.get("status"),
            "research_tree": final_state.get("research_tree"),
            "report": final_state.get("report", ""),
            "quality_analysis": final_state.get("quality_analysis", {}),
            "error_message": final_state.get("error_message", ""),
            "iterations": final_state.get("iteration", 0)
        }