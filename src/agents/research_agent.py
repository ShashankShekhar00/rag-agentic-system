"""Research Agent for web search and information gathering."""

from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, AIMessage
from langchain.base_language import BaseLanguageModel
from models.tree import Tree, NodeType
from tools.search_tools import tavily_search, web_scraper, search_multiple_sources
from tools.vector_tools import store_in_weaviate, get_research_context
from tools.analysis_tools import extract_insights


class MockLLM(BaseLanguageModel):
    """Mock LLM for demo purposes when OpenAI API key is not available."""
    
    def _generate(self, messages, stop=None, run_manager=None):
        return AIMessage(content="Mock response: Research agent is working on your query...")
    
    def _llm_type(self):
        return "mock"
    
    @property
    def _identifying_params(self):
        return {}


class ResearchAgent:
    """
    Research Agent that searches, gathers, and curates information from the web.
    
    This agent uses Tavily's search capabilities and other integrated tools to
    perform comprehensive research on any given topic.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """Initialize the Research Agent."""
        from config.settings import OPENAI_API_KEY
        
        if not OPENAI_API_KEY:
            print("⚠️  Warning: OpenAI API key not set. Using mock LLM for demo.")
            # Create a simple mock LLM for demo purposes
            self.llm = MockLLM()
        else:
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
            
        self.tools = [
            tavily_search,
            web_scraper, 
            search_multiple_sources,
            store_in_weaviate,
            get_research_context,
            extract_insights
        ]
        self.tree = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the agent with tools and prompt."""
        from config.settings import OPENAI_API_KEY
        
        if not OPENAI_API_KEY:
            # For demo without OpenAI, use simple tool-based approach
            self.agent_executor = None
            return
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Research Agent specialized in comprehensive information gathering.
            
            Your capabilities include:
            - Searching the web using Tavily API for high-quality, recent information
            - Scraping content from specific URLs when needed
            - Storing research findings in a vector database for future reference
            - Extracting key insights from gathered information
            - Building a structured research tree to organize findings
            
            Guidelines:
            1. Always search for multiple perspectives on a topic
            2. Verify information from multiple sources when possible  
            3. Store important findings in the vector database
            4. Extract and highlight key insights
            5. Organize information in a logical, hierarchical structure
            6. Be thorough but efficient in your research approach
            
            When given a research query, break it down into sub-queries if needed and gather
            comprehensive information from multiple angles."""),
            
            ("human", "{input}"),
            ("assistant", "I'll help you research this topic comprehensively. Let me start by searching for information and gathering relevant data."),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create the agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def research(self, query: str, max_depth: int = 3) -> Tree:
        """
        Conduct research on a given query.
        
        Args:
            query: The research query/topic
            max_depth: Maximum depth of research (number of iterations)
        
        Returns:
            Tree: Research tree with organized findings
        """
        # Initialize research tree
        self.tree = Tree(f"Research: {query}")
        
        # Add initial query node
        query_node_id = self.tree.add_node(
            content=query,
            node_type=NodeType.QUERY,
            metadata={"depth": 0, "type": "initial_query"}
        )
        
        try:
            # Get existing research context
            context = get_research_context(query)
            
            # If no agent executor available (no OpenAI key), use direct tool approach
            if not self.agent_executor:
                # Direct tool usage for demo
                search_results = tavily_search(query, max_results=3)
                
                # Process search results
                research_output = f"Research Results for: {query}\n\n"
                for i, result in enumerate(search_results, 1):
                    if "error" not in result:
                        research_output += f"{i}. {result.get('title', 'No title')}\n"
                        research_output += f"   {result.get('content', 'No content')[:200]}...\n\n"
                
                # Add research results to tree
                result_node_id = self.tree.add_node(
                    content=research_output,
                    node_type=NodeType.RESULT,
                    parent_id=query_node_id,
                    metadata={"agent": "research_agent", "iteration": 1, "mode": "direct"}
                )
                
                # Extract insights
                insights = extract_insights(research_output, query)
                
                # Add insights to tree
                for insight in insights[:5]:
                    self.tree.add_node(
                        content=insight,
                        node_type=NodeType.INSIGHT,
                        parent_id=result_node_id,
                        metadata={"extracted_by": "research_agent"}
                    )
                
                return self.tree
            
            # Perform initial research
            research_prompt = f"""
            Research the following topic comprehensively: {query}
            
            Previous research context:
            {context}
            
            Please:
            1. Search for current information on this topic
            2. Gather data from multiple reliable sources
            3. Extract key insights and findings
            4. Store important information for future reference
            5. Identify any gaps that need further investigation
            
            Be thorough and systematic in your approach.
            """
            
            # Execute research
            result = self.agent_executor.invoke({"input": research_prompt})
            
            # Add research results to tree
            if result and "output" in result:
                result_node_id = self.tree.add_node(
                    content=result["output"],
                    node_type=NodeType.RESULT,
                    parent_id=query_node_id,
                    metadata={"agent": "research_agent", "iteration": 1}
                )
                
                # Extract insights from the results
                insights = extract_insights(result["output"], query)
                
                # Add insights to tree
                for insight in insights[:5]:  # Limit to top 5 insights
                    self.tree.add_node(
                        content=insight,
                        node_type=NodeType.INSIGHT,
                        parent_id=result_node_id,
                        metadata={"extracted_by": "research_agent"}
                    )
            
            return self.tree
            
        except Exception as e:
            # Add error node to tree
            self.tree.add_node(
                content=f"Research error: {str(e)}",
                node_type=NodeType.RESULT,
                parent_id=query_node_id,
                metadata={"error": True, "agent": "research_agent"}
            )
            return self.tree
    
    def deep_research(self, query: str, follow_up_questions: List[str] = None) -> Tree:
        """
        Conduct deep research with follow-up questions.
        
        Args:
            query: Main research query
            follow_up_questions: Additional questions to research
        
        Returns:
            Tree: Comprehensive research tree
        """
        # Start with initial research
        tree = self.research(query)
        
        # Research follow-up questions if provided
        if follow_up_questions:
            for i, follow_up in enumerate(follow_up_questions):
                follow_up_tree = self.research(follow_up)
                
                # Add follow-up results to main tree
                follow_up_node_id = tree.add_node(
                    content=follow_up,
                    node_type=NodeType.QUERY,
                    metadata={"type": "follow_up", "index": i}
                )
                
                # Copy insights from follow-up tree
                for insight_node in follow_up_tree.get_insights():
                    tree.add_node(
                        content=insight_node.content,
                        node_type=NodeType.INSIGHT,
                        parent_id=follow_up_node_id,
                        metadata=insight_node.metadata
                    )
        
        return tree
    
    def get_research_summary(self) -> str:
        """Get a summary of the current research session."""
        if not self.tree:
            return "No research conducted yet."
        
        insights = self.tree.get_insights()
        results = self.tree.get_results()
        
        summary_parts = [
            f"Research Summary ({len(self.tree.nodes)} total nodes)",
            f"- Queries processed: {len([n for n in self.tree.nodes.values() if n.type == NodeType.QUERY])}",
            f"- Results gathered: {len(results)}",
            f"- Insights extracted: {len(insights)}",
            "",
            "Key Insights:"
        ]
        
        for i, insight in enumerate(insights[:10], 1):
            summary_parts.append(f"{i}. {insight.content}")
        
        return "\n".join(summary_parts)
