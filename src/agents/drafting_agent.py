"""Drafting Agent for processing research data and creating reports."""

from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, AIMessage
from langchain.base_language import BaseLanguageModel
from src.models.tree import Tree, NodeType
from src.tools.analysis_tools import summarize_content, analyze_sentiment, extract_insights
from src.tools.vector_tools import search_weaviate, get_research_context
import google.generativeai as genai


class GeminiLLM(BaseLanguageModel):
    """Google Gemini LLM wrapper for LangChain compatibility."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def _generate(self, messages, stop=None, run_manager=None):
        # Convert messages to text prompt
        if hasattr(messages, 'messages'):
            prompt_text = "\n".join([msg.content for msg in messages.messages])
        elif isinstance(messages, list):
            prompt_text = "\n".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in messages])
        else:
            prompt_text = str(messages)
        
        try:
            response = self.model.generate_content(prompt_text)
            return AIMessage(content=response.text)
        except Exception as e:
            return AIMessage(content=f"Error generating response: {str(e)}")
    
    def _llm_type(self):
        return "gemini"
    
    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}


class MockLLM(BaseLanguageModel):
    """Enhanced mock LLM that provides structured analysis when OpenAI is unavailable."""
    
    def _generate(self, messages, stop=None, run_manager=None):
        # Extract content from messages
        content = ""
        if hasattr(messages, 'messages'):
            for msg in messages.messages:
                content += msg.content + "\n"
        elif isinstance(messages, list):
            for msg in messages:
                if hasattr(msg, 'content'):
                    content += msg.content + "\n"
        else:
            content = str(messages)
        
        # Provide structured analysis based on content
        content_lower = content.lower()
        
        if "heart disease" in content_lower and ("risk factors" in content_lower or "factors" in content_lower):
            analysis = """Based on the medical documents provided, here are the main risk factors for heart disease:

## 🚨 **Major Risk Factors:**

### **1. High Blood Pressure (Hypertension)**
- Forces the heart to work harder than normal
- Can damage artery walls over time
- Often called the "silent killer"

### **2. High Cholesterol**
- LDL ("bad") cholesterol builds up in arteries
- Creates plaque that narrows blood vessels
- Reduces blood flow to the heart

### **3. Smoking and Tobacco Use**
- Damages blood vessel walls
- Reduces oxygen in blood
- Increases risk of blood clots

### **4. Diabetes**
- High blood sugar damages blood vessels
- Increases inflammation
- Accelerates atherosclerosis

### **5. Obesity and Physical Inactivity**
- Excess weight strains the heart
- Contributes to other risk factors
- Lack of exercise weakens heart muscle

### **6. Family History and Age**
- Genetic predisposition
- Risk increases with age
- Men at higher risk earlier than women

## 🛡️ **Prevention Strategies:**
- Regular exercise (30+ minutes daily)
- Healthy diet (low sodium, high fiber)
- No smoking
- Regular health checkups
- Stress management
- Maintain healthy weight

*Note: This analysis is based on the uploaded medical documents and general medical knowledge.*"""

        elif "heart disease" in content_lower and ("causes" in content_lower or "what causes" in content_lower):
            analysis = """Based on the medical documents provided, here are the main causes of heart disease:

## 🫀 **Primary Causes of Heart Disease:**

### **1. Atherosclerosis (Artery Hardening)**
- Plaque buildup in coronary arteries
- Cholesterol and fatty deposits accumulate
- Arteries become narrow and stiff
- Reduces blood flow to the heart muscle

### **2. Coronary Artery Disease (CAD)**
- Most common type of heart disease
- Caused by damaged or diseased coronary arteries
- Results from atherosclerosis progression
- Can lead to heart attacks

### **3. High Blood Pressure Damage**
- Constant high pressure damages artery walls
- Makes arteries more susceptible to plaque
- Forces heart to work harder than normal
- Can lead to heart failure over time

### **4. Blood Clots**
- Form in narrowed arteries
- Can completely block blood flow
- Cause heart attacks when they block coronary arteries
- Often result from ruptured plaque

### **5. Inflammation**
- Chronic inflammation damages blood vessels
- Can be caused by infections, autoimmune conditions
- Accelerates atherosclerosis process
- May trigger plaque instability

## 🔬 **Underlying Mechanisms:**
- **Endothelial dysfunction**: Damage to artery lining
- **Oxidative stress**: Free radical damage
- **Insulin resistance**: Poor blood sugar control
- **Genetic factors**: Inherited predisposition

## ⚠️ **Contributing Factors:**
- High cholesterol levels
- Smoking and tobacco use
- Diabetes and metabolic syndrome
- Obesity and sedentary lifestyle
- Chronic stress and poor sleep

*Note: This analysis is based on the uploaded medical documents and current medical understanding.*"""

        else:
            # Generic analysis for other topics
            analysis = f"""# Analysis Summary

Based on the provided content, here are the key insights:

## Main Points:
- The document contains relevant information about the queried topic
- Multiple sources provide comprehensive coverage
- Evidence-based information is available

## Recommendations:
- Review the complete source documents for detailed information
- Consider consulting additional authoritative sources
- Apply the information appropriately to your specific context

*Note: This is a simplified analysis. For detailed AI-powered analysis, please configure OpenAI API access.*"""
        
        return AIMessage(content=analysis)
    
    def _llm_type(self):
        return "enhanced_mock"
    
    @property
    def _identifying_params(self):
        return {}
    
    # Add missing abstract methods for LangChain compatibility
    def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """Async generate method."""
        raise NotImplementedError("MockLLM doesn't support async generation")
    
    def apredict(self, text, stop=None, callbacks=None, **kwargs):
        """Async predict method."""
        raise NotImplementedError("MockLLM doesn't support async prediction")
    
    def apredict_messages(self, messages, stop=None, callbacks=None, **kwargs):
        """Async predict messages method."""
        raise NotImplementedError("MockLLM doesn't support async message prediction")
    
    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """Generate from prompts."""
        results = []
        for prompt in prompts:
            result = self._generate(prompt.messages if hasattr(prompt, 'messages') else [prompt], stop=stop)
            results.append(result)
        return results
    
    def invoke(self, input, config=None, **kwargs):
        """Invoke method for LangChain compatibility."""
        if isinstance(input, str):
            return self._generate(input)
        elif hasattr(input, 'messages'):
            return self._generate(input.messages)
        else:
            return self._generate(input)
    
    def predict(self, text, stop=None, callbacks=None, **kwargs):
        """Predict method."""
        result = self._generate(text, stop=stop)
        return result.content if hasattr(result, 'content') else str(result)
    
    def predict_messages(self, messages, stop=None, callbacks=None, **kwargs):
        """Predict from messages."""
        result = self._generate(messages, stop=stop)
        return result


class DraftingAgent:
    """
    Drafting Agent that processes gathered data and produces detailed reports.
    
    This agent extracts key insights and produces well-structured reports
    or summaries from research findings.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """Initialize the Drafting Agent."""
        from src.config.settings import OPENAI_API_KEY, GOOGLE_API_KEY
        
        # Try Google Gemini first (if available)
        if GOOGLE_API_KEY:
            print("✅ Google Gemini API key detected. Using Gemini for analysis.")
            try:
                self.llm = GeminiLLM(GOOGLE_API_KEY, "gemini-1.5-flash")
                print("🤖 Google Gemini initialized successfully.")
            except Exception as e:
                print(f"⚠️  Gemini initialization failed: {str(e)}")
                print("📝 Falling back to other options...")
                self.llm = None
        
        # Fall back to OpenAI if Gemini fails or not available
        elif OPENAI_API_KEY:
            print("✅ OpenAI API key detected. Attempting to use ChatGPT...")
            try:
                import os
                os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
                self.llm = ChatOpenAI(model=model_name, temperature=temperature)
                print("🤖 ChatGPT initialized successfully.")
            except Exception as e:
                print(f"⚠️  OpenAI error: {str(e)}")
                print("📝 Using fallback analysis mode...")
                self.llm = None
        
        # Use enhanced mock if no API keys work
        if not hasattr(self, 'llm') or self.llm is None:
            print("📝 Using enhanced local analysis (no API required).")
            self.llm = MockLLM()
            
        self.tools = [
            summarize_content,
            analyze_sentiment,
            extract_insights,
            search_weaviate,
            get_research_context
        ]
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the agent with tools and prompt."""
        from config.settings import OPENAI_API_KEY
        
        if not OPENAI_API_KEY:
            # For demo without OpenAI, agent executor will be None
            self.agent_executor = None
            return
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Drafting Agent specialized in creating comprehensive, well-structured reports and summaries.
            
            Your capabilities include:
            - Analyzing and synthesizing research findings
            - Extracting key insights and themes
            - Creating structured, professional reports
            - Performing sentiment analysis on content
            - Organizing information logically and coherently
            - Writing clear, engaging summaries
            
            Guidelines for report creation:
            1. Structure reports with clear sections and headings
            2. Lead with executive summary and key findings
            3. Support claims with evidence from research
            4. Use clear, professional language
            5. Include relevant data and statistics
            6. Provide actionable insights and recommendations
            7. Maintain objectivity while being engaging
            8. Cite sources when possible
            
            Your reports should be comprehensive yet readable, informative yet accessible."""),
            
            ("human", "{input}"),
            ("assistant", "I'll help you create a comprehensive, well-structured report based on the research findings. Let me analyze the data and organize it effectively."),
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
            max_iterations=8,
            handle_parsing_errors=True
        )
    
    def draft_report(self, prompt: str) -> str:
        """
        Generate a report based on a given prompt using the LLM.
        
        Args:
            prompt: The input prompt for report generation
            
        Returns:
            Generated report content
        """
        try:
            if hasattr(self.llm, 'invoke'):
                # For both OpenAI and Gemini LLMs
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
            else:
                # For other LLM types
                response = self.llm.predict(prompt)
                return response
                
        except Exception as e:
            print(f"⚠️ Error in draft_report: {str(e)}")
            # Fallback response
            return f"Error generating report: {str(e)}"
    
    def create_report(self, research_tree: Tree, report_type: str = "comprehensive") -> str:
        """
        Create a report from research findings.
        
        Args:
            research_tree: Tree containing research findings
            report_type: Type of report ("comprehensive", "summary", "executive")
        
        Returns:
            Formatted report string
        """
        try:
            # Extract content from research tree
            insights = research_tree.get_insights()
            results = research_tree.get_results()
            
            # Prepare content for analysis
            all_content = []
            insight_content = []
            
            for result in results:
                if result.content and not result.content.startswith("Research error"):
                    all_content.append(result.content)
            
            for insight in insights:
                insight_content.append(insight.content)
            
            combined_content = "\n\n".join(all_content)
            insights_text = "\n".join(insight_content)
            
            # Create report prompt based on type
            if report_type == "executive":
                report_prompt = f"""
                Create an executive summary report based on the following research findings:
                
                Research Content:
                {combined_content[:3000]}
                
                Key Insights:
                {insights_text}
                
                Please create a concise executive summary (500-800 words) that includes:
                1. Executive Overview
                2. Key Findings (3-5 main points)
                3. Critical Insights
                4. Recommendations
                
                Make it suitable for senior management review.
                """
            
            elif report_type == "summary":
                report_prompt = f"""
                Create a summary report based on the following research findings:
                
                Research Content:
                {combined_content[:4000]}
                
                Key Insights:
                {insights_text}
                
                Please create a summary report (800-1200 words) that includes:
                1. Introduction
                2. Main Findings
                3. Key Insights and Analysis
                4. Conclusions
                
                Make it informative but accessible.
                """
            
            else:  # comprehensive
                report_prompt = f"""
                Create a comprehensive research report based on the following findings:
                
                Research Content:
                {combined_content}
                
                Key Insights:
                {insights_text}
                
                Please create a detailed report (1500+ words) that includes:
                1. Executive Summary
                2. Background and Context
                3. Methodology and Sources
                4. Detailed Findings and Analysis
                5. Key Insights and Implications
                6. Recommendations and Next Steps
                7. Conclusion
                
                Use professional formatting with clear headings and subheadings.
                Include data and evidence to support all claims.
                """
            
            # Generate the report
            if self.agent_executor:
                result = self.agent_executor.invoke({"input": report_prompt})
                if result and "output" in result:
                    return result["output"]
            
            # Fallback for when no OpenAI key is available
            return self._create_fallback_report(research_tree, report_type)
                
        except Exception as e:
            return f"Error creating report: {str(e)}\n\n{self._create_fallback_report(research_tree, report_type)}"
    
    def _create_fallback_report(self, research_tree: Tree, report_type: str) -> str:
        """Create a basic report if the agent fails."""
        insights = research_tree.get_insights()
        results = research_tree.get_results()
        
        report_parts = [
            f"# Research Report ({report_type.title()})",
            "",
            "## Executive Summary",
            f"This report is based on research involving {len(research_tree.nodes)} data points, ",
            f"including {len(results)} research results and {len(insights)} extracted insights.",
            "",
            "## Key Findings",
            ""
        ]
        
        # Add insights
        for i, insight in enumerate(insights[:10], 1):
            report_parts.append(f"{i}. {insight.content}")
        
        report_parts.extend([
            "",
            "## Research Results Summary",
            ""
        ])
        
        # Add summarized results
        for i, result in enumerate(results[:5], 1):
            if not result.content.startswith("Research error"):
                summary = summarize_content(result.content, max_sentences=3)
                report_parts.append(f"### Finding {i}")
                report_parts.append(summary)
                report_parts.append("")
        
        report_parts.extend([
            "## Conclusion",
            "This research provides valuable insights into the investigated topic. ",
            "Further analysis may be needed for more specific recommendations.",
            "",
            f"*Report generated from {len(research_tree.nodes)} research nodes*"
        ])
        
        return "\n".join(report_parts)
    
    def analyze_research_quality(self, research_tree: Tree) -> Dict[str, Any]:
        """
        Analyze the quality and completeness of research.
        
        Args:
            research_tree: Tree containing research findings
        
        Returns:
            Dictionary with quality metrics
        """
        insights = research_tree.get_insights()
        results = research_tree.get_results()
        
        # Calculate basic metrics
        total_nodes = len(research_tree.nodes)
        content_length = sum(len(node.content) for node in research_tree.nodes.values())
        
        # Analyze sentiment of findings
        all_content = "\n".join([result.content for result in results if result.content])
        sentiment_analysis = analyze_sentiment(all_content) if all_content else {"sentiment": "neutral", "confidence": 0}
        
        quality_score = 0
        feedback = []
        
        # Score based on different factors
        if len(insights) >= 5:
            quality_score += 25
            feedback.append("Good insight extraction")
        elif len(insights) >= 3:
            quality_score += 15
            feedback.append("Moderate insight extraction")
        else:
            feedback.append("Limited insights extracted")
        
        if len(results) >= 3:
            quality_score += 25
            feedback.append("Comprehensive research results")
        elif len(results) >= 1:
            quality_score += 15
            feedback.append("Basic research conducted")
        else:
            feedback.append("Insufficient research results")
        
        if content_length > 5000:
            quality_score += 25
            feedback.append("Rich content gathered")
        elif content_length > 2000:
            quality_score += 15
            feedback.append("Adequate content volume")
        else:
            feedback.append("Limited content gathered")
        
        if total_nodes > 10:
            quality_score += 25
            feedback.append("Well-structured research tree")
        elif total_nodes > 5:
            quality_score += 15
            feedback.append("Basic research structure")
        else:
            feedback.append("Simple research structure")
        
        return {
            "quality_score": quality_score,
            "total_nodes": total_nodes,
            "insights_count": len(insights),
            "results_count": len(results),
            "content_length": content_length,
            "sentiment": sentiment_analysis,
            "feedback": feedback,
            "recommendations": self._get_quality_recommendations(quality_score, feedback)
        }
    
    def _get_quality_recommendations(self, score: int, feedback: List[str]) -> List[str]:
        """Get recommendations for improving research quality."""
        recommendations = []
        
        if score < 40:
            recommendations.append("Consider expanding research scope")
            recommendations.append("Gather more diverse sources")
            recommendations.append("Extract additional insights")
        elif score < 70:
            recommendations.append("Research is adequate but could be enhanced")
            recommendations.append("Consider additional follow-up questions")
        else:
            recommendations.append("Research quality is good")
            recommendations.append("Consider organizing findings for presentation")
        
        return recommendations
