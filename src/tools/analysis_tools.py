"""Analysis tools for content processing and insight extraction."""

from typing import List, Dict, Any
from .decorators import tool


@tool(
    name="extract_insights",
    description="Extract key insights from research content"
)
def extract_insights(content: str, topic: str = "") -> Dict[str, Any]:
    """
    Extract key insights from research content.
    
    Args:
        content: Text content to analyze
        topic: Optional topic context
    
    Returns:
        Dictionary with insights and metadata
    """
    if not content or len(content.strip()) < 50:
        return {"insights": ["Insufficient content for insight extraction"], "count": 0}
    
    # Simple keyword-based insight extraction
    # In a real implementation, you would use LLMs or NLP models
    
    insights = []
    content_lower = content.lower()
    
    # Look for key patterns that indicate insights
    insight_indicators = [
        "according to", "research shows", "study finds", "data indicates",
        "analysis reveals", "evidence suggests", "findings show",
        "results demonstrate", "conclusion", "important", "significant",
        "key finding", "discovery", "breakthrough", "trend", "pattern"
    ]
    
    sentences = content.split('.')
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:
            for indicator in insight_indicators:
                if indicator in sentence.lower():
                    insights.append(sentence.capitalize())
                    break
    
    # Remove duplicates and limit results
    unique_insights = list(set(insights))[:10]
    
    if not unique_insights:
        # Fallback: extract sentences with numbers or statistics
        for sentence in sentences[:20]:  # Check first 20 sentences
            if any(char.isdigit() for char in sentence) and len(sentence) > 30:
                unique_insights.append(sentence.strip().capitalize())
                if len(unique_insights) >= 5:
                    break
    
    final_insights = unique_insights if unique_insights else ["No clear insights extracted from content"]
    
    return {
        "insights": final_insights,
        "count": len(final_insights),
        "topic": topic,
        "content_length": len(content)
    }


@tool(
    name="summarize_content",
    description="Summarize research content into key points"
)
def summarize_content(content: str, max_sentences: int = 5) -> str:
    """
    Summarize content into key points.
    
    Args:
        content: Text content to summarize
        max_sentences: Maximum number of sentences in summary
    
    Returns:
        Summary string
    """
    if not content or len(content.strip()) < 100:
        return "Content too short for meaningful summary."
    
    # Simple extractive summarization
    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
    
    if len(sentences) <= max_sentences:
        return '. '.join(sentences) + '.'
    
    # Score sentences based on simple heuristics
    sentence_scores = []
    
    for i, sentence in enumerate(sentences):
        score = 0
        
        # Prefer sentences with numbers/statistics
        if any(char.isdigit() for char in sentence):
            score += 2
        
        # Prefer sentences with important keywords
        important_words = [
            'important', 'significant', 'key', 'main', 'primary', 'major',
            'research', 'study', 'analysis', 'finding', 'result', 'conclusion',
            'data', 'evidence', 'shows', 'indicates', 'suggests', 'reveals'
        ]
        
        sentence_lower = sentence.lower()
        for word in important_words:
            if word in sentence_lower:
                score += 1
        
        # Prefer sentences not at the very beginning or end
        if 0 < i < len(sentences) - 1:
            score += 1
        
        # Prefer medium-length sentences
        if 50 <= len(sentence) <= 200:
            score += 1
        
        sentence_scores.append((sentence, score))
    
    # Sort by score and take top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    
    # Sort selected sentences by their original order
    selected_indices = []
    for sentence, _ in top_sentences:
        try:
            idx = sentences.index(sentence)
            selected_indices.append((idx, sentence))
        except ValueError:
            continue
    
    selected_indices.sort(key=lambda x: x[0])
    summary_sentences = [sentence for _, sentence in selected_indices]
    
    return '. '.join(summary_sentences) + '.' if summary_sentences else "Unable to generate summary."


@tool(
    name="analyze_sentiment",
    description="Analyze the sentiment of research content"
)
def analyze_sentiment(content: str) -> Dict[str, Any]:
    """
    Analyze sentiment of content.
    
    Args:
        content: Text content to analyze
    
    Returns:
        Dictionary with sentiment analysis results
    """
    if not content:
        return {"sentiment": "neutral", "confidence": 0.0, "reason": "No content provided"}
    
    # Simple sentiment analysis using keyword matching
    positive_words = [
        'good', 'great', 'excellent', 'positive', 'beneficial', 'improvement',
        'success', 'effective', 'promising', 'breakthrough', 'advance', 'progress'
    ]
    
    negative_words = [
        'bad', 'poor', 'negative', 'harmful', 'decline', 'failure', 'ineffective',
        'problem', 'issue', 'concern', 'risk', 'challenge', 'limitation'
    ]
    
    content_lower = content.lower()
    
    positive_count = sum(1 for word in positive_words if word in content_lower)
    negative_count = sum(1 for word in negative_words if word in content_lower)
    
    total_sentiment_words = positive_count + negative_count
    
    if total_sentiment_words == 0:
        return {"sentiment": "neutral", "confidence": 0.5, "reason": "No clear sentiment indicators"}
    
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = positive_count / total_sentiment_words
    elif negative_count > positive_count:
        sentiment = "negative"  
        confidence = negative_count / total_sentiment_words
    else:
        sentiment = "neutral"
        confidence = 0.5
    
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "positive_indicators": positive_count,
        "negative_indicators": negative_count,
        "reason": f"Based on {total_sentiment_words} sentiment indicators found"
    }
