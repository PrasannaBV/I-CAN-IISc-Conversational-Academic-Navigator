"""
Prompt Manager for I-CAN Agent
Handles dynamic prompt selection and context injection based on query analysis
"""

from typing import Dict, List, Optional, Tuple
import re
from prompting.prompt_templates import SYSTEM_PROMPT, ACADEMIC_PROMPTS

from prompting.context_analyzer import ContextAnalyzer

context_analyzer = ContextAnalyzer()



class PromptManager:
    """Manages prompt selection and context injection for the I-CAN agent"""
    
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT
        self.academic_prompts = ACADEMIC_PROMPTS
        self.context_keywords = self._build_context_keywords()
    
    def _build_context_keywords(self) -> Dict[str, List[str]]:
        """Build keyword mappings for each prompt category"""
        return {
            "clarification_needed": [
                "what", "how", "when", "where", "why", "which", "unclear", 
                "confused", "don't understand", "explain", "clarify"
            ],
            "admission_inquiry": [
                "admission", "apply", "application", "eligibility", "entrance", 
                "requirement", "deadline", "fee", "process", "apply for"
            ],
            "course_information": [
                "course", "subject", "curriculum", "syllabus", "credits", 
                "semester", "elective", "core", "prerequisites"
            ],
            "research_guidance": [
                "research", "thesis", "project", "supervisor", "guide", 
                "phd", "mtech", "publication", "lab", "faculty"
            ],
            "academic_procedures": [
                "procedure", "process", "steps", "how to", "registration", 
                "enrollment", "grade", "exam", "assessment"
            ],
            "campus_life": [
                "hostel", "accommodation", "mess", "campus", "facilities", 
                "library", "sports", "clubs", "events"
            ],
            "technical_support": [
                "technical", "software", "system", "login", "access", 
                "portal", "website", "error", "problem"
            ]
        }
    
    def select_prompt_template(self, query: str, context_analysis: Optional[Dict] = None) -> str:
        """
        Select the most appropriate prompt template based on query analysis
        
        Args:
            query: User's query
            context_analysis: Pre-computed context analysis (optional)
            
        Returns:
            Selected prompt template string
        """
        if context_analysis is None:
            context_analysis = context_analyzer.analyze_query_context(query)

        
        # Get the primary intent
        primary_intent = context_analysis.get('primary_intent', 'general')
        confidence = context_analysis.get('confidence', 0.5)
        
        # If confidence is low, use clarification prompt
        if confidence < 0.6:
            return self.academic_prompts.get('clarification_needed', self.system_prompt)
        
        # Select based on primary intent
        selected_prompt = self.academic_prompts.get(primary_intent, self.system_prompt)
        
        return selected_prompt
    
    def enhance_query_with_context(
        self, 
        query: str, 
        retrieved_docs: List[Dict], 
        context_analysis: Optional[Dict] = None
    ) -> str:
        """
        Enhance the user query with relevant context and prompt template
        
        Args:
            query: Original user query
            retrieved_docs: Documents retrieved from vector store
            context_analysis: Query context analysis
            
        Returns:
            Enhanced query with context and appropriate prompting
        """
        if context_analysis is None:
            context_analysis = context_analyzer.analyze_query_context(query)

        
        # Select appropriate prompt template
        prompt_template = self.select_prompt_template(query, context_analysis)
        
        # Build context from retrieved documents
        context_text = self._build_context_from_docs(retrieved_docs)
        
        # Build the enhanced query
        enhanced_query = f"""{prompt_template}

CONTEXT INFORMATION:
{context_text}

QUERY ANALYSIS:
- Primary Intent: {context_analysis.get('primary_intent', 'general')}
- Confidence: {context_analysis.get('confidence', 0.5):.2f}
- Key Topics: {', '.join(context_analysis.get('topics', []))}
- Urgency Level: {context_analysis.get('urgency', 'normal')}

USER QUESTION:
{query}

Please provide a helpful, accurate response based on the context and your role as I-CAN agent."""
        
        return enhanced_query
    
    def _build_context_from_docs(self, retrieved_docs: List[Dict], max_length: int = 2000) -> str:
        """
        Build context text from retrieved documents
        
        Args:
            retrieved_docs: List of retrieved documents with scores
            max_length: Maximum length of context text
            
        Returns:
            Formatted context text
        """
        if not retrieved_docs:
            return "No specific context found in the knowledge base."
        
        context_parts = []
        current_length = 0
        
        for i, (doc, score) in enumerate(retrieved_docs):
            if current_length >= max_length:
                break
                
            doc_text = doc.page_content.strip()
            if doc_text:
                # Add document with relevance score
                doc_section = f"[Relevance: {score:.3f}]\n{doc_text}\n"
                
                if current_length + len(doc_section) > max_length:
                    # Truncate if needed
                    remaining = max_length - current_length
                    doc_section = doc_section[:remaining] + "..."
                
                context_parts.append(doc_section)
                current_length += len(doc_section)
        
        return "\n---\n".join(context_parts)
    
    def get_followup_prompts(self, query: str, response: str) -> List[str]:
        """
        Generate follow-up question suggestions based on the query and response
        
        Args:
            query: Original user query
            response: Agent's response
            
        Returns:
            List of suggested follow-up questions
        """
        context_analysis = context_analyzer.analyze_query_context(query)
        primary_intent = context_analysis.get('primary_intent', 'general')
        
        followup_maps = {
            "admission_inquiry": [
                "What are the specific eligibility criteria?",
                "What documents do I need to prepare?",
                "When is the application deadline?",
                "What is the fee structure?"
            ],
            "course_information": [
                "What are the prerequisites for this course?",
                "How many credits is this course worth?",
                "Who are the faculty teaching this course?",
                "What is the assessment pattern?"
            ],
            "research_guidance": [
                "How do I find a research supervisor?",
                "What is the typical timeline for completion?",
                "What funding opportunities are available?",
                "What are the publication requirements?"
            ],
            "campus_life": [
                "What facilities are available on campus?",
                "How is the food and accommodation?",
                "What extracurricular activities can I join?",
                "How is the campus connectivity?"
            ]
        }
        
        return followup_maps.get(primary_intent, [
            "Can you provide more specific information?",
            "What are the next steps I should take?",
            "Are there any important deadlines I should know?",
            "Where can I get official confirmation of this information?"
        ])
    
    def format_response_with_sources(self, response: str, sources: List[Dict]) -> str:
        """
        Format the response with source attribution
        
        Args:
            response: Agent's response
            sources: List of source documents
            
        Returns:
            Formatted response with sources
        """
        if not sources:
            return response
        
        formatted_response = f"{response}\n\n"
        formatted_response += "**Sources:**\n"
        
        for i, (doc, score) in enumerate(sources[:3], 1):
            # Extract source info if available in metadata
            source_info = doc.metadata.get('source', 'Internal Knowledge Base')
            formatted_response += f"{i}. {source_info} (Relevance: {score:.3f})\n"
        
        return formatted_response
    
    def validate_response_quality(self, query: str, response: str) -> Dict[str, any]:
        """
        Validate the quality of the agent's response
        
        Args:
            query: Original query
            response: Agent's response
            
        Returns:
            Quality assessment dictionary
        """
        assessment = {
            "length_appropriate": 50 <= len(response) <= 2000,
            "contains_specific_info": not self._is_generic_response(response),
            "addresses_query": self._response_addresses_query(query, response),
            "professional_tone": self._has_professional_tone(response),
            "actionable": self._contains_actionable_info(response)
        }
        
        assessment["overall_score"] = sum(assessment.values()) / len(assessment)
        return assessment
    
    def _is_generic_response(self, response: str) -> bool:
        """Check if response is too generic"""
        generic_phrases = [
            "i don't know", "not sure", "maybe", "perhaps", 
            "generally speaking", "it depends", "sorry, i can't"
        ]
        return any(phrase in response.lower() for phrase in generic_phrases)
    
    def _response_addresses_query(self, query: str, response: str) -> bool:
        """Check if response addresses the query"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        return overlap >= min(3, len(query_words) * 0.3)
    
    def _has_professional_tone(self, response: str) -> bool:
        """Check if response maintains professional tone"""
        unprofessional_indicators = ["lol", "omg", "btw", "ur", "gonna"]
        return not any(indicator in response.lower() for indicator in unprofessional_indicators)
    
    def _contains_actionable_info(self, response: str) -> bool:
        """Check if response contains actionable information"""
        actionable_indicators = [
            "visit", "contact", "apply", "submit", "register", 
            "check", "follow", "complete", "prepare", "deadline"
        ]
        return any(indicator in response.lower() for indicator in actionable_indicators)


# Singleton instance for global use
prompt_manager = PromptManager()


def get_enhanced_query(query: str, retrieved_docs: List[Dict]) -> str:
    """
    Convenience function to get enhanced query with context
    
    Args:
        query: User's original query
        retrieved_docs: Retrieved documents from vector store
        
    Returns:
        Enhanced query with appropriate prompting and context
    """
    return prompt_manager.enhance_query_with_context(query, retrieved_docs)


def get_followup_suggestions(query: str, response: str) -> List[str]:
    """
    Convenience function to get follow-up suggestions
    
    Args:
        query: Original user query
        response: Agent's response
        
    Returns:
        List of follow-up question suggestions
    """
    return prompt_manager.get_followup_prompts(query, response)