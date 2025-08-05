"""
Gemma LLM Client for ICICI Bank HR Assistant
===========================================

This module provides a client interface for the Google Gemma language model,
specifically designed for the HR Assistant system. It handles AI-powered responses
for various HR queries including policy information, intent classification, and
complex query parsing.

Key Features:
- Intelligent response generation for HR queries
- Intent classification for routing to appropriate agents
- JSON response parsing for structured data
- Caching system for improved performance
- Rate limiting to prevent API abuse
- Fallback mechanisms for error handling

The client integrates with the main Gemma model implementation and provides
a simplified interface for the HR Assistant agents.

Author: ICICI Bank Development Team
Version: 1.0.0
"""

import json
import time
from typing import Dict, Any, Optional, List, Union
from loguru import logger
import sys
import os

# Add the gemma_chatbot directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gemma_chatbot'))
# Import the main Gemma model implementation
from gemma import call_gemma

class GemmaClientError(Exception):
    """Custom exception for Gemma client errors"""
    pass

class GemmaClient:
    """
    Gemma LLM client wrapper for HR Assistant system
    
    This class provides a high-level interface to the Gemma language model
    with features like caching, rate limiting, and error handling.
    """
    
    def __init__(self):
        """Initialize the Gemma client with caching and rate limiting"""
        self.cache = {}  # Response cache for improved performance
        self.cache_ttl = 300  # Cache time-to-live: 5 minutes
        self.last_request_time = 0  # For rate limiting
        self.min_request_interval = 1.0  # Minimum time between requests (seconds)
        
    def _get_cache_key(self, prompt: str, response_type: str, max_tokens: int, temperature: float) -> str:
        """
        Generate a unique cache key for a request
        
        Args:
            prompt (str): The input prompt
            response_type (str): Type of response expected
            max_tokens (int): Maximum tokens for response
            temperature (float): Temperature setting
            
        Returns:
            str: Unique cache key
        """
        return f"{hash(prompt)}_{response_type}_{max_tokens}_{temperature}"
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """
        Check if a cache entry is still valid
        
        Args:
            timestamp (float): Cache entry timestamp
            
        Returns:
            bool: True if cache entry is still valid
        """
        return time.time() - timestamp < self.cache_ttl
        
    def _rate_limit(self):
        """
        Implement rate limiting to prevent API abuse
        
        Ensures minimum interval between requests to avoid overwhelming
        the LLM service.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def generate(self, prompt: str, response_type: str = None, query_type: str = None, max_tokens: int = 1000, temperature: float = 0.3, **kwargs) -> str:
        """
        Generate response using Gemma model with caching and rate limiting
        
        This is the main method for generating AI responses. It handles:
        - Backward compatibility with different parameter names
        - Response caching for improved performance
        - Rate limiting to prevent API abuse
        - Error handling and fallback responses
        
        Args:
            prompt (str): Input prompt for the LLM
            response_type (str, optional): Type of response expected ('text', 'json', 'intent')
            query_type (str, optional): Legacy parameter name for response_type
            max_tokens (int): Maximum tokens for response
            temperature (float): Temperature setting (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional parameters for backward compatibility
            
        Returns:
            str: Generated response from the LLM
        """
        try:
            # Handle backward compatibility - prioritize response_type, fall back to query_type
            if response_type is None and query_type is not None:
                response_type = query_type
            elif response_type is None:
                response_type = "text"
            
            # Handle legacy parameter names (for backward compatibility)
            if 'use_cache' in kwargs:
                pass  # We always use cache
            if 'system_prompt' in kwargs:
                pass  # We don't use system prompts in this simple version
            
            # Check cache first for improved performance
            cache_key = self._get_cache_key(prompt, response_type, max_tokens, temperature)
            
            if cache_key in self.cache:
                cached_response, timestamp = self.cache[cache_key]
                if self._is_cache_valid(timestamp):
                    logger.info(f"✅ Cache hit for {response_type} query")
                    return cached_response
                else:
                    # Remove expired cache entry
                    del self.cache[cache_key]
        
            # Apply rate limiting
            self._rate_limit()
            
            # Call the Gemma model
            response = call_gemma(prompt, response_type, max_tokens, temperature)
            
            # Cache the response for future use
            self.cache[cache_key] = (response, time.time())
            
            # Clean old cache entries periodically to prevent memory bloat
            if len(self.cache) > 100:  # Arbitrary limit
                current_time = time.time()
                expired_keys = [
                    key for key, (_, timestamp) in self.cache.items()
                    if current_time - timestamp > self.cache_ttl
                ]
                for key in expired_keys:
                    del self.cache[key]
            
            logger.info(f"✅ Generated {len(response)} characters for {response_type} query")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return self._get_fallback_response(response_type)
    
    def _get_fallback_response(self, response_type: str) -> str:
        """
        Provides fallback responses for different error scenarios.
        
        Args:
            response_type (str): The type of response that failed.
            
        Returns:
            str: A fallback response string.
        """
        if response_type == "json":
            return '{"error": "API unavailable", "fallback": true}'
        elif response_type == "intent":
            return "SINGLE:chatbot"
        else:
            return "I apologize, but I'm having trouble processing your request right now. Please try again later."
    
    def generate_intent(self, query: str) -> str:
        """Optimized intent classification"""
        prompt = f"""Analyze this HR query and classify the intent. Respond with ONE of these formats:
- SINGLE:agent_name (for single agent tasks)
- MULTI:agent1,agent2 (for multi-agent tasks)  
- COMPLEX:sequential (for complex multi-step workflows)

Query: "{query}"

Examples:
- "check my leave balance" → SINGLE:leave
- "apply for leave and also request laptop" → COMPLEX:sequential
- "check in" → SINGLE:attendance

Response format only:"""
        
        return self.generate(prompt, "intent", 50, 0.1)
    
    def generate_json(self, prompt: str) -> str:
        """Optimized JSON generation"""
        return self.generate(prompt, "json", 200, 0.1)
    
    def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """Optimized text generation"""
        return self.generate(prompt, "text", max_tokens, 0.3)
    
    def parse_complex_query(self, query: str) -> str:
        """Parse complex multi-step queries"""
        prompt = f"""Parse this complex HR query into sequential tasks. Return JSON array with tasks.

Query: "{query}"

Format each task as:
{{"task_type": "leave_request|balance_check|equipment_request|complaint", "sub_query": "specific query", "agent": "leave|complaint|attendance|chatbot"}}

Examples:
Input: "check leave balance and apply for 3 days leave"
Output: [{{"task_type": "balance_check", "sub_query": "check my leave balance", "agent": "leave"}}, {{"task_type": "leave_request", "sub_query": "apply for 3 days leave", "agent": "leave"}}]

Respond with valid JSON array only:"""
        
        return self.generate_json(prompt)
    
    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_ttl": self.cache_ttl,
            "last_request_time": self.last_request_time
        }
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Legacy method for backward compatibility"""
        return self.generate(prompt, **kwargs)
    
    def parse_json_response(self, prompt: str, schema: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Legacy method for parsing JSON responses"""
        try:
            response = self.generate(prompt, response_type="json", **kwargs)
            # Try to parse as JSON
            if response.startswith('{') or response.startswith('['):
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    pass
            # Return fallback structure
            return {"error": "Failed to parse JSON", "raw_response": response}
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            return {"error": str(e), "fallback": True}
    
    def analyze_sentiment(self, text: str) -> str:
        """Legacy method for sentiment analysis"""
        prompt = f"What is the sentiment of this text: '{text[:100]}'"
        response = self.generate(prompt, response_type="text", max_tokens=20, temperature=0.1)
        return response.lower().strip()
    
    def extract_entities(self, text: str, entity_types: List[str]) -> Dict[str, List[str]]:
        """Legacy method for entity extraction"""
        prompt = f"Extract {', '.join(entity_types)} from: {text[:200]}"
        try:
            response = self.parse_json_response(prompt)
            return response.get("entities", {entity_type: [] for entity_type in entity_types})
        except:
            return {entity_type: [] for entity_type in entity_types}
    
    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Legacy method for text summarization"""
        prompt = f"Summarize this text in {max_length} characters: {text[:300]}"
        return self.generate(prompt, response_type="text", max_tokens=max_length // 3, temperature=0.2)

# Global instance
gemma_client = GemmaClient()

# Backward compatibility functions
def generate_intent(query: str) -> str:
    """Generate intent classification"""
    return gemma_client.generate_intent(query)

def generate_json(prompt: str) -> str:
    """Generate JSON response"""
    return gemma_client.generate_json(prompt)

def generate_text(prompt: str, max_tokens: int = 500) -> str:
    """Generate text response"""
    return gemma_client.generate_text(prompt, max_tokens)

def parse_complex_query(query: str) -> str:
    """Parse complex queries"""
    return gemma_client.parse_complex_query(query)

def call_gemma_with_fallback(prompt: str, response_type: str = "text", max_tokens: int = 1000, temperature: float = 0.3) -> str:
    """Legacy function wrapper"""
    return gemma_client.generate(prompt, response_type, max_tokens, temperature)

# Legacy functions that agents are trying to import
def llm_generate(prompt: str, **kwargs) -> str:
    """Generate text using Gemma model - legacy function"""
    max_tokens = kwargs.get('max_tokens', 500)
    temperature = kwargs.get('temperature', 0.3)
    return gemma_client.generate(prompt, "text", max_tokens, temperature)

def llm_json(prompt: str, schema: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
    """Generate and parse JSON response - legacy function"""
    try:
        max_tokens = kwargs.get('max_tokens', 200)
        response = gemma_client.generate(prompt, "json", max_tokens, 0.1)
        
        # Try to parse as JSON
        if response.startswith('{') or response.startswith('['):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
        
        # Return fallback structure
        return {"error": "Failed to parse JSON", "raw_response": response}
        
    except Exception as e:
        logger.error(f"llm_json failed: {e}")
        return {"error": str(e), "fallback": True}

def llm_decision(context: str, options: List[str], **kwargs) -> str:
    """Make a decision - legacy function"""
    prompt = f"Context: {context}\nOptions: {', '.join(options)}\nChoose the best option:"
    return gemma_client.generate(prompt, "text", 50, 0.1)

def llm_parse_json_response(prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
    """Parse JSON response - legacy function"""
    try:
        response = gemma_client.generate(prompt, "json", max_tokens, 0.1)
        
        # Try to extract JSON from response
        if '{' in response and '}' in response:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        
        return {"error": "No JSON found in response", "raw": response}
        
    except Exception as e:
        logger.error(f"JSON parsing failed: {e}")
        return {"error": str(e), "raw": response if 'response' in locals() else ""}

# Additional legacy functions for compatibility
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment - legacy function"""
    prompt = f"What is the sentiment of this text: '{text[:100]}'"
    response = gemma_client.generate(prompt, "text", 20, 0.1)
    return response.lower().strip()

def extract_entities(text: str, entity_types: List[str]) -> Dict[str, List[str]]:
    """Extract entities - legacy function"""
    prompt = f"Extract {', '.join(entity_types)} from: {text[:200]}"
    try:
        response = llm_json(prompt)
        return response.get("entities", {entity_type: [] for entity_type in entity_types})
    except:
        return {entity_type: [] for entity_type in entity_types}

def summarize_text(text: str, max_length: int = 100) -> str:
    """Summarize text - legacy function"""
    prompt = f"Summarize this text in {max_length} characters: {text[:300]}"
    return gemma_client.generate(prompt, "text", max_length // 3, 0.2) 