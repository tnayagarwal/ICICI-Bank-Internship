"""
Groq API Client for Llama 3.3 70B Versatile
Optimized for minimal API calls and maximum performance
"""
import json
import time
from typing import Dict, Any, Optional, List, Union
from groq import Groq
from loguru import logger
from config import settings

class GroqClientError(Exception):
    """Custom exception for Groq client errors"""
    pass

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model_name = settings.MODEL_NAME
        self.request_cache = {}
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Rate limiting
        
    def _rate_limit(self):
        """Simple rate limiting to avoid API throttling"""
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            time.sleep(self.min_request_interval - (current_time - self.last_request_time))
        self.last_request_time = time.time()
    
    def _cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for request memoization"""
        cache_data = {
            "prompt": prompt,
            "model": self.model_name,
            **kwargs
        }
        return str(hash(json.dumps(cache_data, sort_keys=True)))
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
        system_prompt: str = "You are a helpful AI assistant.",
        **kwargs
    ) -> str:
        """
        Generate text using Groq Llama 3.3 70B
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use response caching
            system_prompt: System message for context
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
            
        Raises:
            GroqClientError: If API call fails
        """
        # Use defaults from settings
        max_tokens = max_tokens or settings.MAX_TOKENS
        temperature = temperature or settings.TEMPERATURE
        
        # Check cache first
        if use_cache:
            cache_key = self._cache_key(prompt, max_tokens=max_tokens, temperature=temperature)
            if cache_key in self.request_cache:
                logger.info("Using cached response for prompt")
                return self.request_cache[cache_key]
        
        try:
            self._rate_limit()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stream=False,
                **kwargs
            )
            
            result = response.choices[0].message.content.strip()
            
            # Cache successful response
            if use_cache:
                self.request_cache[cache_key] = result
                # Limit cache size
                if len(self.request_cache) > 100:
                    # Remove oldest entries
                    oldest_keys = list(self.request_cache.keys())[:20]
                    for key in oldest_keys:
                        del self.request_cache[key]
            
            logger.info(f"Generated {len(result)} characters with {max_tokens} max tokens")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            if "no healthy upstream" in error_msg:
                logger.error("Groq API temporarily unavailable")
                return "[Groq API temporarily unavailable. Please try again in a moment.]"
            elif "rate limit" in error_msg:
                logger.error("Groq API rate limit exceeded")
                time.sleep(2)  # Wait before retry
                return "[Rate limit exceeded. Please wait a moment before trying again.]"
            else:
                logger.error(f"Groq API error: {str(e)}")
                return "[Groq API Error: Unable to generate response]"
    
    def parse_json_response(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        max_retries: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate and parse JSON response with schema validation
        
        Args:
            prompt: Input prompt expecting JSON output
            schema: Expected JSON schema for validation
            max_retries: Number of retry attempts
            **kwargs: Additional parameters for generate()
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            GroqClientError: If parsing fails after retries
        """
        json_prompt = f"""
{prompt}

Respond with valid JSON only. No explanation or markdown.
Expected schema: {json.dumps(schema, indent=2)}
"""
        
        for attempt in range(max_retries + 1):
            try:
                response = self.generate(
                    json_prompt,
                    temperature=0.1,  # Lower temperature for structured output
                    **kwargs
                )
                
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON structure found in response")
                
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Basic schema validation
                for key in schema.keys():
                    if key not in parsed:
                        raise ValueError(f"Missing required field: {key}")
                
                logger.info(f"Successfully parsed JSON response on attempt {attempt + 1}")
                return parsed
                
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries:
                    logger.error(f"Failed to parse JSON after {max_retries + 1} attempts: {str(e)}")
                    raise GroqClientError(f"JSON parsing failed: {str(e)}")
                
                logger.warning(f"JSON parsing attempt {attempt + 1} failed: {str(e)}")
                time.sleep(0.5)  # Brief delay before retry
    
    def analyze_sentiment(self, text: str) -> str:
        """Quick sentiment analysis for complaints/requests"""
        prompt = f"""
Analyze the sentiment of this text in one word: positive, negative, or neutral.

Text: "{text}"

Sentiment:"""
        
        return self.generate(
            prompt, 
            max_tokens=10, 
            temperature=0.1,
            system_prompt="You are a sentiment analysis expert."
        ).lower().strip()
    
    def extract_entities(self, text: str, entity_types: List[str]) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        prompt = f"""
Extract {', '.join(entity_types)} from this text.
Return as JSON with entity types as keys and lists of found entities as values.

Text: "{text}"
"""
        
        schema = {entity_type: [] for entity_type in entity_types}
        return self.parse_json_response(prompt, schema, max_tokens=150)
    
    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Generate concise summary of text"""
        prompt = f"""
Summarize this text in maximum {max_length} characters:

{text}

Summary:"""
        
        return self.generate(
            prompt,
            max_tokens=max_length // 3,  # Rough token estimate
            temperature=0.3,
            system_prompt="You are a concise summarization expert."
        )
    
    def clear_cache(self):
        """Clear the response cache"""
        self.request_cache.clear()
        logger.info("Response cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_responses": len(self.request_cache),
            "memory_usage_kb": len(str(self.request_cache)) // 1024
        }

# Global client instance
groq_client = GroqClient()

# Convenience functions for common operations
def llm_generate(prompt: str, **kwargs) -> str:
    """Quick LLM generation"""
    return groq_client.generate(prompt, **kwargs)

def llm_json(prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Quick JSON parsing"""
    return groq_client.parse_json_response(prompt, schema, **kwargs)

def llm_decision(context: str, options: List[str], **kwargs) -> str:
    """Quick decision making"""
    prompt = f"""
Given this context: {context}

Choose the best option from: {', '.join(options)}

Respond with only the chosen option, no explanation.
"""
    
    response = groq_client.generate(prompt, max_tokens=20, temperature=0.1, **kwargs)
    
    # Find closest matching option
    for option in options:
        if option.lower() in response.lower():
            return option
    
    # Fallback to first option if no match
    return options[0]

def llm_parse_json_response(prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
    """Parse JSON response from LLM with error handling"""
    try:
        response = groq_client.generate(
            prompt + "\n\nReturn valid JSON only.",
            max_tokens=max_tokens,
            temperature=0.1
        )
        
        # Try to extract JSON from response
        import json
        import re
        
        # Find JSON pattern in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_data = json.loads(json_str)
            return {
                "success": True,
                "data": parsed_data
            }
        else:
            return {
                "success": False,
                "error": "No valid JSON found in response"
            }
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {str(e)}")
        return {
            "success": False,
            "error": f"Invalid JSON format: {str(e)}"
        }
    except Exception as e:
        logger.error(f"LLM parsing failed: {str(e)}")
        return {
            "success": False,
            "error": f"Parsing error: {str(e)}"
        } 