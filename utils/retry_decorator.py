"""
Reflection and Retry Decorator for LLM Operations
Provides self-healing capabilities with intelligent retry patterns
"""
import asyncio
import time
import json
from functools import wraps
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass
from loguru import logger

@dataclass
class RetryAttempt:
    attempt_number: int
    timestamp: float
    error: Optional[str]
    confidence: Optional[float]
    result: Optional[Any]
    reflection: Optional[str]

class RetryWithReflection:
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 backoff_factor: float = 2.0, min_confidence: float = 0.7):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.min_confidence = min_confidence
        self.attempts_log: List[RetryAttempt] = []
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(func):
                return self._execute_async(func, args, kwargs)
            else:
                return self._execute_sync(func, args, kwargs)
        return wrapper
    
    def _execute_sync(self, func: Callable, args: tuple, kwargs: dict):
        """Execute synchronous function with retry logic"""
        self.attempts_log.clear()
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                start_time = time.time()
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Analyze result confidence
                confidence = self._analyze_confidence(result, func.__name__)
                
                # Log successful attempt
                self.attempts_log.append(RetryAttempt(
                    attempt_number=attempt,
                    timestamp=start_time,
                    error=None,
                    confidence=confidence,
                    result=result,
                    reflection=None
                ))
                
                # Check if confidence is acceptable
                if confidence >= self.min_confidence:
                    logger.info(f"Function {func.__name__} succeeded on attempt {attempt} with confidence {confidence:.2f}")
                    return result
                else:
                    # Low confidence - trigger reflection
                    reflection = self._generate_reflection(func.__name__, args, kwargs, result, self.attempts_log)
                    self.attempts_log[-1].reflection = reflection
                    
                    if attempt < self.max_attempts:
                        logger.warning(f"Low confidence ({confidence:.2f}) on attempt {attempt}, retrying with reflection")
                        # Modify kwargs based on reflection
                        kwargs = self._apply_reflection(kwargs, reflection)
                        time.sleep(self.base_delay * (self.backoff_factor ** (attempt - 1)))
                    else:
                        logger.warning(f"Final attempt had low confidence ({confidence:.2f}), returning result anyway")
                        return result
                
            except Exception as e:
                error_msg = str(e)
                
                # Log failed attempt
                self.attempts_log.append(RetryAttempt(
                    attempt_number=attempt,
                    timestamp=time.time(),
                    error=error_msg,
                    confidence=0.0,
                    result=None,
                    reflection=None
                ))
                
                if attempt < self.max_attempts:
                    # Generate reflection for error
                    reflection = self._generate_error_reflection(func.__name__, args, kwargs, error_msg, self.attempts_log)
                    self.attempts_log[-1].reflection = reflection
                    
                    logger.warning(f"Attempt {attempt} failed: {error_msg}. Retrying with reflection.")
                    
                    # Modify parameters based on reflection
                    kwargs = self._apply_error_reflection(kwargs, reflection, error_msg)
                    time.sleep(self.base_delay * (self.backoff_factor ** (attempt - 1)))
                else:
                    logger.error(f"All {self.max_attempts} attempts failed for {func.__name__}")
                    raise e
        
        # Should not reach here, but just in case
        raise Exception(f"Unexpected end of retry loop for {func.__name__}")
    
    async def _execute_async(self, func: Callable, args: tuple, kwargs: dict):
        """Execute async function with retry logic"""
        self.attempts_log.clear()
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                start_time = time.time()
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Analyze result confidence
                confidence = self._analyze_confidence(result, func.__name__)
                
                # Log successful attempt
                self.attempts_log.append(RetryAttempt(
                    attempt_number=attempt,
                    timestamp=start_time,
                    error=None,
                    confidence=confidence,
                    result=result,
                    reflection=None
                ))
                
                # Check if confidence is acceptable
                if confidence >= self.min_confidence:
                    logger.info(f"Async function {func.__name__} succeeded on attempt {attempt} with confidence {confidence:.2f}")
                    return result
                else:
                    # Low confidence - trigger reflection
                    reflection = self._generate_reflection(func.__name__, args, kwargs, result, self.attempts_log)
                    self.attempts_log[-1].reflection = reflection
                    
                    if attempt < self.max_attempts:
                        logger.warning(f"Low confidence ({confidence:.2f}) on attempt {attempt}, retrying with reflection")
                        # Modify kwargs based on reflection
                        kwargs = self._apply_reflection(kwargs, reflection)
                        await asyncio.sleep(self.base_delay * (self.backoff_factor ** (attempt - 1)))
                    else:
                        logger.warning(f"Final attempt had low confidence ({confidence:.2f}), returning result anyway")
                        return result
                
            except Exception as e:
                error_msg = str(e)
                
                # Log failed attempt
                self.attempts_log.append(RetryAttempt(
                    attempt_number=attempt,
                    timestamp=time.time(),
                    error=error_msg,
                    confidence=0.0,
                    result=None,
                    reflection=None
                ))
                
                if attempt < self.max_attempts:
                    # Generate reflection for error
                    reflection = self._generate_error_reflection(func.__name__, args, kwargs, error_msg, self.attempts_log)
                    self.attempts_log[-1].reflection = reflection
                    
                    logger.warning(f"Attempt {attempt} failed: {error_msg}. Retrying with reflection.")
                    
                    # Modify parameters based on reflection
                    kwargs = self._apply_error_reflection(kwargs, reflection, error_msg)
                    await asyncio.sleep(self.base_delay * (self.backoff_factor ** (attempt - 1)))
                else:
                    logger.error(f"All {self.max_attempts} attempts failed for {func.__name__}")
                    raise e
        
        # Should not reach here, but just in case
        raise Exception(f"Unexpected end of retry loop for {func.__name__}")
    
    def _analyze_confidence(self, result: Any, function_name: str) -> float:
        """Analyze confidence level of the result"""
        try:
            # Different confidence analysis based on function type
            if function_name in ['generate', 'llm_generate']:
                return self._analyze_text_confidence(result)
            elif function_name in ['parse_json_response', 'llm_json']:
                return self._analyze_json_confidence(result)
            elif isinstance(result, dict) and 'confidence' in result:
                return float(result.get('confidence', 0.5))
            elif isinstance(result, dict) and 'success' in result:
                return 0.9 if result.get('success') else 0.1
            else:
                # Default confidence for unknown result types
                return 0.8 if result else 0.2
                
        except Exception as e:
            logger.warning(f"Confidence analysis failed: {str(e)}")
            return 0.5  # Neutral confidence
    
    def _analyze_text_confidence(self, text: str) -> float:
        """Analyze confidence of text generation"""
        if not text or len(text.strip()) < 10:
            return 0.1
        
        # Check for common error indicators
        error_indicators = [
            "i don't know", "i'm not sure", "i cannot", "unable to",
            "error", "failed", "sorry", "unavailable", "unclear"
        ]
        
        text_lower = text.lower()
        error_count = sum(1 for indicator in error_indicators if indicator in text_lower)
        
        if error_count > 0:
            return max(0.1, 0.8 - (error_count * 0.2))
        
        # Check for completeness
        if len(text) < 50:
            return 0.6
        elif len(text) > 500:
            return 0.9
        else:
            return 0.8
    
    def _analyze_json_confidence(self, json_result: Any) -> float:
        """Analyze confidence of JSON parsing result"""
        if not isinstance(json_result, dict):
            return 0.1
        
        # Check if JSON has required structure
        if len(json_result) == 0:
            return 0.1
        
        # Check for error fields
        if 'error' in json_result:
            return 0.2
        
        # Check completeness
        null_fields = sum(1 for v in json_result.values() if v is None or v == "")
        total_fields = len(json_result)
        
        if total_fields == 0:
            return 0.1
        
        completeness = 1.0 - (null_fields / total_fields)
        return max(0.3, completeness)
    
    def _generate_reflection(self, function_name: str, args: tuple, kwargs: dict, 
                           result: Any, attempts: List[RetryAttempt]) -> str:
        """Generate reflection on low-confidence result"""
        try:
            # Simple rule-based reflection for now
            reflection = f"Low confidence result for {function_name}. "
            
            if function_name in ['generate', 'llm_generate']:
                reflection += "Consider: 1) More specific prompt, 2) Lower temperature, 3) Better context"
            elif function_name in ['parse_json_response', 'llm_json']:
                reflection += "Consider: 1) Simpler JSON schema, 2) Clearer instructions, 3) Temperature=0.1"
            else:
                reflection += "Consider: 1) Adjust parameters, 2) Add more context, 3) Simplify request"
                
            return reflection
            
        except Exception as e:
            logger.warning(f"Reflection generation failed: {str(e)}")
            return "Unable to generate reflection - retry with modified parameters"
    
    def _generate_error_reflection(self, function_name: str, args: tuple, kwargs: dict,
                                 error_msg: str, attempts: List[RetryAttempt]) -> str:
        """Generate reflection on error"""
        try:
            error_lower = error_msg.lower()
            reflection = f"Error in {function_name}: {error_msg[:100]}. "
            
            if 'json' in error_lower and 'parse' in error_lower:
                reflection += "Fix: Use simpler JSON structure, add 'Return valid JSON only' to prompt"
            elif 'timeout' in error_lower or 'limit' in error_lower:
                reflection += "Fix: Reduce max_tokens, simplify request"
            elif 'rate limit' in error_lower:
                reflection += "Fix: Wait longer between requests"
            else:
                reflection += "Fix: Adjust parameters, check inputs"
                
            return reflection
            
        except Exception as e:
            logger.warning(f"Error reflection generation failed: {str(e)}")
            return f"Error reflection failed, try with different parameters for: {error_msg}"
    
    def _apply_reflection(self, kwargs: dict, reflection: str) -> dict:
        """Apply reflection insights to modify parameters"""
        modified_kwargs = kwargs.copy()
        
        try:
            # Simple rule-based modifications based on reflection content
            reflection_lower = reflection.lower()
            
            if 'reduce' in reflection_lower and 'tokens' in reflection_lower:
                if 'max_tokens' in modified_kwargs:
                    modified_kwargs['max_tokens'] = max(50, int(modified_kwargs['max_tokens'] * 0.8))
            
            if 'lower temperature' in reflection_lower or 'temperature=0.1' in reflection_lower:
                if 'temperature' in modified_kwargs:
                    modified_kwargs['temperature'] = min(0.1, modified_kwargs.get('temperature', 0.3))
            
            if 'more specific' in reflection_lower or 'clearer' in reflection_lower:
                if 'prompt' in modified_kwargs:
                    modified_kwargs['prompt'] += "\n\nBe specific and clear in your response."
            
            if 'context' in reflection_lower:
                if 'prompt' in modified_kwargs:
                    modified_kwargs['prompt'] = "Provide detailed context. " + modified_kwargs['prompt']
            
            # Log modification
            if modified_kwargs != kwargs:
                logger.info("Applied reflection modifications to parameters")
            
        except Exception as e:
            logger.warning(f"Failed to apply reflection: {str(e)}")
        
        return modified_kwargs
    
    def _apply_error_reflection(self, kwargs: dict, reflection: str, error_msg: str) -> dict:
        """Apply error reflection to fix parameters"""
        modified_kwargs = kwargs.copy()
        
        try:
            reflection_lower = reflection.lower()
            error_lower = error_msg.lower()
            
            # Handle specific error types
            if 'json' in error_lower and 'parse' in error_lower:
                if 'prompt' in modified_kwargs:
                    modified_kwargs['prompt'] += "\n\nReturn only valid JSON, no explanations."
                if 'temperature' in modified_kwargs:
                    modified_kwargs['temperature'] = 0.1  # More deterministic
            
            if 'timeout' in error_lower or 'limit' in error_lower:
                if 'max_tokens' in modified_kwargs:
                    modified_kwargs['max_tokens'] = min(150, modified_kwargs.get('max_tokens', 300))
            
            if 'rate limit' in error_lower:
                # Will be handled by sleep in retry logic
                pass
            
            # Apply reflection suggestions
            if 'reduce' in reflection_lower and 'tokens' in reflection_lower:
                if 'max_tokens' in modified_kwargs:
                    modified_kwargs['max_tokens'] = max(50, int(modified_kwargs.get('max_tokens', 300) * 0.7))
            
        except Exception as e:
            logger.warning(f"Failed to apply error reflection: {str(e)}")
        
        return modified_kwargs

# Decorator instances
retry_with_reflection = RetryWithReflection(max_attempts=3, min_confidence=0.7)
retry_json_parsing = RetryWithReflection(max_attempts=3, min_confidence=0.8)
retry_critical_operation = RetryWithReflection(max_attempts=5, min_confidence=0.9)

# Convenience decorators
def smart_retry(max_attempts: int = 3, min_confidence: float = 0.7):
    """Decorator factory for custom retry parameters"""
    return RetryWithReflection(max_attempts=max_attempts, min_confidence=min_confidence)

# Pre-configured decorators for different use cases
llm_retry = retry_with_reflection
json_retry = retry_json_parsing
critical_retry = retry_critical_operation 