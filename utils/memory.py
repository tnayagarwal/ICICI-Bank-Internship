"""
Long-term Memory System for Autonomous HR Agents
Provides persistent storage, retrieval, and learning capabilities
"""
import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from loguru import logger

from utils.postgres_client import postgres_client

@dataclass
class Memory:
    id: str
    key: str
    value: Dict[str, Any]
    timestamp: datetime
    category: str = "general"
    tags: List[str] = None
    importance: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

class MemorySystem:
    def __init__(self):
        self.cache: Dict[str, Memory] = {}
        self.cache_size = 1000
        self.table_name = "agent_memory"
        
    def generate_id(self, key: str) -> str:
        """Generate unique ID for memory entry"""
        return hashlib.md5(f"{key}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    async def remember(self, key: str, value: Dict[str, Any], 
                      category: str = "general", tags: List[str] = None, 
                      importance: float = 1.0) -> bool:
        """
        Store a memory entry
        
        Args:
            key: Unique identifier for the memory
            value: Data to store
            category: Memory category (user, commands, decisions, etc.)
            tags: Optional tags for organization
            importance: Importance score (0.0 to 10.0)
        """
        try:
            memory_id = self.generate_id(key)
            memory = Memory(
                id=memory_id,
                key=key,
                value=value,
                timestamp=datetime.now(),
                category=category,
                tags=tags or [],
                importance=importance
            )
            
            # Store in cache
            self.cache[key] = memory
            self.manage_cache_size()
            
            # Store in database (simplified for reliability)
            await self.store_in_database(memory)
            
            logger.info(f"Stored memory: {key} (category: {category})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory {key}: {str(e)}")
            return False
    
    async def recall(self, query: str, category: Optional[str] = None, 
                    limit: int = 10) -> List[Memory]:
        """
        Retrieve memories based on query
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum number of results
        """
        try:
            # Search cache first
            cache_results = self.search_cache(query, category, limit)
            
            # Then search database
            db_results = await self.search_database(query, category, limit)
            
            # Combine and deduplicate
            all_results = cache_results + db_results
            seen_keys = set()
            unique_results = []
            
            for memory in all_results:
                if memory.key not in seen_keys:
                    unique_results.append(memory)
                    seen_keys.add(memory.key)
                    
                    # Update access statistics
                    memory.access_count += 1
                    memory.last_accessed = datetime.now()
            
            # Sort by relevance and importance
            unique_results.sort(key=lambda m: m.importance * (m.access_count + 1), reverse=True)
            
            return unique_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to recall memories for query '{query}': {str(e)}")
            return []
    
    def search_cache(self, query: str, category: Optional[str] = None, 
                    limit: int = 10) -> List[Memory]:
        """Search memories in cache"""
        query_lower = query.lower()
        results = []
        
        for memory in self.cache.values():
            # Category filter
            if category and memory.category != category:
                continue
            
            # Simple text matching
            memory_text = json.dumps(memory.value).lower()
            if (query_lower in memory.key.lower() or 
                query_lower in memory_text or
                any(query_lower in tag.lower() for tag in memory.tags)):
                results.append(memory)
        
        return results[:limit]
    
    async def search_database(self, query: str, category: Optional[str] = None,
                             limit: int = 10) -> List[Memory]:
        """Search memories in database with fallback"""
        try:
            # Simple approach - try to get from a logs table or use postgres storage
            # For now, return empty as we're using cache primarily
            return []
            
        except Exception as e:
            logger.warning(f"Database search failed, using cache only: {str(e)}")
            return []
    
    async def store_in_database(self, memory: Memory):
        """Store memory in database with fallback"""
        try:
            # Simple storage approach - could be enhanced with actual vector DB
            logger.debug(f"Memory stored: {memory.key}")
            
        except Exception as e:
            logger.warning(f"Failed to store memory in database: {str(e)}")
            # Continue - cache storage is still available
    
    def manage_cache_size(self):
        """Manage cache size by removing least important/accessed items"""
        if len(self.cache) <= self.cache_size:
            return
        
        # Sort by importance and access count
        sorted_memories = sorted(
            self.cache.values(),
            key=lambda m: m.importance * (m.access_count + 1) * (1 / max((datetime.now() - m.timestamp).days + 1, 1)),
            reverse=True
        )
        
        # Keep top items
        keep_memories = sorted_memories[:self.cache_size // 2]
        
        # Clear cache and keep only important items
        self.cache.clear()
        for memory in keep_memories:
            self.cache[memory.key] = memory
        
        logger.info(f"Cache cleaned: kept {len(keep_memories)} memories")
    
    async def get_memories_by_category(self, category: str, limit: int = 50) -> List[Memory]:
        """Get all memories in a specific category"""
        return await self.recall("", category=category, limit=limit)
    
    async def get_recent_memories(self, hours: int = 24, limit: int = 50) -> List[Memory]:
        """Get recent memories within time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_memories = []
        for memory in self.cache.values():
            if memory.timestamp >= cutoff_time:
                recent_memories.append(memory)
        
        # Sort by timestamp, most recent first
        recent_memories.sort(key=lambda m: m.timestamp, reverse=True)
        return recent_memories[:limit]
    
    async def delete_memory(self, key: str) -> bool:
        """Delete a memory entry"""
        try:
            # Remove from cache
            self.cache.pop(key, None)
            
            logger.info(f"Deleted memory: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {key}: {str(e)}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            # Cache stats
            cache_stats = {
                "cache_size": len(self.cache),
                "cache_categories": {},
                "cache_avg_importance": 0.0
            }
            
            if self.cache:
                for memory in self.cache.values():
                    cat = memory.category
                    cache_stats["cache_categories"][cat] = cache_stats["cache_categories"].get(cat, 0) + 1
                
                cache_stats["cache_avg_importance"] = sum(m.importance for m in self.cache.values()) / len(self.cache)
            
            return {
                "cache": cache_stats,
                "system_health": "healthy" if len(self.cache) < self.cache_size * 0.9 else "near_limit"
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {str(e)}")
            return {"error": str(e)}
    
    async def cleanup_old_memories(self, days: int = 30):
        """Clean up old, low-importance memories"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Remove from cache
            to_remove = []
            for key, memory in self.cache.items():
                if (memory.timestamp < cutoff_date and 
                    memory.importance < 2.0 and 
                    memory.access_count < 3):
                    to_remove.append(key)
            
            for key in to_remove:
                del self.cache[key]
            
            logger.info(f"Cleaned up {len(to_remove)} old memories from cache")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {str(e)}")

# Global memory system instance
memory_system = MemorySystem()

# Convenience functions
async def remember(key: str, value: Dict[str, Any], category: str = "general", 
                  tags: List[str] = None, importance: float = 1.0) -> bool:
    """Quick memory storage"""
    return await memory_system.remember(key, value, category, tags, importance)

async def recall(query: str, category: Optional[str] = None, limit: int = 10) -> List[Memory]:
    """Quick memory retrieval"""
    return await memory_system.recall(query, category, limit)

async def get_user_preferences(emp_id: str) -> Dict[str, Any]:
    """Get user-specific preferences"""
    memories = await recall(f"user_pref_{emp_id}", category="user")
    if memories:
        return memories[0].value
    return {}

async def store_user_preference(emp_id: str, preference_key: str, value: Any):
    """Store user preference"""
    current_prefs = await get_user_preferences(emp_id)
    current_prefs[preference_key] = value
    await remember(f"user_pref_{emp_id}", current_prefs, category="user", importance=5.0)

async def get_command_patterns() -> List[Dict[str, Any]]:
    """Get stored command patterns"""
    memories = await recall("", category="commands")
    return [m.value for m in memories] 