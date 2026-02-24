"""
Smart Cache Manager for Stock Data API
Reduces API calls by caching responses with intelligent TTL strategies
"""
import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Intelligent caching system to minimize API calls
    
    Cache TTL Strategy:
    - Stock historical data: 1 hour (market data updates periodically)
    - Quote data: 5 minutes (during market hours), 1 hour (after hours)
    - News: 2 hours (news doesn't change frequently)
    - Sentiment: 4 hours (sentiment is derived metric)
    - Company profile: 24 hours (fundamental data rarely changes)
    - Search results: 1 hour (ticker symbols are stable)
    """
    
    def __init__(self, cache_dir: str = 'cache'):
        """Initialize cache manager with directory"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Cache manager initialized at {cache_dir}")
    
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate unique cache key from endpoint and parameters"""
        # Sort params for consistent hashing
        sorted_params = sorted(params.items())
        param_str = json.dumps(sorted_params, sort_keys=True)
        key_input = f"{endpoint}:{param_str}"
        return hashlib.md5(key_input.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _is_market_hours(self) -> bool:
        """Check if current time is during US market hours (9:30 AM - 4:00 PM EST)"""
        now = datetime.now()
        # Simplified check - could be enhanced with timezone awareness
        hour = now.hour
        weekday = now.weekday()
        # Monday=0, Friday=4
        is_weekday = weekday < 5
        # Rough approximation of market hours
        is_trading_time = 14 <= hour < 21  # Approximate EST in UTC
        return is_weekday and is_trading_time
    
    def get(self, endpoint: str, params: Dict[str, Any], ttl_seconds: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached data if valid
        
        Args:
            endpoint: API endpoint identifier
            params: Request parameters
            ttl_seconds: Time-to-live in seconds (None = no TTL check, return any cached data)
            
        Returns:
            Cached data if valid, None otherwise
        """
        cache_key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # If ttl_seconds is None, return cached data regardless of age
            if ttl_seconds is None:
                logger.info(f"Cache HIT for {endpoint} (stale mode, no TTL check)")
                return cached['data']
            
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cached['timestamp'])
            age_seconds = (datetime.now() - cached_time).total_seconds()
            
            if age_seconds < ttl_seconds:
                logger.info(f"Cache HIT for {endpoint} (age: {age_seconds:.0f}s)")
                return cached['data']
            else:
                logger.info(f"Cache EXPIRED for {endpoint} (age: {age_seconds:.0f}s, TTL: {ttl_seconds}s)")
                # Don't delete - keep for stale fallback
                return None
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Cache read error: {e}")
            # Remove corrupted cache
            try:
                os.remove(cache_path)
            except:
                pass
            return None
    
    def set(self, endpoint: str, params: Dict[str, Any], data: Any) -> None:
        """
        Store data in cache
        
        Args:
            endpoint: API endpoint identifier
            params: Request parameters
            data: Data to cache
        """
        cache_key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)
        
        # Custom encoder for Timestamp and datetime
        def json_serial(obj):
            from datetime import datetime
            import pandas as pd
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
            
        try:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
                
            cache_entry = {
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint,
                'data': data
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2, default=json_serial)
            
            logger.info(f"Cache SET for {endpoint}")
            
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def clear_expired(self) -> int:
        """
        Clean up expired cache files
        
        Returns:
            Number of files removed
        """
        removed = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    # Remove files older than 24 hours
                    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_age > timedelta(hours=24):
                        os.remove(filepath)
                        removed += 1
                except:
                    pass
            
            if removed > 0:
                logger.info(f"Cleaned up {removed} expired cache files")
                
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
        
        return removed
    
    def clear_all(self) -> int:
        """
        Clear all cache files
        
        Returns:
            Number of files removed
        """
        removed = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
                    removed += 1
            logger.info(f"Cleared {removed} cache files")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            total_files = len(cache_files)
            
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) 
                for f in cache_files
            )
            
            return {
                'total_files': total_files,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_dir': self.cache_dir
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {'error': str(e)}


# Global cache instance
_cache = CacheManager()

def get_cache() -> CacheManager:
    """Get global cache manager instance"""
    return _cache
