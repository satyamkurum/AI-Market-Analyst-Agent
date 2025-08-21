# app/core/session_manager.py

import logging
import uuid
import time
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages session lifecycle and provides unique session IDs."""
    
    def __init__(self, session_timeout_minutes: int = 60):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new session with a unique ID.
        
        Args:
            user_id: Optional user identifier for tracking
            
        Returns:
            Unique session ID
        """
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'user_id': user_id,
            'status': 'active'
        }
        
        logger.info(f"Created new session: {session_id} for user: {user_id or 'anonymous'}")
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """
        Validate if a session exists and is not expired.
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            True if session is valid, False otherwise
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Check if session expired
        if datetime.now() - session['last_accessed'] > self.session_timeout:
            logger.warning(f"Session expired: {session_id}")
            self.delete_session(session_id)
            return False
        
        # Update last accessed time
        session['last_accessed'] = datetime.now()
        return True
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session and clean up its resources."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up all expired sessions and return count removed."""
        expired_count = 0
        current_time = datetime.now()
        
        for session_id in list(self.sessions.keys()):
            session = self.sessions[session_id]
            if current_time - session['last_accessed'] > self.session_timeout:
                self.delete_session(session_id)
                expired_count += 1
        
        logger.info(f"Cleaned up {expired_count} expired sessions")
        return expired_count
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a session."""
        return self.sessions.get(session_id)
    
    def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)

# Global session manager instance
session_manager = SessionManager()