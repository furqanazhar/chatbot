"""
Conversation database module for storing user/agent conversations in SQLite.

This module provides functionality to:
- Initialize the conversation database
- Store user and agent messages
- Retrieve conversation history
"""

import sqlite3
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = "conversations.db"


class ConversationDB:
    """SQLite database handler for storing conversations."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize the conversation database.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create the conversations table if it doesn't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        assistance_required INTEGER DEFAULT 0,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better query performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_id 
                    ON conversations(session_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON conversations(timestamp)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_role 
                    ON conversations(role)
                """)
                
                conn.commit()
                logger.info(f"✅ Conversation database initialized: {self.db_path}")
                
        except Exception as e:
            logger.error(f"❌ Error initializing conversation database: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def save_message(self, 
                    session_id: str, 
                    role: str, 
                    content: str, 
                    assistance_required: bool = False,
                    metadata: Optional[Dict] = None) -> int:
        """Save a message to the database.
        
        Args:
            session_id (str): Unique session identifier
            role (str): Message role ('user' or 'assistant')
            content (str): Message content
            assistance_required (bool): Whether human assistance was required
            metadata (Optional[Dict]): Additional metadata to store as JSON
            
        Returns:
            int: The ID of the inserted message
        """
        try:
            timestamp = datetime.now()
            metadata_json = None
            if metadata:
                import json
                metadata_json = json.dumps(metadata)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO conversations 
                    (session_id, timestamp, role, content, assistance_required, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    timestamp,
                    role,
                    content,
                    1 if assistance_required else 0,
                    metadata_json
                ))
                conn.commit()
                message_id = cursor.lastrowid
                logger.debug(f"Saved {role} message (ID: {message_id}) for session {session_id}")
                return message_id
                
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            raise
    
    def get_conversation_history(self, 
                                 session_id: str, 
                                 limit: Optional[int] = None) -> List[Dict]:
        """Retrieve conversation history for a session.
        
        Args:
            session_id (str): Session identifier
            limit (Optional[int]): Maximum number of messages to retrieve
            
        Returns:
            List[Dict]: List of message dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                query = """
                    SELECT id, session_id, timestamp, role, content, 
                           assistance_required, metadata, created_at
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, (session_id,))
                rows = cursor.fetchall()
                
                messages = []
                for row in rows:
                    metadata = None
                    if row['metadata']:
                        import json
                        try:
                            metadata = json.loads(row['metadata'])
                        except json.JSONDecodeError:
                            pass
                    
                    messages.append({
                        'id': row['id'],
                        'session_id': row['session_id'],
                        'timestamp': row['timestamp'],
                        'role': row['role'],
                        'content': row['content'],
                        'assistance_required': bool(row['assistance_required']),
                        'metadata': metadata,
                        'created_at': row['created_at']
                    })
                
                return messages
                
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def get_all_sessions(self) -> List[str]:
        """Get all unique session IDs.
        
        Returns:
            List[str]: List of session IDs
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT session_id 
                    FROM conversations 
                    ORDER BY MIN(timestamp) DESC
                """)
                rows = cursor.fetchall()
                return [row['session_id'] for row in rows]
                
        except Exception as e:
            logger.error(f"Error retrieving sessions: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get conversation statistics.
        
        Returns:
            Dict: Statistics about conversations
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Total messages
                cursor.execute("SELECT COUNT(*) as count FROM conversations")
                total_messages = cursor.fetchone()['count']
                
                # Total sessions
                cursor.execute("SELECT COUNT(DISTINCT session_id) as count FROM conversations")
                total_sessions = cursor.fetchone()['count']
                
                # Messages by role
                cursor.execute("""
                    SELECT role, COUNT(*) as count 
                    FROM conversations 
                    GROUP BY role
                """)
                messages_by_role = {row['role']: row['count'] for row in cursor.fetchall()}
                
                # Assistance required count
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM conversations 
                    WHERE assistance_required = 1
                """)
                assistance_required = cursor.fetchone()['count']
                
                return {
                    'total_messages': total_messages,
                    'total_sessions': total_sessions,
                    'messages_by_role': messages_by_role,
                    'assistance_required': assistance_required
                }
                
        except Exception as e:
            logger.error(f"Error retrieving statistics: {e}")
            return {}
    
    def delete_session(self, session_id: str) -> bool:
        """Delete all messages for a session.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
                conn.commit()
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} messages for session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False

