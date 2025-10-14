"""Database compatibility module for API Gateway"""
from app.DATABASE_CORE import get_db_session, DatabaseWrapper, DatabaseManager

__all__ = ["get_db_session", "DatabaseWrapper", "DatabaseManager"]
