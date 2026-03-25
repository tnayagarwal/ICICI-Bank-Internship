"""
Configuration settings for ICICI Bank HR Management System
========================================================

This module contains all configuration settings for the HR Assistant system including:
- Email configuration for notifications
- LLM (Language Model) settings for AI responses
- Business rules for attendance and leave management
- Performance and system settings
- Error and success message templates
- Autonomous system configuration

The settings use Pydantic for validation and can be overridden via environment variables
or a .env file.

Author: ICICI Bank Development Team
Version: 1.0.0
"""

import os
from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings using Pydantic for validation and environment variable support
    
    All settings can be overridden via environment variables or .env file.
    For example: GMAIL_ADDRESS=your_email@gmail.com
    """
    
    # Email Settings - Used for sending notifications and leave requests
    GMAIL_ADDRESS: str = Field(
        default="x",
        description="Gmail address for sending system notifications"
    )
    GMAIL_APP_PASSWORD: str = Field(
        default="x",
        description="Gmail app password for authentication (use app-specific password)"
    )
    
    # LLM Settings - Configuration for the Gemma language model
    MAX_TOKENS: int = Field(
        default=2048,
        description="Maximum number of tokens for LLM responses"
    )
    TEMPERATURE: float = Field(
        default=0.7,
        description="Temperature setting for LLM responses (0.0 = deterministic, 1.0 = creative)"
    )
    
    # Business Rules - Company-specific policies and constraints
    OFFICE_COORDS: tuple = Field(
        default=(17.424091, 78.336075),
        description="Office coordinates (latitude, longitude) for attendance validation"
    )
    ATTENDANCE_RADIUS_METERS: int = Field(
        default=100,
        description="Maximum distance from office for valid attendance check-in/out"
    )
    WORK_END_TIME: int = Field(
        default=17,
        description="Standard work end time in 24-hour format (17 = 5 PM)"
    )
    OVERTIME_CUTOFF: str = Field(
        default="17:15:00",
        description="Time after which overtime is calculated (HH:MM:SS format)"
    )
    
    # Performance Settings - System optimization parameters
    VECTOR_SEARCH_LIMIT: int = Field(
        default=10,
        description="Maximum number of documents to retrieve in vector search"
    )
    CONVERSATION_MEMORY_LIMIT: int = Field(
        default=10,
        description="Maximum number of conversation turns to remember"
    )
    
    # API Configuration - External service credentials
    GROQ_API_KEY: str = Field(
        default=os.getenv("GROQ_API_KEY", "x"),
        description="API key for Groq LLM service"
    )
    MODEL_NAME: str = Field(
        default=os.getenv("MODEL_NAME", "gemma2-9b-it"),
        description="Name of the LLM model to use"
    )
    
    class Config:
        """Pydantic configuration for environment variable support"""
        env_file = ".env"  # Load settings from .env file
        env_file_encoding = "utf-8"  # File encoding

# Global settings instance - import this in other modules
settings = Settings()

# Agent registry is now managed in agents.json file for dynamic agent configuration

# Error Messages - User-friendly error messages for different scenarios
ERROR_MESSAGES = {
    "gps_unavailable": "🚫 GPS location is required for attendance operations",
    "invalid_employee": "❌ Invalid employee ID. Please check and try again",
    "network_error": "🌐 Network error. Please check your connection",
    "api_error": "⚠️ Service temporarily unavailable. Please try again",
    "permission_denied": "🔒 Insufficient permissions for this operation",
    "invalid_input": "📝 Invalid input format. Please check your request"
}

# Success Messages - Confirmation messages for successful operations
SUCCESS_MESSAGES = {
    "checkin": "✅ Check-in successful at {time}",
    "checkout": "⏰ Check-out recorded at {time}",
    "leave_approved": "🎉 Leave request approved automatically",
    "leave_sent": "📧 Leave request sent to department head",
    "email_sent": "📨 Email sent successfully",
    "request_processed": "✅ Request processed successfully"
} 

# Autonomous System Configuration - Feature flags for advanced functionality
AUTONOMOUS_MODE = False  # Set to True to enable autonomous operations (auto-approval, etc.)
SCHEDULER_ENABLED = False  # Set to True to enable background scheduler for automated tasks 
