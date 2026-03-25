#!/usr/bin/env python3
"""
ICICI Bank HR Assistant - Main Application Entry Point
=====================================================

This is the main entry point for the ICICI Bank HR Assistant backend system.
The application provides a unified multi-agent system for employee services including:
- Leave management
- Attendance tracking
- Equipment requests and complaints
- Policy information and general HR queries

The system uses LangGraph orchestrator to intelligently route user queries to appropriate
specialized agents based on the query content and context.

Author: ICICI Bank Development Team
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent))

# Import core system components
from agents.langgraph_orchestrator import process_dynamic_request_full
from utils.postgres_client import postgres_client
from loguru import logger

def main():
    """
    Main application entry point
    
    This function:
    1. Tests database connectivity
    2. Validates the orchestrator system
    3. Provides startup feedback
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("ICICI Bank HR Assistant - Backend System")
    print("=" * 50)
    
    # Test database connection to ensure the system can access employee data
    try:
        # Simple test by trying to get an employee record
        test_emp = postgres_client.get_employee("12345")
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database error: {e}")
        print("Please ensure PostgreSQL is running and database is properly configured")
        return 1
    
    # Test the orchestrator system to ensure it can process requests
    try:
        test_response = process_dynamic_request_full("12345", "Hello")
        if test_response.get('success'):
            print("✅ Orchestrator system operational")
        else:
            print("❌ Orchestrator system error")
            return 1
    except Exception as e:
        print(f"❌ Orchestrator error: {e}")
        print("Please check agent configurations and dependencies")
        return 1
    
    # System is ready for operation
    print("\n🎉 System ready for operation")
    print("\nNext steps:")
    print("• To start web interface: python frontend.py")
    print("• To run system tests: python -m pytest tests/")
    print("• To check system status: python -c 'from app import main; main()'")
    
    return 0

if __name__ == "__main__":
    # Exit with appropriate code based on system status
    sys.exit(main()) 