#!/usr/bin/env python3
"""
ICICI Bank HR Assistant - Web Frontend
======================================

This Flask-based web application provides a user-friendly interface for the HR Assistant system.
It handles user authentication, chat interactions, and provides real-time feedback to employees.

Features:
- Employee login and session management
- Real-time chat interface with the HR Assistant
- Chat history management
- Employee information display
- System health monitoring
- User feedback collection

The frontend communicates with the backend orchestrator to process user queries and
displays responses in a conversational format.

Author: ICICI Bank Development Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add current directory to Python path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent))

# Import required modules with error handling
try:
    from flask import Flask, render_template, request, jsonify, session, redirect, url_for
    from agents.langgraph_orchestrator import process_dynamic_request_full
    from utils.postgres_client import postgres_client
    from datetime import datetime
    import json
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Initialize Flask application with static and template folders
app = Flask(__name__, 
           static_folder='frontend_consolidated/static', 
           template_folder='frontend_consolidated/templates')

# Set secret key for session management (should be stored securely in production)
app.secret_key = 'hr_assistant_secret_key_2024'

def validate_employee(emp_id: str) -> dict:
    """
    Validate employee ID and return employee information
    
    Args:
        emp_id (str): Employee ID to validate
        
    Returns:
        dict: Employee information if valid, None otherwise
    """
    try:
        employee_info = postgres_client.get_employee(emp_id)
        return employee_info
    except Exception as e:
        print(f"❌ Error validating employee: {e}")
        return None

@app.route('/')
def index():
    """
    Main page - show login or chat based on session status
    
    Returns:
        str: Rendered template (login.html or chat.html)
    """
    if not session.get('logged_in'):
        return render_template('login.html')
    
    return render_template('chat.html', 
                         employee_info=session.get('employee_info', {}),
                         emp_id=session.get('emp_id', ''))

@app.route('/login', methods=['POST'])
def login():
    """
    Handle employee login
    
    Expects JSON with 'emp_id' field
    Returns JSON response with success status and message
    """
    try:
        data = request.get_json()
        emp_id = data.get('emp_id', '').strip()
        
        # Validate input
        if not emp_id:
            return jsonify({"success": False, "message": "Please enter your Employee ID"})
        
        # Validate employee
        employee_info = validate_employee(emp_id)
        if employee_info:
            # Set session data
            session['logged_in'] = True
            session['emp_id'] = emp_id
            session['employee_info'] = employee_info
            session['chat_history'] = []
            
            return jsonify({
                "success": True, 
                "message": f"Welcome, {employee_info['name']}!",
                "employee_info": employee_info
            })
        else:
            return jsonify({"success": False, "message": "Employee not found. Please check your ID."})
                
    except Exception as e:
        return jsonify({"success": False, "message": f"Login error: {str(e)}"})

@app.route('/logout')
def logout():
    """
    Handle user logout by clearing session data
    
    Returns:
        redirect: Redirects to index page
    """
    session.clear()
    return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat messages with the HR Assistant
    
    Expects JSON with 'message' field
    Returns JSON response with assistant's reply
    """
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "Please login first"})
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        # Validate input
        if not user_message:
            return jsonify({"success": False, "message": "Please enter a message"})
        
        # Get employee ID from session
        emp_id = session.get('emp_id')
        
        # Process the request through the orchestrator
        orchestrator_response = process_dynamic_request_full(emp_id, user_message)
        
        # Extract the message from the orchestrator response
        message = orchestrator_response.get("message", "No response available")
        
        # Store in chat history
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'assistant_message': message,
            'success': orchestrator_response.get('success', False)
        }
        
        session['chat_history'].append(chat_entry)
        
        # Keep only last 50 messages to prevent session bloat
        if len(session['chat_history']) > 50:
            session['chat_history'] = session['chat_history'][-50:]
        
        return jsonify({
            "success": True,
            "message": message,
            "timestamp": chat_entry['timestamp']
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": f"Chat error: {str(e)}"})

@app.route('/history')
def history():
    """
    Get chat history for the current session
    
    Returns:
        JSON: Chat history array
    """
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "Please login first"})
    
    return jsonify({
        "success": True,
        "history": session.get('chat_history', [])
    })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """
    Clear chat history for the current session
    
    Returns:
        JSON: Success status
    """
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "Please login first"})
    
    session['chat_history'] = []
    return jsonify({"success": True, "message": "Chat history cleared"})

@app.route('/employee_info')
def employee_info():
    """
    Get current employee information
    
    Returns:
        JSON: Employee information
    """
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "Please login first"})
    
    return jsonify({
        "success": True,
        "employee_info": session.get('employee_info', {})
    })

@app.route('/health')
def health():
    """
    System health check endpoint
    
    Returns:
        JSON: System status information
    """
    try:
        # Test database connection
        test_emp = postgres_client.get_employee("12345")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    return jsonify({
        "status": "running",
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Handle user feedback for chat interactions
    
    Expects JSON with 'rating' and 'comment' fields
    Returns JSON response with success status
    """
    if not session.get('logged_in'):
        return jsonify({"success": False, "message": "Please login first"})
    
    try:
        data = request.get_json()
        rating = data.get('rating', 0)
        comment = data.get('comment', '')
        
        # Store feedback (in production, this would go to a database)
        feedback_data = {
            'emp_id': session.get('emp_id'),
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        
        # TODO: Implement feedback storage to database
        print(f"Feedback received: {feedback_data}")
        
        return jsonify({
            "success": True,
            "message": "Thank you for your feedback!"
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": f"Feedback error: {str(e)}"})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"success": False, "message": "Page not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"success": False, "message": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask application in development mode
    print("🚀 Starting ICICI Bank HR Assistant Frontend...")
    print("📱 Web interface will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 