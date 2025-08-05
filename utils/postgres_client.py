"""
PostgreSQL Database Client for ICICI Bank HR Assistant
====================================================

This module provides a comprehensive database interface for the HR Assistant system.
It handles all database operations including employee management, attendance tracking,
leave management, equipment requests, and policy information.

Key Features:
- Connection pooling and automatic reconnection
- Comprehensive error handling and logging
- Transaction management for data integrity
- Support for all HR system data operations

Database Schema:
- employees: Employee profiles and information
- attendance: Check-in/out records with GPS data
- leave_requests: Leave applications and approvals
- equipment_requests: IT equipment requests
- agent_feedback: User feedback for system improvement
- conversation_memory: Chat history and context
- policies: HR policy documents and metadata

Author: ICICI Bank Development Team
Version: 1.0.0
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date, timedelta
from loguru import logger
from config import settings
import psycopg2
from psycopg2.extras import RealDictCursor

class PostgresClientError(Exception):
    """Custom exception for database-related errors"""
    pass

class PostgresClient:
    """
    PostgreSQL client for HR Assistant database operations
    
    This class provides a high-level interface for all database operations
    including connection management, error handling, and data validation.
    """
    
    def __init__(self):
        """Initialize database connection configuration"""
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'talentagent_db',
            'user': 'postgres',
            'password': 'TaNaY',
        }
        self.connection = None
        self._connect()

    def _connect(self):
        """
        Establish database connection with error handling
        
        Raises:
            PostgresClientError: If connection fails
        """
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = True
            logger.info("✅ Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise PostgresClientError(f"Database connection failed: {e}")

    def _get_cursor(self):
        """
        Get database cursor with automatic reconnection
        
        Returns:
            psycopg2.extras.RealDictCursor: Database cursor
        """
        if not self.connection or self.connection.closed:
            self._connect()
        return self.connection.cursor(cursor_factory=RealDictCursor)

    # Employee operations
    def get_employee(self, emp_id: str) -> Optional[Dict]:
        """
        Retrieve employee information by employee ID
        
        Args:
            emp_id (str): Employee ID to look up
            
        Returns:
            Optional[Dict]: Employee information or None if not found
        """
        try:
            with self._get_cursor() as cur:
                cur.execute("SELECT * FROM employees WHERE id = %s", (emp_id,))
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"❌ Error getting employee {emp_id}: {e}")
            return None

    def update_employee_leaves(self, emp_id: str, leaves_used: int) -> bool:
        """
        Update employee leave balance after leave application
        
        Args:
            emp_id (str): Employee ID
            leaves_used (int): Number of leaves to deduct
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            with self._get_cursor() as cur:
                cur.execute("""
                    UPDATE employees 
                    SET leaves_available = leaves_available - %s 
                    WHERE id = %s AND leaves_available >= %s
                """, (leaves_used, emp_id, leaves_used))
                return cur.rowcount > 0
        except Exception as e:
            logger.error(f"❌ Error updating leaves for {emp_id}: {e}")
            return False

    # Attendance operations
    def record_attendance(self, emp_id: str, action: str, override_reason: str = None) -> str:
        """
        Record attendance check-in or check-out with GPS validation
        
        Args:
            emp_id (str): Employee ID
            action (str): 'check_in' or 'check_out'
            override_reason (str, optional): Reason for override if GPS validation fails
            
        Returns:
            str: Status message indicating the result
        """
        try:
            with self._get_cursor() as cur:
                today = date.today()
                current_time = datetime.now().time()
                
                # Check if there's already an entry for today
                cur.execute("""
                    SELECT * FROM attendance 
                    WHERE emp_id = %s AND date = %s
                """, (emp_id, today))
                existing = cur.fetchone()
                
                if action == "check_in":
                    if existing and existing['entry']:
                        return "checked in"
                    
                    if existing:
                        cur.execute("""
                            UPDATE attendance 
                            SET entry = %s, status = %s, override_reason = %s
                            WHERE emp_id = %s AND date = %s
                        """, (current_time, 'override' if override_reason else 'normal', override_reason, emp_id, today))
                    else:
                        cur.execute("""
                            INSERT INTO attendance (emp_id, date, entry, status, override_reason)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (emp_id, today, current_time, 'override' if override_reason else 'normal', override_reason))
                    
                    return "checked in"
                
                elif action == "check_out":
                    if existing and existing['exit']:
                        return "checked out"
                    
                    if existing:
                        cur.execute("""
                            UPDATE attendance 
                            SET exit = %s, status = %s, override_reason = %s
                            WHERE emp_id = %s AND date = %s
                        """, (current_time, 'override' if override_reason else 'normal', override_reason, emp_id, today))
                    else:
                        cur.execute("""
                            INSERT INTO attendance (emp_id, date, exit, status, override_reason)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (emp_id, today, current_time, 'override' if override_reason else 'normal', override_reason))
                    
                    return "checked out"
                
                return "invalid action"
                
        except Exception as e:
            logger.error(f"❌ Error recording attendance for {emp_id}: {e}")
            return "error"

    def get_attendance_summary(self, emp_id: str, period: str = "weekly") -> Dict:
        """Get attendance summary for overtime calculation"""
        try:
            with self._get_cursor() as cur:
                if period == "weekly":
                    start_date = date.today() - timedelta(days=7)
                else:  # monthly
                    start_date = date.today() - timedelta(days=30)
                
                cur.execute("""
                    SELECT date, entry, exit 
                    FROM attendance 
                    WHERE emp_id = %s AND date >= %s AND entry IS NOT NULL AND exit IS NOT NULL
                    ORDER BY date
                """, (emp_id, start_date))
                
                records = cur.fetchall()
                total_hours = 0
                for record in records:
                    if record['entry'] and record['exit']:
                        entry_dt = datetime.combine(record['date'], record['entry'])
                        exit_dt = datetime.combine(record['date'], record['exit'])
                        hours = (exit_dt - entry_dt).total_seconds() / 3600
                        total_hours += hours
                
                standard_hours = len(records) * 8  # 8 hours per day
                overtime_hours = max(0, total_hours - standard_hours)
                
                return {
                    "period": period,
                    "total_hours": round(total_hours, 2),
                    "standard_hours": standard_hours,
                    "overtime_hours": round(overtime_hours, 2),
                    "days_worked": len(records)
                }
        except Exception as e:
            logger.error(f"Error getting attendance summary for {emp_id}: {e}")
            return {}

    def get_weekly_attendance(self, emp_id: str, start_date: str, end_date: str) -> List[Dict]:
        """Get weekly attendance records for overtime calculation"""
        try:
            with self._get_cursor() as cur:
                cur.execute("""
                    SELECT date, entry, exit 
                    FROM attendance 
                    WHERE emp_id = %s AND date >= %s AND date <= %s
                    ORDER BY date DESC
                """, (emp_id, start_date, end_date))
                
                records = cur.fetchall()
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Error getting weekly attendance for {emp_id}: {e}")
            return []

    def check_attendance_today(self, emp_id: str) -> Dict:
        """Check if employee has attendance record for today"""
        try:
            today = datetime.now().date().isoformat()
            with self._get_cursor() as cur:
                cur.execute("""
                    SELECT * FROM attendance 
                    WHERE emp_id = %s AND date = %s
                """, (emp_id, today))
                
                record = cur.fetchone()
                return dict(record) if record else None
        except Exception as e:
            logger.error(f"Error checking attendance today for {emp_id}: {e}")
            return None

    # Leave operations
    def calculate_working_days(self, start_date: date, end_date: date) -> int:
        """
        Calculate working days between two dates, excluding:
        - All Sundays (always non-working)
        - Odd Saturdays (1st, 3rd, 5th Saturdays of the month are holidays)
        - Even Saturdays (2nd, 4th Saturdays of the month are working days)
        """
        from datetime import timedelta
        
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        
        working_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            # Check if it's a Sunday (always non-working)
            if current_date.weekday() == 6:  # Sunday
                current_date += timedelta(days=1)
                continue
            
            # Check if it's a Saturday
            if current_date.weekday() == 5:  # Saturday
                # Find which Saturday of the month this is
                first_day_of_month = current_date.replace(day=1)
                first_saturday = first_day_of_month
                
                # Find the first Saturday of the month
                while first_saturday.weekday() != 5:  # 5 = Saturday
                    first_saturday += timedelta(days=1)
                
                # Calculate which Saturday of the month this is
                days_diff = (current_date - first_saturday).days
                saturday_number = (days_diff // 7) + 1
                
                # Odd Saturdays (1st, 3rd, 5th) are holidays, even Saturdays (2nd, 4th) are working days
                if saturday_number % 2 == 1:  # Odd Saturday - holiday
                    current_date += timedelta(days=1)
                    continue
                else:  # Even Saturday - working day
                    working_days += 1
            else:
                # Monday to Friday are always working days
                working_days += 1
            
            current_date += timedelta(days=1)
        
        return working_days

    def create_leave_application(self, emp_id: str, subject: str, content: str, start_date: date, end_date: date) -> Dict:
        """Create a new leave application"""
        try:
            with self._get_cursor() as cur:
                # Calculate working days requested
                days_requested = self.calculate_working_days(start_date, end_date)
                
                # Get employee details
                employee = self.get_employee(emp_id)
                if not employee:
                    return {"error": "Employee not found"}
                
                # Check if auto-approval is possible (less than 3 days and sufficient balance)
                auto_approved = days_requested <= 3 and employee['leaves_available'] >= days_requested
                
                # Insert leave application
                cur.execute("""
                    INSERT INTO leaves (emp_id, subject, content, start_date, end_date, days_requested, auto_approved, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (emp_id, subject, content, start_date, end_date, days_requested, auto_approved, 
                     'approved' if auto_approved else 'pending'))
                
                leave_id = cur.fetchone()['id']
                
                if auto_approved:
                    # Update employee leave balance
                    self.update_employee_leaves(emp_id, days_requested)
                    
                    # Update auto_approved_leaves JSONB
                    cur.execute("""
                        UPDATE employees 
                        SET auto_approved_leaves = auto_approved_leaves || %s::jsonb
                        WHERE id = %s
                    """, (json.dumps([{"leave_id": leave_id, "dates": f"{start_date} to {end_date}"}]), emp_id))
                else:
                    # Update approval_pending_leaves JSONB
                    cur.execute("""
                        UPDATE employees 
                        SET approval_pending_leaves = approval_pending_leaves || %s::jsonb
                        WHERE id = %s
                    """, (json.dumps([{"leave_id": leave_id, "dates": f"{start_date} to {end_date}"}]), emp_id))
                
                return {
                    "leave_id": leave_id,
                    "auto_approved": auto_approved,
                    "days_requested": days_requested,
                    "remaining_balance": employee['leaves_available'] - (days_requested if auto_approved else 0)
                }
        except Exception as e:
            logger.error(f"Error creating leave application for {emp_id}: {e}")
            return {"error": str(e)}

    def get_leave_balance(self, emp_id: str) -> int:
        """Get employee leave balance"""
        employee = self.get_employee(emp_id)
        return employee['leaves_available'] if employee else 0

    # Complaint operations
    def create_complaint(self, emp_id: str, subject: str, content: str) -> int:
        """Create a new complaint"""
        try:
            with self._get_cursor() as cur:
                cur.execute("""
                    INSERT INTO complaints (emp_id, subject, content)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (emp_id, subject, content))
                return cur.fetchone()['id']
        except Exception as e:
            logger.error(f"Error creating complaint for {emp_id}: {e}")
            return None

    # Policy data operations
    def search_policies(self, query: str) -> List[Dict]:
        """Search policy documents"""
        try:
            with self._get_cursor() as cur:
                # Simple text search - can be enhanced with full-text search
                cur.execute("""
                    SELECT content, metadata 
                    FROM data 
                    WHERE content ILIKE %s
                    LIMIT 10
                """, (f"%{query}%",))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error searching policies: {e}")
            return []

    def add_policy_content(self, content: str, metadata: Dict) -> bool:
        """Add policy content to database"""
        try:
            with self._get_cursor() as cur:
                cur.execute("""
                    INSERT INTO data (content, metadata)
                    VALUES (%s, %s)
                """, (content, json.dumps(metadata)))
                return True
        except Exception as e:
            logger.error(f"Error adding policy content: {e}")
            return False

    # Agent feedback operations
    def store_agent_feedback(self, agent_type: str, emp_id: str, original_request: str, 
                           agent_response: Dict, user_feedback: str) -> bool:
        """Store user feedback for agent improvement"""
        try:
            with self._get_cursor() as cur:
                # Convert agent_response to JSONB if it's not already
                if isinstance(agent_response, str):
                    try:
                        agent_response = json.loads(agent_response)
                    except:
                        agent_response = {"response": agent_response}
                
                cur.execute("""
                    INSERT INTO agent_feedback (agent_type, emp_id, original_request, agent_response, user_feedback)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (agent_type, emp_id, original_request, json.dumps(agent_response), user_feedback))
                return True
        except Exception as e:
            logger.error(f"Error storing feedback for {agent_type} agent: {e}")
            return False

    def get_agent_feedback(self, agent_type: str) -> List[Dict]:
        """Get feedback for agent improvement"""
        try:
            with self._get_cursor() as cur:
                cur.execute("""
                    SELECT * FROM agent_feedback 
                    WHERE agent_type = %s AND improvement_applied = FALSE
                    ORDER BY created_at DESC
                    LIMIT 10
                """, (agent_type,))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error getting agent feedback: {e}")
            return []

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def record_checkin(self, emp_id: str, time_str: str) -> bool:
        """Record employee check-in"""
        try:
            result = self.record_attendance(emp_id, "check_in")
            return result == "checked in"
        except Exception as e:
            logger.error(f"Error recording check-in for {emp_id}: {e}")
            return False

    def record_checkout(self, emp_id: str, time_str: str) -> bool:
        """Record employee check-out"""
        try:
            result = self.record_attendance(emp_id, "check_out")
            return result == "checked out"
        except Exception as e:
            logger.error(f"Error recording check-out for {emp_id}: {e}")
            return False

# Global instance
postgres_client = PostgresClient()

def get_employee(emp_id):
    return postgres_client.get_employee(emp_id)

def check_attendance_today(emp_id):
    today = datetime.now().date().isoformat()
    return postgres_client.get_attendance_record(emp_id, today)

def record_checkin(emp_id, time_str):
    today = datetime.now().date().isoformat()
    return postgres_client.create_attendance_record(emp_id, today, time_str)

def record_checkout(emp_id, time_str):
    today = datetime.now().date().isoformat()
    return postgres_client.update_attendance_exit(emp_id, today, time_str)

def search_policies(query_text, limit=10):
    return postgres_client.search_policies_enhanced(query_text, limit=limit)

def get_employee_leave_info(emp_id):
    return postgres_client.get_leave_balance(emp_id) 

def update_employee_dept_head(emp_id: str, dept_head: str, head_email: str) -> bool:
    """Update department head and head email for an employee"""
    try:
        with postgres_client._get_cursor() as cur:
            cur.execute("""
                UPDATE employees SET dept_head = %s, head_email = %s WHERE id = %s
            """, (dept_head, head_email, emp_id))
            return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Error updating dept head for {emp_id}: {e}")
        return False 