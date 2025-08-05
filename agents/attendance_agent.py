# Attendance Agent using LangGraph
# Handles check-in, check-out, and overtime calculations (assumes person is always in office)
from datetime import datetime, date, time as datetime_time, timedelta
from typing import Dict, Any, List, Optional, TypedDict, Literal
from zoneinfo import ZoneInfo
from langgraph.graph import StateGraph, END
from loguru import logger
from config import settings, SUCCESS_MESSAGES, ERROR_MESSAGES
from utils.groq_client import groq_client
from utils.postgres_client import postgres_client
from utils.email_client import notify_attendance

# Import autonomous components
try:
    from utils.retry_decorator import retry_with_reflection
    AUTONOMOUS_FEATURES = True
except ImportError:
    AUTONOMOUS_FEATURES = False
    # Fallback decorator that does nothing
    def retry_with_reflection(func):
        return func

class AttendanceState(TypedDict):
    emp_id: str
    operation: str
    current_time: str
    current_date: str
    employee_info: Optional[dict]
    attendance_record: Optional[dict]
    can_proceed: bool
    action_taken: str
    weekly_records: List[dict]
    overtime_summary: str
    overtime_query: str
    overtime_period: str
    result_message: str
    email_sent: bool
    error: Optional[str]

class AttendanceAgent:
    def __init__(self):
        self.cutoff_time = datetime.strptime(settings.OVERTIME_CUTOFF, "%H:%M:%S").time()
        
    def _get_ist_time(self) -> datetime:
        """Get current time in IST"""
        return datetime.now(ZoneInfo("Asia/Kolkata"))
    
    def initialize_state(self, state: AttendanceState) -> AttendanceState:
        """Initialize attendance state with current time (assumes person is always in office)"""
        try:
            current_dt = self._get_ist_time()
            state["current_time"] = current_dt.strftime("%H:%M:%S")
            state["current_date"] = current_dt.date().isoformat()
            
            # Validate employee
            employee_info = postgres_client.get_employee(state["emp_id"])
            if not employee_info:
                state["error"] = ERROR_MESSAGES["invalid_employee"]
                state["can_proceed"] = False
                return state
            
            state["employee_info"] = employee_info
            
            # Always assume person is in office - no location verification needed
            state["can_proceed"] = True
            logger.info(f"Attendance state initialized for {state['emp_id']}, operation: {state['operation']}")
            
        except Exception as e:
            logger.error(f"State initialization error: {str(e)}")
            state["error"] = f"Initialization failed: {str(e)}"
            state["can_proceed"] = False
        
        return state
    
    def check_existing_record(self, state: AttendanceState) -> AttendanceState:
        """Check existing attendance record for today"""
        try:
            if state["operation"] == "overtime":
                # Parse overtime query for time range
                query = state.get("overtime_query", "weekly")
                
                if "month" in query.lower():
                    # Get monthly records (last 30 days)
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=29)
                elif "week" in query.lower():
                    # Get weekly records (last 7 days)
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=6)
                else:
                    # Default to weekly
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=6)
                
                weekly_records = postgres_client.get_weekly_attendance(
                    state["emp_id"], 
                    start_date.isoformat(), 
                    end_date.isoformat()
                )
                state["weekly_records"] = weekly_records
                state["overtime_period"] = "monthly" if "month" in query.lower() else "weekly"
                return state
            
            # For check-in/check-out, get today's record
            attendance_record = postgres_client.check_attendance_today(state["emp_id"])
            state["attendance_record"] = attendance_record
            
            logger.info(f"Attendance record check completed for {state['emp_id']}")
            
        except Exception as e:
            logger.error(f"Record check error: {str(e)}")
            state["error"] = f"Failed to check attendance record: {str(e)}"
            state["can_proceed"] = False
        
        return state
    
    def make_decision(self, state: AttendanceState) -> AttendanceState:
        """Use LLM to make attendance decisions"""
        try:
            if state["operation"] == "checkin":
                if state["attendance_record"]:
                    # Already checked in today
                    state["action_taken"] = "ALREADY_CHECKED_IN"
                    state["result_message"] = f"⚠️ Already checked in today at {state['attendance_record']['entry']}"
                else:
                    # Proceed with check-in
                    state["action_taken"] = "PROCEED_CHECKIN"
                    
            elif state["operation"] == "checkout":
                if not state["attendance_record"]:
                    state["action_taken"] = "NO_CHECKIN_RECORD"
                    state["result_message"] = "❌ No check-in record found for today"
                elif state["attendance_record"].get("exit"):
                    state["action_taken"] = "ALREADY_CHECKED_OUT"
                    state["result_message"] = f"⚠️ Already checked out today at {state['attendance_record']['exit']}"
                else:
                    state["action_taken"] = "PROCEED_CHECKOUT"
                    
            elif state["operation"] == "overtime":
                state["action_taken"] = "CALCULATE_OVERTIME"
            
            logger.info(f"Decision made for {state['emp_id']}: {state['action_taken']}")
            
        except Exception as e:
            logger.error(f"Decision making error: {str(e)}")
            state["error"] = f"Decision making failed: {str(e)}"
            state["can_proceed"] = False
        
        return state
    
    def execute_action(self, state: AttendanceState) -> AttendanceState:
        """Execute the determined action"""
        try:
            if state["action_taken"] == "PROCEED_CHECKIN":
                success = postgres_client.record_checkin(state["emp_id"], state["current_time"])
                if success:
                    state["result_message"] = SUCCESS_MESSAGES["checkin"].format(time=state["current_time"])
                else:
                    state["error"] = "Failed to record check-in"
                    return state
                    
            elif state["action_taken"] == "PROCEED_CHECKOUT":
                success = postgres_client.record_checkout(state["emp_id"], state["current_time"])
                if success:
                    state["result_message"] = SUCCESS_MESSAGES["checkout"].format(time=state["current_time"])
                else:
                    state["error"] = "Failed to record check-out"
                    return state
                    
            elif state["action_taken"] == "CALCULATE_OVERTIME":
                overtime_summary = self._calculate_overtime(state["weekly_records"], state["emp_id"], state["overtime_period"])
                state["overtime_summary"] = overtime_summary
                state["result_message"] = "🧠 Weekly overtime calculation completed"
            
            logger.info(f"Action executed for {state['emp_id']}: {state['action_taken']}")
            
        except Exception as e:
            logger.error(f"Action execution error: {str(e)}")
            state["error"] = f"Action execution failed: {str(e)}"
        
        return state
    
    def _calculate_overtime(self, weekly_records: List[dict], emp_id: str, period: str) -> str:
        """Calculate overtime using LLM logic"""
        try:
            if not weekly_records:
                return f"No attendance records found for {period} period"
            
            # Prepare data for LLM analysis
            records_text = "\n".join([
                f"Date: {record.get('date', 'N/A')}, Entry: {record.get('entry', 'N/A')}, Exit: {record.get('exit', 'N/A')}"
                for record in weekly_records
            ])
            
            prompt = f"""Analyze the following attendance records and calculate overtime hours for the {period} period.

Attendance Records:
{records_text}

Calculate:
1. Total working hours for each day
2. Total overtime hours (hours worked beyond 8 hours per day)
3. Summary of {period} overtime

Return a concise summary in this format:
"Total working days: X, Total hours: Y, Overtime hours: Z, Average daily overtime: W hours"

Focus on practical overtime calculation based on standard 8-hour workday."""

            response = groq_client.generate(prompt, max_tokens=150, temperature=0.1)
            
            if response and len(response.strip()) > 10:
                return response.strip()
            else:
                # Fallback calculation
                total_hours = 0
                overtime_hours = 0
                working_days = 0
                
                for record in weekly_records:
                    if record.get('entry') and record.get('exit'):
                        try:
                            entry_time = datetime.strptime(str(record['entry']), '%H:%M:%S').time()
                            exit_time = datetime.strptime(str(record['exit']), '%H:%M:%S').time()
            
                            # Calculate hours worked
                            entry_dt = datetime.combine(date.today(), entry_time)
                            exit_dt = datetime.combine(date.today(), exit_time)
                            
                            if exit_dt < entry_dt:
                                exit_dt += timedelta(days=1)
                            
                            hours_worked = (exit_dt - entry_dt).total_seconds() / 3600
                            total_hours += hours_worked
                            
                            if hours_worked > 8:
                                overtime_hours += hours_worked - 8
                            
                            working_days += 1
                            
                        except (ValueError, TypeError):
                            continue
                
                avg_overtime = overtime_hours / working_days if working_days > 0 else 0
                return f"Total working days: {working_days}, Total hours: {total_hours:.1f}, Overtime hours: {overtime_hours:.1f}, Average daily overtime: {avg_overtime:.1f} hours"
            
        except Exception as e:
            logger.error(f"Overtime calculation error: {str(e)}")
            return f"Error calculating overtime: {str(e)}"
    
    def finalize_result(self, state: AttendanceState) -> AttendanceState:
        """Finalize the result with proper formatting"""
        try:
            if state.get("error"):
                state["result_message"] = f"❌ {state['error']}"
                return state
            
            # Add email status only for overtime calculations
            action_taken = state.get("action_taken", "")
            if action_taken == "CALCULATE_OVERTIME":
                if state.get("email_sent", False):
                    state["result_message"] += "\n📧 Overtime report sent to your email"
                else:
                    state["result_message"] += "\n⚠️ Email notification failed"
            
            logger.info(f"Attendance process completed for {state['emp_id']}")
            
        except Exception as e:
            logger.error(f"Result finalization error: {str(e)}")
            state["result_message"] = f"❌ Process completed with errors: {str(e)}"
        
        return state
    
    def process_attendance(self, emp_id: str, operation: str, overtime_query: str = "") -> Dict[str, Any]:
        """Simple attendance processing method (assumes person is always in office)"""
        try:
            # Get employee info
            employee = postgres_client.get_employee(emp_id)
            if not employee:
                return {
                    "success": False,
                    "message": "❌ Employee not found",
                    "action": "ERROR"
                }
            
            emp_name = employee.get('name', f'Employee {emp_id}')
            
            # Process based on operation
            if operation == "checkin":
                # First check if already checked in
                attendance_today = postgres_client.check_attendance_today(emp_id)
                if attendance_today and attendance_today.get('entry'):
                    return {
                        "success": True,
                        "message": f"ℹ️ {emp_name} is already checked in today at {attendance_today.get('entry')}",
                        "action": "ALREADY_CHECKED_IN"
                    }
                
                result = postgres_client.record_attendance(emp_id, "check_in")
                if result == "checked in":
                    return {
                        "success": True,
                        "message": f"✅ {emp_name} successfully checked in at {datetime.now().strftime('%H:%M')}",
                        "action": "CHECKIN_SUCCESS"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"❌ Check-in failed: {result}",
                        "action": "CHECKIN_ERROR"
                    }
                    
            elif operation == "checkout":
                # First check current status
                attendance_today = postgres_client.check_attendance_today(emp_id)
                if not attendance_today or not attendance_today.get('entry'):
                    return {
                        "success": False,
                        "message": f"❌ Cannot check out - {emp_name} has not checked in today",
                        "action": "NOT_CHECKED_IN"
                    }
                if attendance_today.get('exit'):
                    return {
                        "success": True,
                        "message": f"ℹ️ {emp_name} is already checked out today at {attendance_today.get('exit')}",
                        "action": "ALREADY_CHECKED_OUT"
                    }
                
                result = postgres_client.record_attendance(emp_id, "check_out")
                if result == "checked out":
                    return {
                        "success": True,
                        "message": f"✅ {emp_name} successfully checked out at {datetime.now().strftime('%H:%M')}",
                        "action": "CHECKOUT_SUCCESS"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"❌ Check-out failed: {result}",
                        "action": "CHECKOUT_ERROR"
                    }
                    
            elif operation == "overtime":
                period = "weekly" if "weekly" in overtime_query else "monthly"
                overtime_info = postgres_client.get_attendance_summary(emp_id, period)
                return {
                    "success": True,
                    "message": f"📊 {emp_name}'s {period} overtime summary: {overtime_info}",
                    "action": "OVERTIME_CALCULATED",
                    "overtime_summary": overtime_info
                }
                
            elif operation == "status":
                # Get today's attendance status
                attendance_today = postgres_client.check_attendance_today(emp_id)
                
                if attendance_today:
                    check_in = attendance_today.get('check_in_time', 'Not checked in')
                    check_out = attendance_today.get('check_out_time', 'Not checked out')
                    
                    return {
                        "success": True,
                        "message": f"📅 {emp_name}'s attendance today:\n• Check-in: {check_in}\n• Check-out: {check_out}",
                        "action": "STATUS_RETRIEVED"
                    }
                else:
                    return {
                        "success": True,
                        "message": f"📅 {emp_name} has no attendance record for today",
                        "action": "NO_RECORDS"
                    }
            
            else:
                return {
                    "success": False,
                    "message": f"❌ Unknown operation: {operation}",
                    "action": "UNKNOWN_OPERATION"
                }
                
        except Exception as e:
            logger.error(f"Attendance processing error: {str(e)}")
            return {
                "success": False,
                "message": f"❌ System error: {str(e)}",
                "action": "ERROR"
            }

# Global agent instance
attendance_agent = AttendanceAgent()

def process_attendance(state: AttendanceState) -> AttendanceState:
    """Process attendance check-in, check-out"""
    emp_id = state["emp_id"]
    action = state["action"].lower()
    
    try:
        # Check if employee exists
        employee = postgres_client.get_employee(emp_id)
        if not employee:
            return {
                **state,
                "status": "error",
                "message": "Employee not found"
            }
        
        # Process attendance based on action
        if action in ["check_in", "check_out"]:
            result = postgres_client.record_attendance(emp_id, action)
            
            if result == "checked_in":
                return {
                    **state,
                    "status": "success",
                    "message": "checked in"
                }
            elif result == "checked_out":
                return {
                    **state,
                    "status": "success", 
                    "message": "checked out"
                }
            elif result == "already_checked_in":
                return {
                    **state,
                    "status": "error",
                    "message": "Already checked in today"
                }
            elif result == "already_checked_out":
                return {
                    **state,
                    "status": "error",
                    "message": "Already checked out today"
                }
            elif result == "not_checked_in":
                return {
                    **state,
                    "status": "error",
                    "message": "Cannot check out without checking in first"
                }
            else:
                return {
                    **state,
                    "status": "error",
                    "message": f"Attendance error: {result}"
                }
        else:
            return {
                **state,
            "status": "error", 
                "message": f"Invalid action: {action}"
        }
        
    except Exception as e:
        logger.error(f"Attendance processing error: {str(e)}")
        return {
            **state,
            "status": "error",
            "message": f"System error: {str(e)}"
        }

def format_response(state: AttendanceState) -> AttendanceState:
    """Format the final response"""
    try:
        if state.get("status") == "success":
            state["message"] = f"✅ {state['message']}"
        elif state.get("status") == "error":
            state["message"] = f"❌ {state['message']}"
        
        return state
    except Exception as e:
        logger.error(f"Response formatting error: {str(e)}")
        state["message"] = f"❌ Formatting error: {str(e)}"
    return state

# Create the workflow
workflow = StateGraph(AttendanceState)
workflow.add_node("process_attendance", process_attendance)
workflow.add_node("format_response", format_response)

workflow.set_entry_point("process_attendance")
workflow.add_edge("process_attendance", "format_response")
workflow.add_edge("format_response", END)

attendance_agent = workflow.compile()

def attendance_agent_main(emp_id: str, action: str) -> str:
    """Main entry point for attendance agent (assumes person is always in office)"""
    try:
        # Convert action to operation
        operation_map = {
            "check_in": "checkin",
            "check_out": "checkout",
            "weekly_overtime": "overtime",
            "monthly_overtime": "overtime"
        }
        
        operation = operation_map.get(action)
        if not operation:
            return "Invalid action"
            
        # Initialize state
        state = {
            "emp_id": emp_id,
            "operation": operation,
            "current_time": "",
            "current_date": "",
            "employee_info": None,
            "attendance_record": None,
            "can_proceed": True,
            "action_taken": "",
            "weekly_records": [],
            "overtime_summary": "",
            "overtime_query": "",
            "overtime_period": "",
            "result_message": "",
            "email_sent": False,
            "error": None
        }
        
        # Process attendance
        if operation == "checkin":
            success = postgres_client.record_checkin(emp_id, datetime.now().strftime("%H:%M:%S"))
            return "checked in" if success else "error checking in"
            
        elif operation == "checkout":
            success = postgres_client.record_checkout(emp_id, datetime.now().strftime("%H:%M:%S"))
            return "checked out" if success else "error checking out"
            
        else:
            return "Invalid operation"
        
    except Exception as e:
        logger.error(f"Attendance agent error: {str(e)}")
        return str(e) 