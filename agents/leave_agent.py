# -*- coding: utf-8 -*-
"""
Leave Management Agent - AI-Powered Leave Request Processing
Updated for new database structure with Indian date format
"""
import re
import json
import time
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List, TypedDict
from loguru import logger

from utils.groq_client import groq_client
from utils.postgres_client import postgres_client
from utils.email_client import email_client
from config import settings
from langgraph.graph import StateGraph, END

class LeaveAgent:
    def __init__(self):
        self.max_retries = 3
        self.cache = {}
        
    def format_indian_date(self, date_obj):
        """Format date in Indian format (dd-mm-yyyy)"""
        if isinstance(date_obj, str):
            # Convert from YYYY-MM-DD to dd-mm-yyyy
            try:
                dt = datetime.strptime(date_obj, "%Y-%m-%d")
                return dt.strftime("%d-%m-%Y")
            except:
                return date_obj
        elif isinstance(date_obj, (datetime, date)):
            return date_obj.strftime("%d-%m-%Y")
        return str(date_obj)
    
    def parse_indian_date(self, date_str):
        """Parse Indian format date (dd-mm-yyyy) to datetime object"""
        try:
            if " to " in date_str:
                # Handle date range format "dd-mm-yyyy to dd-mm-yyyy"
                start_str, end_str = date_str.split(" to ")
                start_date = datetime.strptime(start_str.strip(), "%d-%m-%Y")
                end_date = datetime.strptime(end_str.strip(), "%d-%m-%Y")
                return start_date, end_date
            else:
                return datetime.strptime(date_str, "%d-%m-%Y")
        except:
            return None
    
    def calculate_working_days(self, start_date: datetime, end_date: datetime) -> int:
        """
        Calculate working days between two dates, excluding:
        - All Sundays (always non-working)
        - Odd Saturdays (1st, 3rd, 5th Saturdays of the month are holidays)
        - Even Saturdays (2nd, 4th Saturdays of the month are working days)
        """
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        working_days = 0
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() == 6:  # Sunday
                current_date += timedelta(days=1)
                continue
            if current_date.weekday() == 5:  # Saturday
                first_day_of_month = current_date.replace(day=1)
                first_saturday = first_day_of_month
                while first_saturday.weekday() != 5:
                    first_saturday += timedelta(days=1)
                days_diff = (current_date - first_saturday).days
                saturday_number = (days_diff // 7) + 1
                if saturday_number % 2 == 1:  # Odd Saturday - holiday
                    current_date += timedelta(days=1)
                    continue
                else:  # Even Saturday - working day
                    working_days += 1
            else:
                working_days += 1
            current_date += timedelta(days=1)
        return working_days

    def is_working_day(self, date: datetime) -> bool:
        """
        Check if a given date is a working day (Mon-Fri and odd Saturdays)
        """
        if date.weekday() == 6:  # Sunday
            return False
        if date.weekday() < 5:  # Monday to Friday
            return True
        if date.weekday() == 5:  # Saturday
            first_day_of_month = date.replace(day=1)
            first_saturday = first_day_of_month
            while first_saturday.weekday() != 5:
                first_saturday += timedelta(days=1)
            days_diff = (date - first_saturday).days
            saturday_number = (days_diff // 7) + 1
            return saturday_number % 2 == 1  # Odd Saturday = working day
        return False
        
    def parse_leave_request(self, raw_input: str, emp_id: str) -> Dict[str, Any]:
        """
        Parse leave request from natural language input using LLM
        """
        try:
            from datetime import datetime
            today_str = datetime.now().strftime("%Y-%m-%d")
            year_str = datetime.now().strftime("%Y")
            
            prompt = f"""TODAY'S DATE: {today_str}
CURRENT YEAR: {year_str}

You are an expert HR assistant. Extract the following fields from the leave request below and return as a JSON object:
- emp_id (string, required)
- leave_type (sick, casual, annual, maternity, paternity, required; default to 'casual' if not specified, unless the user mentions another type)
- start_date (YYYY-MM-DD format, required)
- end_date (YYYY-MM-DD format, required)
- reason (string, required; use the user's original message if not specified)
- days_requested (integer, calculated from start_date to end_date, required)

IMPORTANT:
- days_requested must be calculated as working days only, using the following rules:
  - All Sundays are holidays (non-working)
  - Odd Saturdays (1st, 3rd, 5th Saturdays of the month) are working days
  - Even Saturdays (2nd, 4th Saturdays of the month) are holidays (non-working)
  - Monday to Friday are always working days
  - You must use the correct month and year for the date range, and use calendar logic to determine which Saturdays are odd/even
  - Do not hardcode the number of working days for any month; always calculate using the above rules
  - You MUST calculate working days for ANY date range, including multi-month ranges
  - Use the current year ({year_str}) for all calculations unless the date range is in the past
- If the user says 'tomorrow', always resolve to the next calendar day from today (use the current system date, not a hardcoded year)
- If the user says 'today', use the current system date
- If the user says 'whole of [month] and [month]', use the 1st of the first month to the last day of the second month as the date range, and always calculate working days for the full range
- If the user says 'whole of [month]', use the 1st to the last day of that month
- If the user says 'next Monday', resolve to the next Monday from today (using the current system date)
- If the user says 'first week of [month]', use the 1st to 7th of that month
- Use the current year unless the date range is in the past, then use the next year
- You MUST always include the field 'days_requested' as an integer in your JSON output. If you cannot calculate it, return an error field instead.
- All output fields must be present and correct for the current year.
- If you cannot determine a field, make your best guess or use a sensible default.

EXACT WORKING DAYS EXAMPLES for {year_str}:
- "25th July to 30th July" = 4 working days (Fri, Mon, Tue, Wed - Sat/Sun are weekends)
- "25th July to 30th August" = 29 working days (including odd Saturdays as working days)
- "1st August to 31st August" = 24 working days
- "1st July to 31st July" = 25 working days
- "1st July to 31st August" = 49 working days

CRITICAL: You MUST calculate working days for any date range provided. Do not return null for days_requested.

Message: "{raw_input}"
Employee ID: {emp_id}

If any field is missing or ambiguous, make your best guess and add a note in a 'notes' field.
If the message is not a leave request, return an error field with a message.

Return ONLY a valid JSON object, with NO extra text, NO markdown, and NO explanation.
Do NOT include any introductory or trailing text."""

            response = groq_client.generate(prompt, max_tokens=200, temperature=0.1)
            logger.info(f"Raw LLM response for leave request: {response}")
            
            # Handle markdown-wrapped JSON from Gemma
            if response and response.strip().startswith('```json'):
                # Extract JSON from markdown code block
                lines = response.strip().split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip() == '```json':
                        in_json = True
                        continue
                    elif line.strip() == '```':
                        break
                    elif in_json:
                        json_lines.append(line)
                response = '\n'.join(json_lines)
            
            # Clean up any remaining markdown or extra text
            response = response.strip()
            if response.startswith('```') and response.endswith('```'):
                response = response[3:-3].strip()
            
            if not response:
                return {
                    "error": "LLM did not return a response",
                    "message": "Could not parse leave request. Please try again."
                }
            
            data = json.loads(response)
            
            # Check if we got a valid leave request data structure
            required_fields = ['emp_id', 'start_date', 'end_date', 'reason', 'days_requested', 'leave_type']
            if not isinstance(data, dict) or not all(field in data for field in required_fields):
                return {
                    "error": "LLM did not return all required fields",
                    "message": "Could not parse leave request. Please try again.",
                    "llm_response": response
                }
            return data
        except Exception as e:
            logger.error(f"Leave request parsing failed: {e}")
            return {
                "error": "LLM parsing failed",
                "message": "Could not parse leave request. Please try again."
            }
    
    def _fallback_parse_leave_request(self, raw_input: str, emp_id: str) -> Dict[str, Any]:
        """Fallback parsing for leave requests when LLM parsing fails"""
        try:
            raw_lower = raw_input.lower()
            leave_type = "casual"
            if "sick" in raw_lower:
                leave_type = "sick"
            elif "annual" in raw_lower or "vacation" in raw_lower:
                leave_type = "annual"
            elif "maternity" in raw_lower:
                leave_type = "maternity"
            elif "paternity" in raw_lower:
                leave_type = "paternity"

            import re
            from datetime import datetime, timedelta

            # Month mapping
            month_map = {
                'january': 1, 'jan': 1,
                'february': 2, 'feb': 2,
                'march': 3, 'mar': 3,
                'april': 4, 'apr': 4,
                'may': 5,
                'june': 6, 'jun': 6,
                'july': 7, 'jul': 7,
                'august': 8, 'aug': 8,
                'september': 9, 'sep': 9,
                'october': 10, 'oct': 10,
                'november': 11, 'nov': 11,
                'december': 12, 'dec': 12
            }

            start_date = None
            end_date = None

            # Handle 'whole of [month]' and 'whole of [month] and [month]'
            whole_months_pattern = r'whole of ([a-z]+)(?: and ([a-z]+))?'
            match = re.search(whole_months_pattern, raw_lower)
            if match:
                month1 = match.group(1)
                month2 = match.group(2)
                year = datetime.now().year
                m1 = month_map.get(month1, None)
                m2 = month_map.get(month2, None)
                if m1:
                    start_date = datetime(year, m1, 1)
                    if m2:
                        # End of second month
                        from calendar import monthrange
                        last_day = monthrange(year, m2)[1]
                        end_date = datetime(year, m2, last_day)
                    else:
                        # End of first month
                        from calendar import monthrange
                        last_day = monthrange(year, m1)[1]
                        end_date = datetime(year, m1, last_day)
            else:
                # Existing natural language and date pattern logic
                date_patterns = [
                    r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:to|-)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
                    r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(?:to|-)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
                    r'\d{4}-\d{2}-\d{2}',
                    r'\d{2}-\d{2}-\d{4}',
                    r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}',
                ]
                natural_pattern = date_patterns[0]
                match = re.search(natural_pattern, raw_lower)
                if match:
                    start_day = int(match.group(1))
                    end_day = int(match.group(2))
                    month_name = match.group(3)
                    month = month_map.get(month_name.lower(), 1)
                    year = datetime.now().year
                    if month < datetime.now().month:
                        year += 1
                    start_date = datetime(year, month, start_day)
                    end_date = datetime(year, month, end_day)
                    if start_date > end_date:
                        start_date, end_date = end_date, start_date
                else:
                    dates_found = []
                    for pattern in date_patterns[2:]:
                        dates_found.extend(re.findall(pattern, raw_input))
                    if not dates_found:
                        start_date = datetime.now() + timedelta(days=1)
                        end_date = start_date
                    else:
                        try:
                            date_str = dates_found[0]
                            if '-' in date_str and len(date_str.split('-')[0]) == 4:
                                start_date = datetime.strptime(date_str, "%Y-%m-%d")
                            else:
                                start_date = datetime.strptime(date_str, "%d-%m-%Y")
                            if len(dates_found) > 1:
                                date_str2 = dates_found[1]
                                if '-' in date_str2 and len(date_str2.split('-')[0]) == 4:
                                    end_date = datetime.strptime(date_str2, "%Y-%m-%d")
                                else:
                                    end_date = datetime.strptime(date_str2, "%d-%m-%Y")
                            else:
                                end_date = start_date
                        except:
                            start_date = datetime.now() + timedelta(days=1)
                            end_date = start_date

            # Use unified working days calculation (odd Saturdays are holidays, even are working)
            days_requested = postgres_client.calculate_working_days(start_date, end_date) if start_date and end_date else 0

            return {
                "emp_id": emp_id,
                "leave_type": leave_type,
                "start_date": start_date.strftime("%Y-%m-%d") if start_date else None,
                "end_date": end_date.strftime("%Y-%m-%d") if end_date else None,
                "reason": raw_input,
                "days_requested": days_requested,
                "notes": "Parsed using fallback method due to LLM parsing failure"
            }
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
            return {
                "error": "Could not parse leave request",
                "message": "Please provide leave request in format: 'I need leave from YYYY-MM-DD to YYYY-MM-DD for [reason]'"
            }
    
    def get_employee_data(self, emp_id: str) -> Dict[str, Any]:
        """Get employee data for leave processing from correct database structure"""
        try:
            with postgres_client._get_cursor() as cur:
                # Get employee info
                cur.execute("SELECT * FROM employees WHERE id = %s", (emp_id,))
                employee = cur.fetchone()
                
                if not employee:
                    return {
                        "success": False,
                        "error": "Employee not found"
                    }
                
                # Get leave balance data from leaves table using employee name
                employee_name = employee['name']
                cur.execute("SELECT * FROM leaves WHERE name = %s", (employee_name,))
                leave_data = cur.fetchone()
                
                if not leave_data:
                    return {
                        "success": False,
                        "error": "Leave balance not found"
                    }
                
                # Build response with correct structure
                employee_data = {
                    "id": employee['id'],
                    "name": employee['name'],
                    "email": employee['email'],
                    "dept_head": employee['dept_head'],
                    "head_email": employee['head_email'],
                    "leaves_left": leave_data['leaves_left'],
                    "last_leave": leave_data.get('last_leave', 'N/A'),
                    "last_ai_approved": leave_data.get('last_ai_approved', 'N/A'),
                    "pending_approval_hr": leave_data.get('pending_approval_hr', 'N/A')
                }
                
                return {
                    "success": True,
                    "data": employee_data
                }
                
        except Exception as e:
            logger.error(f"Error getting employee data: {str(e)}")
            return {
                "success": False,
                "error": f"Database error: {str(e)}"
            }
    
    def make_leave_decision(self, request_data: Dict[str, Any], employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make leave decision based on company policies - never deny, only auto-approve or escalate"""
        try:
            # Extract employee info from the data structure
            if "data" in employee_data:
                emp_info = employee_data["data"]
            else:
                emp_info = employee_data
            
            current_leaves = emp_info.get("leaves_left", 0)
            last_leave_str = emp_info.get("last_leave", "")
            requested_days = request_data.get("days_requested", 1)
            leave_type = request_data.get("leave_type", "vacation").lower()
            
            # Policy: Auto-approve for 2 days or less with sufficient balance, escalate everything else
            
            # Auto-approval rules for short leaves with sufficient balance
            if requested_days <= 2 and current_leaves >= requested_days:
                if leave_type == "sick":
                    return {
                        "decision": "AUTO_APPROVED",
                        "message": f"✅ Sick leave auto-approved ({requested_days} day(s)). Remaining balance: {current_leaves - requested_days} days",
                        "can_proceed": True,
                        "reason": "sick_leave_auto"
                    }
                else:
                    return {
                        "decision": "AUTO_APPROVED",
                        "message": f"✅ Leave auto-approved ({requested_days} day(s)). Remaining balance: {current_leaves - requested_days} days",
                        "can_proceed": True,
                        "reason": "short_leave_auto"
                    }
            
            # All other cases escalate to department head (never deny)
            escalation_reasons = []
            
            if requested_days > 2:
                escalation_reasons.append(f"Extended leave ({requested_days} days)")
            
            if current_leaves < requested_days:
                escalation_reasons.append(f"Insufficient balance (Available: {current_leaves}, Requested: {requested_days})")
            
            # Check gap from last leave (informational only)
            if last_leave_str:
                try:
                    if " to " in last_leave_str:
                        _, last_end_date = self.parse_indian_date(last_leave_str)
                        start_date = datetime.strptime(request_data.get("start_date", "1970-01-01"), "%Y-%m-%d")
                        days_since_last = (start_date - last_end_date).days
                        
                        if days_since_last < 7 and leave_type not in ["sick", "emergency"]:
                            escalation_reasons.append(f"Short gap from last leave ({days_since_last} days)")
                except (ValueError, TypeError):
                    pass
            
            reason_text = " | ".join(escalation_reasons) if escalation_reasons else "Requires manager approval"
            return {
                    "decision": "ESCALATE_TO_DEPT_HEAD",
                "message": f"📋 Leave request escalated to department head for approval. Reason: {reason_text}",
                    "can_proceed": True,
                "reason": "manager_approval_required",
                "escalation_details": escalation_reasons
                }
                
        except Exception as e:
            logger.error(f"Leave decision making error: {str(e)}")
            return {
                "decision": "ERROR",
                "message": f"Error processing request: {str(e)}",
                "can_proceed": False,
                "reason": "processing_error"
            }
    
    def generate_email_content(self, request: Dict[str, Any], employee: Dict[str, Any], decision: str) -> Dict[str, str]:
        """Generate concise, direct email content using LLM for all leave requests"""
        try:
            emp_id = request["emp_id"]
            start = self.format_indian_date(request["start_date"])
            end = self.format_indian_date(request["end_date"])
            days = request["days_requested"]
            reason = request["reason"]
            leave_type = request.get("leave_type", "vacation")
            if "data" in employee:
                emp_data = employee["data"]
            else:
                emp_data = employee
            emp_name = emp_data.get("name", f"Employee {emp_id}")
            dept_head = emp_data.get("dept_head", "Department Head")
            head_email = emp_data.get("head_email", "hr@company.com")
            
            # Generate different email content based on decision
            if decision == "AUTO_APPROVED":
                # For auto-approved leaves, send notification email
                subject = f"Leave Auto-Approved: {start} to {end} ({days} working day{'s' if days != 1 else ''})"
                body = (
                    f"Dear {dept_head},\n"
                    f"This is to inform you that {emp_name} (Employee ID: {emp_id}) has been granted {days} working day{'s' if days != 1 else ''} "
                    f"of {leave_type} leave from {start} to {end}.\n"
                    f"{'Reason: ' + reason + '\n' if reason and reason != 'not specified' else ''}"
                    f"This leave has been auto-approved as per company policy.\n"
                    f"Best regards,\n"
                    f"HR System"
                )
            else:
                # For escalated leaves, send request email
                subject = f"Leave Application for {start} to {end} ({days} working day{'s' if days != 1 else ''})"
                body = (
                    f"Dear {dept_head},\n"
                    f"I am writing to request {days} working day{'s' if days != 1 else ''} of {leave_type} leave from {start} to {end}. "
                    f"{'Reason: ' + reason + '. ' if reason and reason != 'not specified' else ''}"
                    f"I would appreciate it if you could review and approve my request.\n"
                    f"Thank you,\n"
                    f"{emp_name}"
                )
            
            return {
                "subject": subject,
                "content": body,
                "recipient": head_email,
                "recipient_name": dept_head
            }
        except Exception as e:
            logger.error(f"Email generation failed: {str(e)}")
            return {
                "subject": "Leave Request Processing Error",
                "content": f"Error generating email content: {str(e)}",
                "recipient": "hr@company.com",
                "recipient_name": "HR Team"
            }
    
    def update_leave_balance(self, emp_id: str, request_data: Dict[str, Any]) -> bool:
        """Update leave balance in correct database structure"""
        try:
            with postgres_client._get_cursor() as cur:
                # Get employee name
                cur.execute("SELECT name FROM employees WHERE id = %s", (emp_id,))
                emp_result = cur.fetchone()
                if not emp_result:
                    return False
                
                emp_name = emp_result['name']
                days_used = request_data.get("days_requested", 0)
                
                # Create date range in Indian format
                start_date = self.format_indian_date(request_data.get("start_date"))
                end_date = self.format_indian_date(request_data.get("end_date"))
                date_range = f"{start_date} to {end_date}"
                
                # Update leaves table - deduct from leaves_left and update last_leave
                cur.execute("""
                    UPDATE leaves 
                    SET leaves_left = GREATEST(0, leaves_left - %s),
                        last_leave = %s
                    WHERE name = %s
                """, (days_used, date_range, emp_name))
                
                logger.info(f"Updated leave balance for {emp_name}: -{days_used} days")
                return True
                
        except Exception as e:
            logger.error(f"Leave balance update failed: {str(e)}")
            return False

    def process_leave_request(self, raw_input: str, emp_id: str, preview_only: bool = False) -> Dict[str, Any]:
        """
        Main processing function for leave requests and balance queries
        """
        try:
            # Check if this is a balance query first (more specific detection)
            balance_keywords = ['check my leave balance', 'leave balance', 'how many leaves', 'leaves do i have', 'remaining leaves', 'available leaves']
            is_balance_query = any(keyword in raw_input.lower() for keyword in balance_keywords)
            
            # Additional check: if it contains "apply for leave" or date patterns, it's likely a leave application
            leave_application_keywords = ['apply for leave', 'need leave', 'request leave', 'take leave', 'from', 'to']
            is_leave_application = any(keyword in raw_input.lower() for keyword in leave_application_keywords)
            
            # If it's both a balance query and contains leave application keywords, prioritize leave application
            if is_balance_query and not is_leave_application:
                # Handle balance query
                employee_data = self.get_employee_data(emp_id)
                if not employee_data or not employee_data.get("success", False) or not employee_data.get("data"):
                    return {
                        "success": False,
                        "message": f"❌ {employee_data.get('error', 'Employee data not found') if employee_data else 'Employee data not found'}",
                        "error": employee_data.get("error", "Employee data not found") if employee_data else "Employee data not found"
                    }
                emp_info = employee_data["data"]
                leaves_left = emp_info.get("leaves_left", 0)
                last_leave = emp_info.get("last_leave", "N/A")
                return {
                    "success": True,
                    "message": f"**Leave Balance for Employee {emp_id}:**\n\n📊 **Current Balance:** {leaves_left} days remaining\n📅 **Last Leave Taken:** {last_leave}\n\n*You can apply for leave using this system by specifying dates and reasons.*",
                    "data": {
                        "leaves_left": leaves_left,
                        "last_leave": last_leave,
                        "emp_id": emp_id
                    }
                }
            # Parse the leave request
            request_data = self.parse_leave_request(raw_input, emp_id)
            if not request_data or "error" in request_data or not isinstance(request_data, dict):
                return {
                    "success": False,
                    "message": f"❌ Could not parse as leave request. Try: 'I need 2 days leave from 2024-01-15 to 2024-01-16 for vacation'",
                    "error": request_data.get("error", "Invalid leave request format") if isinstance(request_data, dict) else "Invalid leave request format"
                }
            # Get employee data
            employee_data = self.get_employee_data(emp_id)
            if not employee_data or "error" in employee_data or not employee_data.get("success", False) or not employee_data.get("data"):
                return {
                    "success": False,
                    "message": f"❌ {employee_data.get('error', 'Employee data not found') if employee_data else 'Employee data not found'}",
                    "error": employee_data.get("error", "Employee data not found") if employee_data else "Employee data not found"
                }
            # Make leave decision
            decision = self.make_leave_decision(request_data, employee_data)
            if not decision or "decision" not in decision or "message" not in decision:
                return {
                    "success": False,
                    "message": "❌ Could not make a leave decision. Please try again later.",
                    "error": "Leave decision error"
                }
            # If auto-approved and not preview_only, update DB immediately
            if decision["decision"] == "AUTO_APPROVED" and not preview_only:
                self.update_leave_balance(emp_id, request_data)
            # Generate email content
            email_content = self.generate_email_content(request_data, employee_data, decision["decision"])
            if not email_content or not isinstance(email_content, dict) or "subject" not in email_content or "content" not in email_content:
                return {
                    "success": False,
                    "message": "❌ Could not generate email content for leave request.",
                    "error": "Email content error"
                }
            # Always include EMAIL DRAFT marker in the message
            formatted_message = f"**✅ Leave Request {decision['decision']}**\n\n**Leave Details:**\n• **From:** {self.format_indian_date(request_data['start_date'])}\n• **To:** {self.format_indian_date(request_data['end_date'])}\n• **Days:** {request_data['days_requested']} day(s)\n• **Reason:** {request_data['reason']}\n• **Status:** {decision['decision']}\n\n**📧 EMAIL DRAFT:**\n**Subject:** {email_content['subject']}\n**To:** {email_content['recipient']}\n\n{email_content['content']}"
            return {
                "success": True,
                "message": formatted_message,
                "subject": email_content["subject"],
                "content": email_content["content"],
                "recipient": email_content["recipient"],
                "decision": decision["decision"],
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing request: {str(e)}",
                "can_proceed": False,
                "reason": "processing_error"
            }

    def generate_leave_preview(self, request_data: Dict[str, Any], employee_data: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a robust preview for leave request with all required fields and fallbacks"""
        try:
            # Extract employee info from the data structure
            if "data" in employee_data:
                emp_info = employee_data["data"]
            else:
                emp_info = employee_data
            
            preview = {
                "emp_id": request_data.get("emp_id", ""),
                "leave_type": request_data.get("leave_type", "vacation"),
                "start_date": request_data.get("start_date", "1970-01-01"),
                "end_date": request_data.get("end_date", "1970-01-01"),
                "duration": request_data.get("days_requested", 1),
                "reason": request_data.get("reason", "Not specified"),
                "decision": decision.get("decision", "PENDING"),
                "can_proceed": decision.get("can_proceed", False),
                "message": decision.get("message", "No message"),
                "content": "",
                "subject": "",
                "recipient": emp_info.get("head_email", "manager@company.com")
            }
            
            # Generate email content
            email_content = self.generate_email_content(request_data, emp_info, decision.get("decision", "PENDING"))
            preview["content"] = email_content.get("content", "No content")
            preview["subject"] = email_content.get("subject", "Leave Request")
            
            logger.info(f"Successfully generated leave preview for {preview['emp_id']}")
            return preview
            
        except Exception as e:
            logger.error(f"Preview generation failed: {str(e)}")
            return {
                "emp_id": request_data.get("emp_id", ""), 
                "decision": "ERROR", 
                "message": str(e), 
                "can_proceed": False, 
                "content": "Error generating preview.",
                "subject": "Error",
                "recipient": "manager@company.com",
                "leave_type": "unknown",
                "start_date": "1970-01-01",
                "end_date": "1970-01-01",
                "duration": 0,
                "reason": "Error"
            }

    def send_leave_notification(self, emp_id: str, request_data: Dict[str, Any], employee_data: Dict[str, Any], decision: Dict[str, Any]) -> bool:
        """Send leave notification email"""
        try:
            # Extract employee info from the data structure
            emp_info = employee_data.get("data", {})
            
            subject = f"Leave Request - Employee {emp_id}"
            
            body = f"""
Leave Request Details:
- Employee ID: {emp_id}
- Department Head: {emp_info.get('dept_head', 'Not specified')}
- Leave Type: {request_data['leave_type']}
- Start Date: {request_data['start_date']}
- End Date: {request_data['end_date']}
- Duration: {request_data['days_requested']} days
- Reason: {request_data['reason']}

Decision: {decision['decision']}
Message: {decision['message']}
            """.strip()
            
            # Just log the email instead of sending (since you don't want actual emails)
            logger.info(f"EMAIL WOULD BE SENT - Subject: {subject}")
            logger.info(f"EMAIL WOULD BE SENT - Body: {body}")
            success = True
            
            return success
            
        except Exception as e:
            logger.error(f"Leave notification failed: {str(e)}")
            return False

    def __call__(self, raw_input: str, emp_id: str, preview_only: bool = False) -> Dict[str, Any]:
        return self.process_leave_request(raw_input, emp_id, preview_only=preview_only)

# Global instance
leave_agent = LeaveAgent()

def process_leave_request(raw_input: str, emp_id: str, preview_only: bool = False) -> Dict[str, Any]:
    """
    Process leave request with preview option
    
    Args:
        raw_input: Natural language leave request
        emp_id: Employee ID
        preview_only: If True, only return preview without sending email
        
    Returns:
        Result dictionary with message and status
    """
    return leave_agent.process_leave_request(raw_input, emp_id, preview_only)

def get_employee(emp_id: str) -> Optional[Dict[str, Any]]:
    """Get employee data from database"""
    try:
        return postgres_client.get_employee_info(emp_id)
    except Exception as e:
        logger.error(f"Failed to get employee {emp_id}: {str(e)}")
        return None 

# LangGraph states for leave workflow
class LeaveState(TypedDict):
    emp_id: str
    request: str
    feedbacks: List[str]
    subject: str
    content: str
    status: str
    auto_approved: bool
    leave_balance: int
    start_date: str
    end_date: str
    days_requested: int

# State functions

def analyze_leave_request(state: LeaveState) -> LeaveState:
    """Analyze leave request and determine if auto-approval is possible"""
    try:
        agent = LeaveAgent()
        
        # Parse the leave request
        request_data = agent.parse_leave_request(state["request"], state["emp_id"])
        
        if not request_data:
            state["status"] = "PARSING_ERROR"
            return state
        
        # Get employee data
        employee_data = agent.get_employee_data(state["emp_id"])
        
        if not employee_data.get("success", False) or not employee_data.get("data"):
            state["status"] = "EMPLOYEE_NOT_FOUND"
            return state
        
        # Update state
        state["start_date"] = request_data.get("start_date", "")
        state["end_date"] = request_data.get("end_date", "")
        state["days_requested"] = request_data.get("days_requested", 0)
        state["leave_balance"] = employee_data["data"].get("leaves_left", 0)
        
        # Check auto-approval logic: <= 1 day
        if state["days_requested"] <= 1 and state["leave_balance"] >= state["days_requested"]:
            state["auto_approved"] = True
            state["status"] = "AUTO_APPROVED"
        else:
            state["auto_approved"] = False
            state["status"] = "ESCALATE_TO_DEPT_HEAD"
        
        # Generate email content
        try:
            email_content = agent.generate_email_content(request_data, employee_data, state["status"])
            state["subject"] = email_content.get("subject", "Leave Request")
            state["content"] = email_content.get("content", "Leave request processed")
        except Exception as e:
            logger.error(f"Email generation error: {e}")
            state["subject"] = "Leave Request"
            state["content"] = "Leave request processed"
        
        return state
        
    except Exception as e:
        logger.error(f"Leave analysis error: {str(e)}")
        state["status"] = "ERROR"
        return state

def check_auto_approval(state: LeaveState) -> LeaveState:
    """Check if leave can be auto-approved - Fixed for <=1 day rule"""
    try:
        # Auto-approval for requests <= 1 day
        if state["days_requested"] <= 1 and state["leave_balance"] >= state["days_requested"]:
            state["auto_approved"] = True
            state["status"] = "AUTO_APPROVED"
        else:
            state["auto_approved"] = False
            state["status"] = "ESCALATE_TO_DEPT_HEAD"
            
        return state
        
    except Exception as e:
        logger.error(f"Auto approval check error: {str(e)}")
        state["status"] = "ERROR"
        return state

def create_leave_record(state: LeaveState) -> LeaveState:
    """Create leave record in database with new structure"""
    try:
        with postgres_client._get_cursor() as cur:
            # Get employee name
            cur.execute("SELECT name FROM employees WHERE id = %s", (state["emp_id"],))
            emp_result = cur.fetchone()
            
            if emp_result:
                emp_name = emp_result['name']
                
                # Create date range in Indian format
                start_date = state["start_date"]
                end_date = state["end_date"]
                
                # Format dates to Indian format
                agent = LeaveAgent()
                start_indian = agent.format_indian_date(start_date)
                end_indian = agent.format_indian_date(end_date)
                date_range = f"{start_indian} to {end_indian}"
                
                if state["auto_approved"]:
                    # Update last_ai_approved field
                    cur.execute("""
                        UPDATE leaves 
                        SET last_ai_approved = %s,
                            leaves_left = leaves_left - %s
                        WHERE name = %s
                    """, (date_range, state["days_requested"], emp_name))
                else:
                    # Update pending_approval_hr field
                    cur.execute("""
                        UPDATE leaves 
                        SET pending_approval_hr = %s
                        WHERE name = %s
                    """, (date_range, emp_name))
                
                logger.info(f"Updated leave record for {emp_name}")
                # Keep the original status (AUTO_APPROVED or ESCALATE_TO_DEPT_HEAD)
                if not state.get("status") or state["status"] == "PENDING":
                    state["status"] = "RECORD_CREATED"
        
        return state
        
    except Exception as e:
        logger.error(f"Leave record creation error: {str(e)}")
        state["status"] = "RECORD_ERROR"
        return state

def format_final_response(state: LeaveState) -> LeaveState:
    """Format final response with Indian date format"""
    try:
        agent = LeaveAgent()
        
        if state["status"] == "AUTO_APPROVED":
            response = f"""✅ Leave Request Auto-Approved!

📅 Duration: {agent.format_indian_date(state['start_date'])} to {agent.format_indian_date(state['end_date'])} ({state['days_requested']} day{'s' if state['days_requested'] > 1 else ''})
💼 Remaining Balance: {state['leave_balance'] - state['days_requested']} days

Your leave has been automatically approved. Department head has been notified.

📧 Email Details:
Subject: {state['subject']}"""
        
        elif state["status"] == "ESCALATE_TO_DEPT_HEAD":
            response = f"""📋 Leave Request Escalated to Department Head

📅 Duration: {agent.format_indian_date(state['start_date'])} to {agent.format_indian_date(state['end_date'])} ({state['days_requested']} day{'s' if state['days_requested'] > 1 else ''})
💼 Current Balance: {state['leave_balance']} days

Your leave request has been sent to your department head for approval.

📧 Email Details:
Subject: {state['subject']}"""
        
        else:
            response = f"""⚠️ Leave Request Status: {state['status']}

Please contact HR for assistance with your leave request."""
        
        state["final_response"] = response
        return state
        
    except Exception as e:
        logger.error(f"Response formatting error: {str(e)}")
        state["final_response"] = f"Error processing leave request: {str(e)}"
        return state

def leave_agent_main(emp_id: str, request: str, feedbacks: List[str] = None) -> Dict[str, str]:
    """Main leave agent function - uses working LeaveAgent class directly"""
    try:
        # Use the working LeaveAgent class directly
        leave_agent = LeaveAgent()
        
        # Process leave request (preview mode gives us proper structure)
        result = leave_agent.process_leave_request(request, emp_id, preview_only=True)
        
        if result.get("success"):
            preview = result.get("preview", {})
            return {
                "response": f"Leave request processed: {preview.get('decision', 'PROCESSED')}",
                "status": "SUCCESS",
                "subject": preview.get("subject", "Leave Request"),
                "content": preview.get("content", "Leave request has been processed")
            }
        else:
            return {
                "response": result.get("message", "Leave request could not be processed"),
                "status": "ERROR",
                "subject": "Leave Request",
                "content": result.get("message", "Error processing leave request")
            }
        
    except Exception as e:
        logger.error(f"Leave agent main error: {str(e)}")
        return {
            "response": f"Error processing leave request: {str(e)}",
            "status": "ERROR",
            "subject": "Leave Request Error",
            "content": f"An error occurred while processing the leave request: {str(e)}"
        }

# Main entry point
def process_leave_request(raw_input: str, emp_id: str, preview_only: bool = False) -> Dict[str, Any]:
    """Process leave request with new database structure"""
    return leave_agent_main(emp_id, raw_input) 

LeaveAgent = LeaveAgent 

# TESTS
if __name__ == "__main__":
    agent = LeaveAgent()
    emp_id = "20001"  # Use a valid emp_id from your DB
    from datetime import datetime
    # Test working day logic
    print("Testing working day logic:")
    test_dates = [
        ("2024-08-03", True),  # 1st Saturday (odd, working)
        ("2024-08-10", False), # 2nd Saturday (even, holiday)
        ("2024-08-17", True),  # 3rd Saturday (odd, working)
        ("2024-08-24", False), # 4th Saturday (even, holiday)
        ("2024-08-31", True),  # 5th Saturday (odd, working)
        ("2024-08-04", False), # Sunday (holiday)
        ("2024-08-05", True),  # Monday (working)
    ]
    for date_str, expected in test_dates:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        result = agent.is_working_day(d)
        print(f"{date_str}: {'Working' if result else 'Holiday'} (Expected: {'Working' if expected else 'Holiday'})")
    # Test leave calculation
    print("\nTest 1: 1-day leave on 1st Saturday (should auto-approve and update DB)")
    result1 = agent.process_leave_request("apply for leave on 2024-08-03", emp_id, preview_only=False)
    print(result1["message"])
    print("Test 2: 1-day leave on 2nd Saturday (should not count as working day, escalate)")
    result2 = agent.process_leave_request("apply for leave on 2024-08-10", emp_id, preview_only=False)
    print(result2["message"])
    print("Test 3: 3-day leave Mon-Wed (should escalate)")
    result3 = agent.process_leave_request("apply for leave from 2024-08-05 to 2024-08-07", emp_id, preview_only=False)
    print(result3["message"]) 