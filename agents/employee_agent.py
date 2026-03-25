# Employee Agent - Handle employee data queries, emails, and profile updates
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict
from loguru import logger
from config import settings, SUCCESS_MESSAGES, ERROR_MESSAGES
from utils.groq_client import groq_client
from utils.postgres_client import postgres_client
from utils.email_client import email_client

class EmployeeState(TypedDict):
    emp_id: str
    query: str
    current_employee: Optional[dict]
    target_employee: Optional[dict]
    action_type: str
    email_content: Optional[dict]
    profile_changes: Optional[dict]
    result_message: str
    success: bool
    error: Optional[str]

class EmployeeAgent:
    def __init__(self):
        self.max_retries = 3
        
    def get_employee_data(self, emp_id: str) -> Optional[Dict[str, Any]]:
        """Get employee data from database"""
        try:
            employee = postgres_client.get_employee(emp_id)
            if employee:
                return {
                    "id": emp_id,
                    "name": employee.get("name", ""),
                    "email": employee.get("email", ""),
                    "department": employee.get("department", ""),
                    "dept_head": employee.get("dept_head", ""),
                    "head_email": employee.get("head_email", ""),
                    "designation": employee.get("designation", ""),
                    "phone": employee.get("phone", ""),
                    "address": employee.get("address", ""),
                    "joining_date": employee.get("joining_date", ""),
                    "salary": employee.get("salary", ""),
                    "status": employee.get("status", "active")
                }
            return None
        except Exception as e:
            logger.error(f"Error getting employee data for {emp_id}: {e}")
            return None
    
    def parse_employee_query(self, query: str, current_emp_id: str) -> Dict[str, Any]:
        """Parse employee-related queries using LLM"""
        try:
            prompt = f"""You are an employee data assistant. Analyze the query and determine the action type and details.

Query: "{query}"
Current Employee ID: {current_emp_id}

Available actions:
1. GET_MY_INFO - Get current employee's information (e.g., "tell my name", "what is my email", "my dept head")
2. GET_EMP_INFO - Get another employee's information (e.g., "tell emp 20002 email", "emp 20003 info")
3. SEND_EMAIL - Send email to another employee (e.g., "write email to emp 20002 asking for project status", "email emp 20003 about updates")
4. UPDATE_PROFILE - Update current employee's profile (e.g., "change my name to tanay", "update my email to new@email.com")

CRITICAL INSTRUCTIONS:
- For SEND_EMAIL: Extract the target employee ID and the complete message content
- For UPDATE_PROFILE: Extract the field to update and the new value
- For GET_EMP_INFO: Only return name and email of other employees, not sensitive data
- For GET_MY_INFO: Return full employee information

Examples:
- "tell my name" → {{"action_type": "GET_MY_INFO"}}
- "what is my dept head email" → {{"action_type": "GET_MY_INFO"}}
- "tell emp 20002 email" → {{"action_type": "GET_EMP_INFO", "target_emp_id": "20002"}}
- "write email to emp 20002 asking for gen ai project status" → {{"action_type": "SEND_EMAIL", "target_emp_id": "20002", "message": "asking for gen ai project status"}}
- "email emp 20003 about project updates" → {{"action_type": "SEND_EMAIL", "target_emp_id": "20003", "message": "about project updates"}}
- "change my name to tanay" → {{"action_type": "UPDATE_PROFILE", "field": "name", "new_value": "tanay"}}
- "change my name in database to tanay" → {{"action_type": "UPDATE_PROFILE", "field": "name", "new_value": "tanay"}}

Return ONLY valid JSON with these fields:
- action_type: one of the above actions
- target_emp_id: employee ID if querying/sending to another employee
- field: field to update (for profile updates)
- new_value: new value (for profile updates)
- message: email message content (for email requests)

Return ONLY valid JSON, no other text."""

            response = groq_client.generate(prompt, max_tokens=150, temperature=0.1)
            logger.info(f"LLM Response for query '{query}': {response}")
            
            try:
                # Try to parse as JSON first
                import json
                data = json.loads(response.strip())
                return data
            except json.JSONDecodeError:
                # If JSON parsing fails, try eval as fallback
                try:
                    data = eval(response.strip())
                    return data
                except:
                    logger.error(f"Failed to parse LLM response: {response}")
                    # Return a default action
                    return {"action_type": "GET_MY_INFO"}
                    
        except Exception as e:
            logger.error(f"Error parsing employee query: {e}")
            return {"action_type": "GET_MY_INFO"}
    
    def get_my_info(self, emp_id: str) -> Dict[str, Any]:
        """Get current employee's information"""
        try:
            employee = self.get_employee_data(emp_id)
            if not employee:
                return {
                    "success": False,
                    "message": "❌ Employee not found",
                    "error": "Employee data not available"
                }
            
            return {
                "success": True,
                "message": f"**📋 Your Information:**\n\n• **Name:** {employee['name']}\n• **Email:** {employee['email']}\n• **Department:** {employee['department']}\n• **Designation:** {employee['designation']}\n• **Phone:** {employee['phone']}\n• **Department Head:** {employee['dept_head']}\n• **Head Email:** {employee['head_email']}",
                "data": employee
            }
        except Exception as e:
            logger.error(f"Error getting my info: {e}")
            return {
                "success": False,
                "message": f"❌ Error retrieving your information: {str(e)}",
                "error": str(e)
            }
    
    def get_emp_info(self, target_emp_id: str, current_emp_id: str) -> Dict[str, Any]:
        """Get another employee's information (limited access)"""
        try:
            target_employee = self.get_employee_data(target_emp_id)
            if not target_employee:
                return {
                    "success": False,
                    "message": f"❌ Employee {target_emp_id} not found",
                    "error": "Employee not found"
                }
            
            # Only show limited information for other employees
            return {
                "success": True,
                "message": f"**👤 Employee {target_emp_id} Information:**\n\n• **Name:** {target_employee['name']}\n• **Email:** {target_employee['email']}\n• **Department:** {target_employee['department']}\n• **Designation:** {target_employee['designation']}",
                "data": {
                    "name": target_employee['name'],
                    "email": target_employee['email'],
                    "department": target_employee['department'],
                    "designation": target_employee['designation']
                }
            }
        except Exception as e:
            logger.error(f"Error getting employee info: {e}")
            return {
                "success": False,
                "message": f"❌ Error retrieving employee information: {str(e)}",
                "error": str(e)
            }
    
    def send_employee_email(self, from_emp_id: str, to_emp_id: str, message: str) -> Dict[str, Any]:
        """Send email from one employee to another"""
        try:
            from_employee = self.get_employee_data(from_emp_id)
            to_employee = self.get_employee_data(to_emp_id)
            
            if not from_employee:
                return {
                    "success": False,
                    "message": "❌ Sender employee not found",
                    "error": "Sender not found"
                }
            
            if not to_employee:
                return {
                    "success": False,
                    "message": f"❌ Recipient employee {to_emp_id} not found",
                    "error": "Recipient not found"
                }
            
            # Generate email content
            subject = f"Message from {from_employee['name']} ({from_emp_id})"
            email_content = f"""Dear {to_employee['name']},

{message}

Best regards,
{from_employee['name']}
Employee ID: {from_emp_id}
Department: {from_employee['department']}"""
            
            # Format output similar to complaint agent
            formatted_message = f"""**📧 Email Generated Successfully!**

**To:** {to_employee['email']}
**Subject:** {subject}

**Message:**
{message}

*Note: Email preview generated. In production, this would be sent via email system.*"""
            
            return {
                "success": True,
                "message": formatted_message,
                "email_content": {
                    "to": to_employee['email'],
                    "subject": subject,
                    "content": email_content,
                    "from": from_employee['email']
                }
            }
        except Exception as e:
            logger.error(f"Error sending employee email: {e}")
            return {
                "success": False,
                "message": f"❌ Error sending email: {str(e)}",
                "error": str(e)
            }
    
    def update_employee_profile(self, emp_id: str, field: str, new_value: str) -> Dict[str, Any]:
        """Update current employee's profile and notify dept head"""
        try:
            # Get current employee data
            current_employee = self.get_employee_data(emp_id)
            if not current_employee:
                return {
                    "success": False,
                    "message": "❌ Employee not found",
                    "error": "Employee not found"
                }
            
            # Define allowed fields for update
            allowed_fields = {
                "name": "name",
                "email": "email", 
                "phone": "phone",
                "address": "address",
                "designation": "designation"
            }
            
            if field not in allowed_fields:
                return {
                    "success": False,
                    "message": f"❌ Field '{field}' cannot be updated. Allowed fields: {', '.join(allowed_fields.keys())}",
                    "error": "Invalid field"
                }
            
            # Update in database
            db_field = allowed_fields[field]
            with postgres_client._get_cursor() as cur:
                cur.execute(f"""
                    UPDATE employees 
                    SET {db_field} = %s 
                    WHERE id = %s
                """, (new_value, emp_id))
                
                if cur.rowcount == 0:
                    return {
                        "success": False,
                        "message": "❌ Failed to update profile",
                        "error": "Database update failed"
                    }
            
            # Generate notification email to dept head
            dept_head_email = current_employee['head_email']
            notification_subject = f"Profile Update Notification - Employee {emp_id}"
            notification_content = f"""Dear {current_employee['dept_head']},

This is to notify you that employee {emp_id} ({current_employee['name']}) has updated their profile information.

**Change Details:**
• Field Updated: {field.title()}
• New Value: {new_value}
• Updated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please review this change if necessary.

Best regards,
HR System"""
            
            return {
                "success": True,
                "message": f"**✅ Profile Updated Successfully!**\n\n**Field Updated:** {field.title()}\n**New Value:** {new_value}\n\n**📧 Notification sent to:** {current_employee['dept_head']} ({dept_head_email})\n\n*Your department head has been notified of this change.*",
                "notification_email": {
                    "to": dept_head_email,
                    "subject": notification_subject,
                    "content": notification_content
                }
            }
        except Exception as e:
            logger.error(f"Error updating employee profile: {e}")
            return {
                "success": False,
                "message": f"❌ Error updating profile: {str(e)}",
                "error": str(e)
            }
    
    def process_employee_request(self, query: str, emp_id: str) -> Dict[str, Any]:
        """Main processing function for employee requests"""
        try:
            # Parse the query
            parsed = self.parse_employee_query(query, emp_id)
            action_type = parsed.get("action_type", "GET_MY_INFO")
            
            if action_type == "GET_MY_INFO":
                return self.get_my_info(emp_id)
            
            elif action_type == "GET_EMP_INFO":
                target_emp_id = parsed.get("target_emp_id")
                if not target_emp_id:
                    return {
                        "success": False,
                        "message": "❌ Please specify which employee you want information about",
                        "error": "Missing target employee ID"
                    }
                return self.get_emp_info(target_emp_id, emp_id)
            
            elif action_type == "SEND_EMAIL":
                target_emp_id = parsed.get("target_emp_id")
                message = parsed.get("message", "No message provided")
                if not target_emp_id:
                    return {
                        "success": False,
                        "message": "❌ Please specify which employee to email",
                        "error": "Missing target employee ID"
                    }
                return self.send_employee_email(emp_id, target_emp_id, message)
            
            elif action_type == "UPDATE_PROFILE":
                field = parsed.get("field")
                new_value = parsed.get("new_value")
                if not field or not new_value:
                    return {
                        "success": False,
                        "message": "❌ Please specify what field to update and the new value",
                        "error": "Missing field or value"
                    }
                return self.update_employee_profile(emp_id, field, new_value)
            
            else:
                return {
                    "success": False,
                    "message": "❌ Unknown action type",
                    "error": "Invalid action"
                }
                
        except Exception as e:
            logger.error(f"Error processing employee request: {e}")
            return {
                "success": False,
                "message": f"❌ Error processing request: {str(e)}",
                "error": str(e)
            }

# Global agent instance
employee_agent = EmployeeAgent()

def process_employee_request(query: str, emp_id: str) -> Dict[str, Any]:
    """Main entry point for employee agent"""
    try:
        agent = EmployeeAgent()
        return agent.process_employee_request(query, emp_id)
    except Exception as e:
        logger.error(f"Employee agent error: {e}")
        return {
            "success": False,
            "message": f"❌ Employee agent error: {str(e)}",
            "error": str(e)
        }

def employee_agent_main(query: str, emp_id: str) -> str:
    """Main function for employee agent"""
    try:
        result = process_employee_request(query, emp_id)
        if result.get("success"):
            return result.get("message", "Request processed successfully")
        else:
            return result.get("message", "Request failed")
    except Exception as e:
        logger.error(f"Employee agent main error: {e}")
        return f"Error: {str(e)}" 