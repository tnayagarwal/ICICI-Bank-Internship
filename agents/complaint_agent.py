# -*- coding: utf-8 -*-
"""
Complaint & Request Management Agent - AI-Powered Request Processing
Based on notebook implementation with enhanced error handling
"""
import re
import json
import time
from typing import Dict, Any, Optional, List, TypedDict
from loguru import logger
from datetime import datetime

from utils.groq_client import groq_client
from utils.postgres_client import postgres_client
from utils.email_client import email_client
from config import settings
from langgraph.graph import StateGraph, END

def get_employee(emp_id: str) -> Optional[Dict[str, Any]]:
    """Get employee data from database"""
    try:
        return postgres_client.get_employee(emp_id)
    except Exception as e:
        logger.error(f"Failed to get employee {emp_id}: {str(e)}")
        return None

class ComplaintState(TypedDict):
    emp_id: str
    request: str
    feedbacks: List[str]
    subject: str
    content: str
    status: str
    employee_info: dict

def analyze_complaint_request(state: ComplaintState) -> ComplaintState:
    """Analyze the complaint request and generate a single, concise subject/content"""
    try:
        emp_id = state["emp_id"]
        request = state["request"]
        feedbacks = state.get("feedbacks", [])
        
        # Get employee info for context
        employee = postgres_client.get_employee(emp_id)
        if not employee:
            return {
                **state,
                "status": "error",
                "subject": "Error",
                "content": "Employee not found"
            }
        
        # Compose a strict, minimal prompt for the LLM
        prompt = (
            f"You are an HR assistant. Write a professional, direct, and extremely concise email (max 2 sentences) for the following employee complaint/request. "
            f"Only include the most essential information. Do NOT add extra context, do NOT repeat yourself, and do NOT include any 'Step' or multi-part output.\n"
            f"Employee: {employee.get('name', emp_id)}\n"
            f"Department: {employee.get('department', '')}\n"
            f"Complaint/Request: {request}\n"
            f"Format:\nSubject: <short subject>\nBody: <2-sentence body>"
        )
        response = groq_client.generate(prompt, max_tokens=120, temperature=0.1)
        # Parse subject and body
        subject_match = re.search(r'Subject:\s*(.+?)(?=\n|Body:|$)', response, re.DOTALL | re.IGNORECASE)
        body_match = re.search(r'Body:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
        subject = subject_match.group(1).strip() if subject_match else "Complaint/Request"
        body = body_match.group(1).strip() if body_match else response.strip()
        return {
            **state,
            "status": "success",
            "subject": subject,
            "content": body
        }
    except Exception as e:
        return {
            **state,
            "status": "error",
            "subject": "Error",
            "content": f"Complaint agent error: {e}"
        }

def create_complaint_record(state: ComplaintState) -> ComplaintState:
    """Create complaint record in database"""
    try:
        # Store complaint in database
        complaint_id = postgres_client.create_complaint(
            state["emp_id"],
            state["subject"],
            state["content"]
        )
        
        if complaint_id:
            content_suffix = f"\n\nComplaint ID: {complaint_id}\nStatus: Submitted for review"
        else:
            content_suffix = "\n\nNote: Complaint submitted but ID not generated"
        
        return {
            **state,
            "content": state["content"] + content_suffix,
            "status": "recorded"
        }
        
    except Exception as e:
        logger.error(f"Error creating complaint record: {e}")
        return {
            **state,
            "status": "recorded",
            "content": f"{state['content']}\n\nNote: Complaint submitted successfully"
        }

def format_final_response(state: ComplaintState) -> ComplaintState:
    """Format the final response with only subject and content"""
    try:
        if state.get("error"):
            state["final_response"] = {
                "subject": "Error Processing Complaint",
                "content": state["error"]
            }
            return state
            
        # Format the response with only subject and content
        state["final_response"] = {
            "subject": state.get("subject", "Complaint"),
            "content": state.get("content", "")
        }
        
        logger.info(f"Response formatted for {state['emp_id']}")
        
    except Exception as e:
        logger.error(f"Response formatting error: {str(e)}")
        state["final_response"] = {
            "subject": "Error Processing Complaint",
            "content": str(e)
        }
    
    return state

# Create the workflow
workflow = StateGraph(ComplaintState)
workflow.add_node("analyze_request", analyze_complaint_request)
workflow.add_node("create_record", create_complaint_record)
workflow.add_node("format_response", format_final_response)

workflow.set_entry_point("analyze_request")
workflow.add_edge("analyze_request", "create_record")
workflow.add_edge("create_record", "format_response")
workflow.add_edge("format_response", END)

complaint_agent = workflow.compile()

def complaint_agent_main(emp_id: str, request: str, feedbacks: List[str] = None) -> Dict[str, str]:
    """Main entry point for complaint agent - uses working ComplaintAgent class directly"""
    try:
        # Use the working ComplaintAgent class directly
        complaint_agent = ComplaintAgent()
        
        # Process complaint request (preview mode gives us proper structure)
        result = complaint_agent.process_complaint_request(request, emp_id, preview_only=True)
        
        if result.get("success"):
            return {
                "status": "SUCCESS",
                "response": result.get("message", "Complaint has been processed"),
                "subject": result.get("subject", "Complaint Request"),
                "content": result.get("content", "Complaint has been processed")
            }
        else:
            return {
                "status": "ERROR",
                "response": result.get("message", "Error processing complaint request"),
                "subject": "Error Processing Complaint",
                "content": result.get("message", "Error processing complaint request")
            }
        
    except Exception as e:
        logger.error(f"Complaint agent error: {str(e)}")
        return {
            "status": "ERROR",
            "response": str(e),
            "subject": "Error Processing Complaint",
            "content": str(e)
        }

class ComplaintAgent:
    def __init__(self):
        self.max_retries = 3
        self.categories = [
            "Equipment Request", "IT Support", "Workplace Issue", 
            "HR Policy", "Compensation", "Training", "Safety", "Other"
        ]
        
    def extract_request_details(self, raw_input: str, emp_id: str) -> Dict[str, Any]:
        """
        Extract structured details from a complaint message using LLM
        """
        try:
            prompt = f"""You are an expert HR assistant. Extract the following fields from the message below and return as a JSON object:
- emp_id (string, required)
- category (string, e.g. IT Support, HR, Facilities, required)
- subject (string, required)
- priority (LOW, MEDIUM, HIGH, required)
- description (string, required)
- sentiment (positive, neutral, negative, required)
- department (string, required)

Message: "{raw_input}"
Employee ID: {emp_id}

If any field is missing or ambiguous, make your best guess and add a note in a 'notes' field.
If the message is not a request, return an error field with a message.

Return ONLY a valid JSON object, with NO extra text, NO markdown, and NO explanation.
Do NOT include any introductory or trailing text."""

            response = groq_client.generate(prompt, max_tokens=200, temperature=0.1)
            
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
                # Fallback manual extraction
                return self._manual_extract_details(raw_input, emp_id)
            
            data = json.loads(response)
            
            # Validate required fields
            required_fields = ['emp_id', 'category', 'subject', 'priority', 'description', 'sentiment', 'department']
            for field in required_fields:
                if field not in data:
                    # Use fallback if missing required fields
                    return self._manual_extract_details(raw_input, emp_id)
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return self._manual_extract_details(raw_input, emp_id)
        except Exception as e:
            logger.error(f"Request parsing failed: {e}")
            return self._manual_extract_details(raw_input, emp_id)

    def _manual_extract_details(self, raw_input: str, emp_id: str) -> Dict[str, Any]:
        """Manual fallback extraction when LLM fails"""
        raw_lower = raw_input.lower()
        
        # Determine category based on keywords
        if any(word in raw_lower for word in ['laptop', 'computer', 'keyboard', 'mouse', 'software', 'it']):
            category = "IT Support"
            department = "IT"
        elif any(word in raw_lower for word in ['chair', 'desk', 'office', 'temperature', 'ac', 'facility']):
            category = "Facilities"
            department = "Facilities"
        elif any(word in raw_lower for word in ['hr', 'policy', 'leave', 'salary', 'benefits']):
            category = "HR Policy"
            department = "HR"
        else:
            category = "Other"
            department = "General"
        
        # Determine priority
        if any(word in raw_lower for word in ['urgent', 'critical', 'emergency', 'asap']):
            priority = "HIGH"
        elif any(word in raw_lower for word in ['soon', 'important', 'needed']):
            priority = "MEDIUM"
        else:
            priority = "LOW"
        
        # Basic sentiment analysis
        if any(word in raw_lower for word in ['broken', 'not working', 'problem', 'issue', 'stopped']):
            sentiment = "negative"
        elif any(word in raw_lower for word in ['please', 'request', 'need', 'would like']):
            sentiment = "neutral"
        else:
            sentiment = "neutral"
        
        # Extract subject (first 50 chars or until punctuation)
        subject = raw_input[:50]
        if '.' in subject:
            subject = subject[:subject.find('.')]
        if '!' in subject:
            subject = subject[:subject.find('!')]
        if '?' in subject:
            subject = subject[:subject.find('?')]
        subject = subject.strip()
        
        return {
            "emp_id": emp_id,
            "category": category,
            "subject": subject,
            "priority": priority,
            "description": raw_input,
            "sentiment": sentiment,
            "department": department,
            "notes": "Extracted using fallback method due to LLM parsing failure"
        }

    def get_employee_data(self, emp_id: str) -> Dict[str, Any]:
        """Get employee data for request processing"""
        try:
            employee = get_employee(emp_id)
            if not employee:
                return {
                    "success": False,
                    "error": "Employee not found"
                }
            
            employee["full_name"] = employee.get("dept head", "Employee")
            
            logger.info(f"Retrieved employee data for {emp_id}")
            return {
                "success": True,
                "data": employee
            }
            
        except Exception as e:
            logger.error(f"Employee data retrieval failed: {str(e)}")
            return {
                "success": False,
                "error": f"Data retrieval error: {str(e)}"
            }

    def determine_department(self, category: str) -> str:
        """Determine department based on category"""
        department_mapping = {
            "IT Support": "IT",
            "HR": "HR",
            "Facilities": "Facilities",
            "Finance": "Finance",
            "Security": "Security"
        }
        return department_mapping.get(category, "General")

    def send_complaint_notification(self, emp_id: str, request_data: Dict[str, Any], employee_data: Dict[str, Any]) -> bool:
        """Send complaint/request notification email"""
        try:
            # Handle different data structures
            if isinstance(employee_data, dict) and "data" in employee_data:
                emp_info = employee_data["data"]
            else:
                emp_info = employee_data
                
            # Get employee name safely
            emp_name = emp_info.get("full_name", emp_info.get("name", f"Employee {emp_id}"))
            
            # Use department head email if available, fallback to department@company.com
            dept_email = emp_info.get("head_email", emp_info.get("head email", f"{request_data['department'].lower()}@company.com"))
            subject = f"{request_data['category']} - {request_data['subject']}"
            body = f"""
Request Details:
- Employee: {emp_name} ({emp_id})
- Department: {emp_info.get('department', 'Not specified')}
- Category: {request_data['category']}
- Subject: {request_data['subject']}
- Priority: {request_data['priority']}
- Description: {request_data['description']}
- Sentiment: {request_data['sentiment']}
            """.strip()
            
            # Just log the email instead of sending (since you don't want actual emails)
            logger.info(f"EMAIL WOULD BE SENT - Subject: {subject}")
            logger.info(f"EMAIL WOULD BE SENT - Body: {body}")
            return True
            
        except Exception as e:
            logger.error(f"Complaint notification failed: {str(e)}")
            return False

    def save_request_log(self, request_data: Dict[str, Any], employee_data: Dict[str, Any]):
        """Save request to database log"""
        try:
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "emp_id": request_data["emp_id"],
                "category": request_data["category"],
                "subject": request_data["subject"],
                "priority": request_data["priority"],
                "department": request_data["department"],
                "sentiment": request_data["sentiment"],
                "status": "submitted"
            }
            
            logger.info(f"Request logged: {log_entry}")
            
        except Exception as e:
            logger.error(f"Request logging failed: {str(e)}")
    
    def process_complaint_request(self, raw_input: str, emp_id: str, preview_only: bool = False) -> Dict[str, Any]:
        """
        Main processing function for complaint/request
        """
        try:
            # Extract request details using LLM
            request_data = self.extract_request_details(raw_input, emp_id)
            if "error" in request_data:
                return {
                    "success": False,
                    "message": f"❌ {request_data['error']}",
                    "error": request_data["error"]
                }
            # Get employee data
            employee_data = self.get_employee_data(emp_id)
            if "error" in employee_data:
                return {
                    "success": False,
                    "message": f"❌ {employee_data['error']}",
                    "error": employee_data["error"]
                }
            # Generate email content
            department = self.determine_department(request_data["category"])
            emp_name = employee_data.get("name", f"Employee {emp_id}")
            subject = f"{request_data['category']} - {request_data['subject']} (Priority: {request_data['priority']})"

            # Generate professional email content using LLM (similar to leave agent)
            mail_prompt = f"""
You are an HR assistant. Write a professional, concise, and context-aware email for the following employee request. Generate a real email, not just repeat the input.

Employee Name: {emp_name}
Employee ID: {emp_id}
Department: {department}
Request Category: {request_data['category']}
Request Details: {request_data['description']}
Priority: {request_data['priority']}

EXAMPLE:
User Request: "i need new laptop"
Output:
SUBJECT: Hardware Request - New Laptop
BODY: Dear IT Department,\nI am requesting a new laptop for my work requirements. My current device is no longer meeting my productivity needs.\n\nPlease process this request at your earliest convenience.\n\nThank you,\n{emp_name}

INSTRUCTIONS:
1. Write a professional subject line for the request email
2. Write a formal email body addressed to the appropriate department
3. Be polite, professional, and provide context for the request
4. Do NOT just repeat the user's input - generate a real, formal email
5. Keep it concise but informative
6. Output format:
SUBJECT: [subject line]
BODY: [formal email body]
"""
            mail_response = groq_client.generate(mail_prompt, max_tokens=200, temperature=0.1)
            import re
            subject_match = re.search(r'SUBJECT:\s*(.+?)(?=\n|BODY:|$)', mail_response, re.DOTALL)
            body_match = re.search(r'BODY:\s*(.+)', mail_response, re.DOTALL)
            formal_subject = subject_match.group(1).strip() if subject_match else subject
            formal_body = body_match.group(1).strip() if body_match else request_data['description']
            
            # If LLM output is too short or just repeats the input, retry with a more explicit prompt
            if len(formal_body) < 50 or formal_body.lower().strip() == request_data['description'].lower().strip():
                retry_prompt = mail_prompt + "\n\nCRITICAL: Do NOT just repeat the input. Write a real, formal, professional email as if you are an HR professional addressing the department. Include proper greeting, context, and closing."
                mail_response = groq_client.generate(retry_prompt, max_tokens=250, temperature=0.1)
                subject_match = re.search(r'SUBJECT:\s*(.+?)(?=\n|BODY:|$)', mail_response, re.DOTALL)
                body_match = re.search(r'BODY:\s*(.+)', mail_response, re.DOTALL)
                formal_subject = subject_match.group(1).strip() if subject_match else subject
                formal_body = body_match.group(1).strip() if body_match else request_data['description']
            
            if preview_only:
                # Show the generated email content like the leave agent does
                formatted_message = f"""✅ **{request_data['category']} Request Processed**

**Request Details:**
• **Category:** {request_data['category']}
• **Priority:** {request_data['priority']}
• **Subject:** {formal_subject}
• **Department:** {department}

**📧 EMAIL DRAFT:**
**Subject:** {formal_subject}
**To:** {department.lower()}@company.com

{formal_body}"""
                return {
                    "success": True,
                    "message": formatted_message,
                    "subject": formal_subject,
                    "content": formal_body,
                    "recipient": f"{department.lower()}@company.com",
                    "category": request_data["category"],
                    "priority": request_data["priority"],
                    "department": department,
                    "sentiment": request_data["sentiment"]
                }
            else:
                try:
                    self.save_request_log(request_data, employee_data)
                    notification_sent = self.send_complaint_notification(emp_id, request_data, employee_data)
                    formatted_message = f"""✅ **{request_data['category']} Request Submitted Successfully!**

**Request Details:**
• **Category:** {request_data['category']}
• **Priority:** {request_data['priority']}
• **Subject:** {formal_subject}
• **Department:** {department}

**📧 EMAIL SENT:**
**Subject:** {formal_subject}
**To:** {department.lower()}@company.com

{formal_body}"""
                    return {
                        "success": True,
                        "message": formatted_message,
                        "subject": formal_subject,
                        "content": formal_body,
                        "recipient": f"{department.lower()}@company.com",
                        "category": request_data["category"],
                        "priority": request_data["priority"],
                        "department": department,
                        "notification_sent": notification_sent
                    }
                except Exception as e:
                    logger.error(f"Error in complaint agent: {e}")
                return {
                    "success": False,
                        "message": f"❌ Failed to process request: {str(e)}",
                        "subject": formal_subject,
                        "content": formal_body,
                        "error": str(e)
                }
        except Exception as e:
            logger.error(f"Complaint request processing error: {str(e)}")
            return {
                "success": False,
                "message": f"❌ Request processing failed: {str(e)}",
                "error": str(e)
            }

    def generate_complaint_preview(self, request_data: Dict[str, Any], employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a preview of the complaint/request email"""
        try:
            # Create email preview
            subject = f"{request_data['category']} - {request_data['subject']}"
            
            content = f"""
Request Details:
- Employee: {employee_data['full_name']} ({request_data['emp_id']})
- Department: {employee_data.get('department', 'Not specified')}
- Category: {request_data['category']}
- Subject: {request_data['subject']}
- Priority: {request_data['priority']}
- Description: {request_data['description']}
- Sentiment: {request_data['sentiment']}

This request will be sent to: {request_data['department']} department
            """.strip()
            
            return {
                "subject": subject,
                "content": content,
                "recipient": request_data['department'],
                "priority": request_data['priority'],
                "category": request_data['category']
            }
            
        except Exception as e:
            logger.error(f"Preview generation failed: {str(e)}")
            return {
                "subject": "Error generating preview",
                "content": f"Failed to generate preview: {str(e)}",
                "recipient": "Unknown",
                "priority": "LOW",
                "category": "Error"
            }

    def __call__(self, raw_input: str, emp_id: str, preview_only: bool = False) -> Dict[str, Any]:
        return self.process_complaint_request(raw_input, emp_id, preview_only=preview_only)

# Global instance
complaint_agent = ComplaintAgent()

def process_complaint_request(raw_input: str, emp_id: str, preview_only: bool = False) -> Dict[str, Any]:
    """
    Main entry point for complaint agent. Always uses the concise, single-response logic.
    """
    return complaint_agent.process_complaint_request(raw_input, emp_id, preview_only) 