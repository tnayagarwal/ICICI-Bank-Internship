"""
Email Client for HR System Notifications
Handles email sending with templates and error recovery
"""
import smtplib
import time
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, List, Optional, Union
from loguru import logger
from config import settings
from datetime import datetime

logger = logging.getLogger(__name__)

class EmailClientError(Exception):
    """Custom exception for email client errors"""
    pass

class EmailClient:
    def __init__(self):
        # Email configuration - you can update these settings
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.username = "hr@company.com"  # Replace with actual email
        self.password = "your_app_password"  # Replace with actual password
        self.connection_cache = {}
        self.last_send_time = 0
        self.min_send_interval = 1  # Rate limiting
        
    def _rate_limit(self):
        """Rate limiting to avoid email server throttling"""
        current_time = time.time()
        if current_time - self.last_send_time < self.min_send_interval:
            time.sleep(self.min_send_interval - (current_time - self.last_send_time))
        self.last_send_time = time.time()
    
    def _create_message(
        self, 
        to_email: str, 
        subject: str, 
        body: str,
        cc_email: Optional[str] = None,
        is_html: bool = False
    ) -> MIMEMultipart:
        """Create email message"""
        msg = MIMEMultipart()
        msg["From"] = self.username
        msg["To"] = to_email
        msg["Subject"] = subject
        
        recipients = [to_email]
        if cc_email:
            msg["Cc"] = cc_email
            recipients.append(cc_email)
        
        # Attach body
        msg.attach(MIMEText(body, "html" if is_html else "plain"))
        
        return msg, recipients
    
    def send_email(self, to_email: str, subject: str, body: str, from_email: str = None) -> bool:
        """Send email using SMTP"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_email or self.username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body to email
            msg.attach(MIMEText(body, 'plain'))
            
            # For now, just log the email instead of actually sending
            # In production, uncomment the SMTP code below
            logger.info(f"EMAIL SENT (simulated)")
            logger.info(f"To: {to_email}")
            logger.info(f"From: {from_email or self.username}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Body: {body[:200]}...")
            
            # Uncomment below for actual email sending:
            # server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            # server.starttls()
            # server.login(self.username, self.password)
            # text = msg.as_string()
            # server.sendmail(from_email or self.username, to_email, text)
            # server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_attendance_notification(
        self, 
        emp_id: str, 
        action: str, 
        time_str: str,
        employee_info: Dict[str, Any]
    ) -> bool:
        """Send attendance notification"""
        subject = f"Attendance {action.title()} - Employee {emp_id}"
        
        body = f"""
Dear {employee_info.get('dept head', 'Manager')},

Employee ID: {emp_id}
Action: {action.title()}
Time: {time_str}
Date: {time.strftime('%Y-%m-%d')}

This is an automated notification from the HR Attendance System.

Best regards,
HR System
"""
        
        return self.send_email(
            to_email=employee_info.get('head email', ''),
            subject=subject,
            body=body,
            from_email=self.username
        )
    
    def send_leave_request(
        self,
        emp_id: str,
        leave_details: Dict[str, Any],
        employee_info: Dict[str, Any],
        action: str = "REQUEST"
    ) -> bool:
        """Send leave request notification"""
        
        if action == "AUTO_APPROVE":
            subject = f"Leave Approved - Employee {emp_id}"
            body = f"""
Dear {employee_info.get('dept head', 'Manager')},

The following leave request has been automatically approved:

Employee ID: {emp_id}
Leave Period: {leave_details.get('start', '')} to {leave_details.get('end', '')}
Duration: {leave_details.get('num_days', 0)} days
Reason: {leave_details.get('reason', '')}
Remaining Leaves: {employee_info.get('leaves', 0) - leave_details.get('num_days', 0)}

This leave met auto-approval criteria (≤2 days, sufficient balance, adequate gap from last leave).

Best regards,
HR System
"""
        
        elif action == "SEND_REQUEST":
            subject = f"Leave Request - Employee {emp_id}"
            body = f"""
Dear {employee_info.get('dept head', 'Manager')},

You have received a new leave request:

Employee ID: {emp_id}
Leave Period: {leave_details.get('start', '')} to {leave_details.get('end', '')}
Duration: {leave_details.get('num_days', 0)} days
Reason: {leave_details.get('reason', '')}
Current Leave Balance: {employee_info.get('leaves', 0)}

Please review and approve/reject this request.

Best regards,
HR System
"""
        
        elif action == "TOO_SOON":
            subject = f"Leave Request Escalation - Employee {emp_id}"
            body = f"""
Dear {employee_info.get('dept head', 'Manager')},

ATTENTION: Frequent leave request detected.

Employee ID: {emp_id}
Requested Leave: {leave_details.get('start', '')} to {leave_details.get('end', '')}
Duration: {leave_details.get('num_days', 0)} days
Reason: {leave_details.get('reason', '')}
Last Leave Date: {employee_info.get('last leave', '')}

This request is within 4 days of the last leave. Please review the pattern and take appropriate action.

Best regards,
HR System
"""
        
        elif action == "ASK_OVERRIDE":
            subject = f"Leave Request - Insufficient Balance - Employee {emp_id}"
            body = f"""
Dear {employee_info.get('dept head', 'Manager')},

Leave request requiring override approval:

Employee ID: {emp_id}
Requested Leave: {leave_details.get('start', '')} to {leave_details.get('end', '')}
Duration: {leave_details.get('num_days', 0)} days
Reason: {leave_details.get('reason', '')}
Current Leave Balance: {employee_info.get('leaves', 0)}

Employee has insufficient leave balance. Please review for possible override.

Best regards,
HR System
"""
        
        return self.send_email(
            to_email=employee_info.get('head email', ''),
            subject=subject,
            body=body,
            from_email=self.username
        )
    
    def send_complaint_notification(
        self,
        emp_id: str,
        complaint_details: Dict[str, Any],
        employee_info: Dict[str, Any]
    ) -> bool:
        """Send complaint/request notification"""
        
        subject = complaint_details.get('subject', f"Employee Request - {emp_id}")
        body = complaint_details.get('content', f"""
Dear {employee_info.get('dept head', 'Manager')},

You have received a new request from Employee {emp_id}.

Please review and take appropriate action.

Best regards,
HR System
""")
        
        return self.send_email(
            to_email=employee_info.get('head email', ''),
            subject=subject,
            body=body,
            from_email=self.username
        )
    
    def send_overtime_report(
        self,
        emp_id: str,
        overtime_summary: str,
        employee_info: Dict[str, Any]
    ) -> bool:
        """Send overtime report to specific employee only"""
        try:
            subject = f"Weekly Overtime Report - {emp_id}"
            
            body = f"""
Weekly Overtime Summary:
{overtime_summary}

Employee: {emp_id}
Week: {datetime.now().strftime('%Y-%m-%d')}
            """.strip()
            
            return self.send_email(
                to_email=employee_info.get("email", f"employee{emp_id}@company.com"),
                subject=subject,
                body=body,
                from_email=self.username
            )
            
        except Exception as e:
            logger.error(f"Overtime report email failed: {str(e)}")
            return False
    
    def send_system_alert(
        self,
        alert_type: str,
        message: str,
        admin_emails: List[str]
    ) -> bool:
        """Send system alerts to administrators"""
        
        subject = f"HR System Alert - {alert_type}"
        body = f"""
HR System Alert

Alert Type: {alert_type}
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{message}

Please investigate if necessary.

Best regards,
HR System
"""
        
        success = True
        for admin_email in admin_emails:
            if not self.send_email(admin_email, subject, body, self.username):
                success = False
        
        return success
    
    def test_connection(self) -> bool:
        """Test email server connection"""
        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.username, self.password)
            logger.info("Email server connection test successful")
            return True
        except Exception as e:
            logger.error(f"Email server connection test failed: {str(e)}")
            return False

# Global instance
email_client = EmailClient()

# Convenience functions
def send_notification(
    notification_type: str,
    recipient_email: str,
    subject: str,
    body: str,
    cc_email: Optional[str] = None
) -> bool:
    """Quick notification sending"""
    return email_client.send_email(recipient_email, subject, body, None)

def notify_attendance(emp_id: str, action: str, time_str: str, employee_info: Dict[str, Any]) -> bool:
    """Quick attendance notification"""
    return email_client.send_attendance_notification(emp_id, action, time_str, employee_info)

def notify_leave_request(emp_id: str, leave_details: Dict[str, Any], employee_info: Dict[str, Any], action: str) -> bool:
    """Quick leave notification"""
    return email_client.send_leave_request(emp_id, leave_details, employee_info, action)

def notify_complaint(emp_id: str, complaint_details: Dict[str, Any], employee_info: Dict[str, Any]) -> bool:
    """Quick complaint notification"""
    return email_client.send_complaint_notification(emp_id, complaint_details, employee_info)

def send_email(to_email: str, subject: str, body: str, from_email: str = None) -> bool:
    """Send email wrapper function"""
    return email_client.send_email(to_email, subject, body, from_email) 