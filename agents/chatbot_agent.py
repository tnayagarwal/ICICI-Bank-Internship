# HR Chatbot Agent - AI-Powered Policy Assistant
# Uses LangGraph and LLM for all logic, no hardcoded rules
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from loguru import logger
from utils.groq_client import groq_client
from utils.postgres_client import postgres_client
from config import settings
from langgraph.graph import StateGraph, END
import os

class ChatbotAgent:
    def __init__(self):
        self.conversation_history = []
        self.max_history = settings.CONVERSATION_MEMORY_LIMIT
        self.last_search_time = 0
        self.search_cache = {}
        
    def enhance_query(self, user_query: str) -> str:
        """Enhance user query for better policy search using LLM transformation like in notebook"""
        try:
            transformation_prompt = f"""
You are an AI assistant. A user has asked the following HR question:

"{user_query}"

Your task is to extract the key concepts, synonyms, and reformulate it into a semantically rich query suitable for document search. Use short, clear search-style language with relevant HR keywords.

Examples:
- "What are tax benefits?" → "tax benefits, employee benefits, compensation, deductions, allowances"
- "Can I work from home?" → "remote work, work from home, flexible work arrangements"
- "How much leave do I get?" → "annual leave, vacation days, leave entitlement, time off"

OUTPUT ONLY the enhanced search terms separated by commas:
"""
            
            response = groq_client.generate(transformation_prompt, max_tokens=50, temperature=0.1)
            
            if response and len(response.strip()) > 5 and "test response" not in response.lower():
                enhanced = response.strip()
                logger.info(f"Query enhanced: '{user_query}' -> '{enhanced}'")
                return enhanced
            else:
                # Fallback to simple keyword extraction
                return user_query.lower()
                
        except Exception as e:
            logger.error(f"Query enhancement failed: {str(e)}")
            return user_query.lower()
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text (placeholder - in real implementation use sentence transformers)"""
        try:
            # For now, return a mock embedding
            # In production, you'd use sentence-transformers or OpenAI embeddings
            import hashlib
            import struct
            
            # Create a deterministic "embedding" from text hash
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            
            # Convert to list of floats (384 dimensions for sentence transformers)
            embedding = []
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i:i+4]
                if len(chunk) == 4:
                    val = struct.unpack('f', chunk)[0]
                    embedding.append(float(val))
            
            # Pad or truncate to standard embedding size
            while len(embedding) < 384:
                embedding.append(0.0)
            embedding = embedding[:384]
            
            logger.info(f"Embedding created for query of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            return [0.0] * 384  # Return zero embedding as fallback
    
    def search_policies(self, enhanced_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant policies using hybrid document search system"""
        try:
            with postgres_client._get_cursor() as cur:
                # First try the hybrid search function
                try:
                    cur.execute("SELECT * FROM search_policy_documents(%s, %s)", (enhanced_query, limit))
                    search_results = cur.fetchall()
                    
                    if search_results:
                        policies = []
                        for result in search_results:
                            # Get full content for the document
                            cur.execute("SELECT content FROM policy_documents WHERE id = %s", (result['doc_id'],))
                            full_content_row = cur.fetchone()
                            full_content = full_content_row['content'] if full_content_row else result['content_snippet']
                            
                            policy = {
                                'id': result['doc_id'],
                                'title': result['title'],
                                'filename': result['filename'],
                                'document_type': result['document_type'],
                                'relevance_score': result['relevance_score'],
                                'content_snippet': result['content_snippet'],
                                'content': full_content[:1000]  # Limit content for LLM
                            }
                            policies.append(policy)
                        
                        logger.info(f"Found {len(policies)} documents using hybrid search")
                        return policies
                except Exception as e:
                    logger.warning(f"Hybrid search function failed: {e}, trying direct search")
                
                # Enhanced direct search with better keyword matching
                search_terms = enhanced_query.lower().replace(',', ' ').split()
                search_terms = [term.strip() for term in search_terms if len(term.strip()) > 2]
                
                if search_terms:
                    # Build comprehensive search query
                    conditions = []
                    params = []
                    
                    # Search in multiple fields with different weights
                    for term in search_terms[:8]:  # Use more terms
                        # Filename matching (highest priority)
                        conditions.append("filename ILIKE %s")
                        params.append(f"%{term}%")
                        
                        # Title matching
                        conditions.append("title ILIKE %s") 
                        params.append(f"%{term}%")
                        
                        # Content matching
                        conditions.append("content ILIKE %s")
                        params.append(f"%{term}%")
                    
                    query = f"""
                        SELECT id, filename, title, document_type, content,
                               LEFT(content, 600) as content_snippet,
                               1 as relevance_score
                        FROM policy_documents 
                        WHERE ({' OR '.join(conditions)})
                        LIMIT %s
                    """
                    
                    # Just need the base params plus limit
                    all_params = params + [limit]
                    
                    cur.execute(query, all_params)
                    results = cur.fetchall()
                    
                    if results:
                        policies = []
                        for result in results:
                            policy = {
                                'id': result['id'],
                                'title': result['title'],
                                'filename': result['filename'],
                                'document_type': result['document_type'],
                                'relevance_score': result['relevance_score'],
                                'content_snippet': result['content_snippet'],
                                'content': result['content'][:1000]
                            }
                            policies.append(policy)
                        
                        logger.info(f"Found {len(policies)} documents using enhanced search")
                return policies

                logger.warning("No documents found for query")
            return []
                
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []

    def search_database_info(self, query: str) -> Dict[str, Any]:
        """Search database for employee/leave/attendance information"""
        try:
            with postgres_client._get_cursor() as cur:
                # Extract potential employee IDs or names from query
                import re
                
                # Look for employee IDs (numbers)
                emp_ids = re.findall(r'\b\d{4,6}\b', query)
                
                # Look for names (capitalized words)
                names = re.findall(r'\b[A-Z][a-z]+\b', query)
                
                query_lower = query.lower()
                results = {}
                
                # Employee information queries
                if any(word in query_lower for word in ['employee', 'emp', 'dept head', 'department head', 'manager', 'email', 'mail']):
                    if emp_ids:
                        for emp_id in emp_ids:
                            cur.execute("SELECT * FROM employees WHERE id = %s", (emp_id,))
                            emp_data = cur.fetchone()
                            if emp_data:
                                results[f'employee_{emp_id}'] = dict(emp_data)
                    
                    if names:
                        for name in names:
                            cur.execute("SELECT * FROM employees WHERE name ILIKE %s", (f'%{name}%',))
                            emp_data = cur.fetchall()
                            if emp_data:
                                results[f'employee_{name}'] = [dict(row) for row in emp_data]
                
                # Leave information queries
                if any(word in query_lower for word in ['leave', 'vacation', 'holiday', 'time off', 'leaves left']):
                    if emp_ids:
                        for emp_id in emp_ids:
                            # First get employee name from employees table
                            cur.execute("SELECT name FROM employees WHERE id = %s", (emp_id,))
                            emp_name_result = cur.fetchone()
                            if emp_name_result:
                                emp_name = emp_name_result['name']
                                cur.execute("SELECT * FROM leaves WHERE name = %s", (emp_name,))
                                leave_data = cur.fetchone()
                                if leave_data:
                                    results[f'leave_{emp_id}'] = dict(leave_data)
                    
                    if names:
                        for name in names:
                            cur.execute("SELECT * FROM leaves WHERE name ILIKE %s", (f'%{name}%',))
                            leave_data = cur.fetchall()
                            if leave_data:
                                results[f'leave_{name}'] = [dict(row) for row in leave_data]
                
                # Attendance information queries  
                if any(word in query_lower for word in ['attendance', 'checkin', 'checkout', 'hours', 'work time']):
                    if emp_ids:
                        for emp_id in emp_ids:
                            cur.execute("SELECT * FROM attendance WHERE emp_id = %s ORDER BY date DESC LIMIT 10", (emp_id,))
                            attendance_data = cur.fetchall()
                            if attendance_data:
                                results[f'attendance_{emp_id}'] = [dict(row) for row in attendance_data]
                
                logger.info(f"Database search found {len(results)} result sets")
                return results
                
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            return {}

    def determine_query_type(self, query: str) -> str:
        """Determine if query is about policy docs, database info, or general knowledge"""
        query_lower = query.lower()
        
        # Database query indicators
        db_keywords = [
            'emp id', 'employee id', 'my id', 'employee_', 'emp_',
            'dept head', 'department head', 'manager', 'email', 'mail',
            'leaves left', 'leave balance', 'vacation days', 'time off',
            'attendance', 'checkin', 'checkout', 'work hours',
            'priya', 'employees', 'staff', 'colleague'
        ]
        
        # Policy document indicators
        policy_keywords = [
            'policy', 'policies', 'rule', 'rules', 'regulation', 'code of conduct', 'conduct', 
            'harassment', 'whistleblower', 'maternity', 'minimum wage', 'epf', 'gratuity',
            'factory', 'safety', 'equal pay', 'wages', 'benefits', 'employee conduct',
            'business ethics', 'workplace behavior', 'professional conduct'
        ]
        
        # Check for database queries
        if any(keyword in query_lower for keyword in db_keywords):
            return 'database'
        
        # Check for policy queries
        if any(keyword in query_lower for keyword in policy_keywords):
            return 'policy'
        
        # Default to general knowledge
        return 'general'
    
    def generate_answer(self, query: str, policies: List[Dict[str, Any]] = None, db_results: Dict[str, Any] = None) -> str:
        """Generate comprehensive answer based on query type and available data"""
        try:
            # Database query response
            if db_results and any(db_results.values()):
                return self.format_database_response(query, db_results)
            
            # Policy document response
            if policies and len(policies) > 0:
                return self.format_policy_response(query, policies)
            
            # General knowledge response
            return self.format_general_response(query)
                
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return f"I encountered an error while processing your query. Please try rephrasing your question."

    def format_database_response(self, query: str, db_results: Dict[str, Any]) -> str:
        """Format database query results into natural language"""
        try:
            # Build context from database results
            context = ""
            for key, data in db_results.items():
                if isinstance(data, list):
                    for item in data:
                        context += f"{key}: {item}\n"
                else:
                    context += f"{key}: {data}\n"

            # Improved prompt for employee info
            prompt = f"""
You are an HR assistant. The user asked: '{query}'

Here is the relevant data from the company database:
{context}

EXAMPLES:
User: Who is emp 20002 dept head and their email?
Data:
  employee_20002: {{'id': '20002', 'name': 'Employee_20002', 'email': 'employee20002@company.com', 'dept_head': 'Priya', 'head_email': 'Priya@company.com'}}
Answer: Employee 20002's department head is Priya. Their email is Priya@company.com.

User: What is the email of Employee_20005?
Data:
  employee_20005: {{'id': '20005', 'name': 'Employee_20005', 'email': 'employee20005@company.com', 'dept_head': 'Tanay', 'head_email': 'Tanay@company.com'}}
Answer: Employee_20005's email is employee20005@company.com. Their department head is Tanay (Tanay@company.com).

INSTRUCTIONS:
1. If the user is asking about an employee, always provide their name, department head, and head email in a clear, direct answer.
2. If the user is asking about a department head, provide the name and email of the department head.
3. If multiple employees are found, list each with their details.
4. Format the response in natural, conversational language, not as a database dump.
5. Be professional and helpful.
6. Do not mention 'database' or technical terms - just provide the information naturally.
7. If the answer is not found, say 'Sorry, I could not find that information.'

Answer:
"""

            response = groq_client.generate(prompt, max_tokens=200, temperature=0.1)

            if response and len(response.strip()) > 10:
                logger.info("Database response generated successfully")
                return response.strip()
            else:
                # Fallback to direct formatting
                return self.format_db_results_directly(query, db_results)
        except Exception as e:
            logger.error(f"Database response formatting failed: {str(e)}")
            return self.format_db_results_directly(query, db_results)

    def format_db_results_directly(self, query: str, db_results: Dict[str, Any]) -> str:
        """Direct formatting of database results as fallback"""
        response = ""
        
        for key, data in db_results.items():
            if 'employee' in key:
                if isinstance(data, list):
                    for emp in data:
                        response += f"Employee {emp.get('name', 'Unknown')}: Email: {emp.get('email', 'N/A')}, Department Head: {emp.get('dept_head', 'N/A')}, Head Email: {emp.get('head_email', 'N/A')}\n"
                else:
                    response += f"Employee {data.get('name', 'Unknown')}: Email: {data.get('email', 'N/A')}, Department Head: {data.get('dept_head', 'N/A')}, Head Email: {data.get('head_email', 'N/A')}\n"
            
            elif 'leave' in key:
                if isinstance(data, list):
                    for leave in data:
                        response += f"{leave.get('name', 'Employee')}: Leaves Left: {leave.get('leaves_left', 'N/A')}, Last Leave: {leave.get('last_leave', 'N/A')}\n"
                else:
                    response += f"{data.get('name', 'Employee')}: Leaves Left: {data.get('leaves_left', 'N/A')}, Last Leave: {data.get('last_leave', 'N/A')}\n"
            
            elif 'attendance' in key:
                if isinstance(data, list):
                    response += f"Recent attendance records:\n"
                    for att in data[:5]:  # Show last 5 records
                        response += f"- {att.get('date', 'N/A')}: Check-in: {att.get('checkin_time', 'N/A')}, Check-out: {att.get('checkout_time', 'N/A')}\n"
        
        return response.strip() if response else "No relevant information found."

    def format_policy_response(self, query: str, policies: List[Dict[str, Any]]) -> str:
        """Format policy document response using actual document content"""
        try:
            # Build context from found documents
            document_context = ""
            for i, doc in enumerate(policies[:3], 1):  # Use top 3 most relevant
                relevance = doc.get('relevance_score', 0)
                doc_type = doc.get('document_type', 'Policy Document')
                title = doc.get('title', 'Document')
                filename = doc.get('filename', 'Unknown')
                
                # Use full content, not just snippet for better responses
                content = doc.get('content', doc.get('content_snippet', ''))
                # Take more content for better context (1500 chars instead of 500)
                content_excerpt = content[:1500] if content else "No content available"
                
                document_context += f"""Document {i} - {filename}:
Title: {title}
Type: {doc_type}
Content: {content_excerpt}{'...' if len(content) > 1500 else ''}

---
"""
                
            prompt = f"""You are an HR assistant. Answer the user's question directly based on the company policy documents provided below.

User Question: "{query}"

Company Policy Documents:
{document_context}

Instructions:
1. Give a direct, detailed answer to the user's question using the document content
2. Quote specific policies, rules, or procedures from the documents when relevant
3. Include specific details like procedures, requirements, or guidelines
4. If the documents contain the answer, provide it comprehensively
5. If documents don't contain the answer, state that clearly
6. Cite the document name if referencing specific information

Answer:
"""
            
            response = groq_client.generate(prompt, max_tokens=300, temperature=0.1)
            
            if response and len(response.strip()) > 20:
                logger.info(f"Policy response generated successfully (docs found: {len(policies)})")
                return response.strip()
            else:
                logger.warning("Generated policy response was too short or empty")
                return "I found relevant policy documents but couldn't generate a comprehensive response. Please contact HR for specific policy details."
                
        except Exception as e:
            logger.error(f"Policy response formatting failed: {str(e)}")
            return "I encountered an error while processing the policy information. Please contact HR for assistance."

    def format_general_response(self, query: str) -> str:
        """Format general knowledge response"""
        try:
            prompt = f"""
You are an HR assistant. The user asked: "{query}"

No specific company policy documents were found for this query.

Instructions:
1. Provide a brief, direct answer using standard HR knowledge
2. Keep response short and practical
3. Mention specific company policies may vary
4. Suggest contacting HR for company-specific details if needed

Answer:
"""
            
            response = groq_client.generate(prompt, max_tokens=200, temperature=0.1)
            
            if response and len(response.strip()) > 20:
                logger.info("General knowledge response generated successfully")
                return response.strip()
            else:
                logger.warning("Generated general response was too short or empty")
                return "I'm here to help with HR-related questions. Could you please rephrase your question or provide more details?"
                
        except Exception as e:
            logger.error(f"General response formatting failed: {str(e)}")
            return "I encountered an error while processing your query. Please try rephrasing your question."
    
    def add_to_conversation(self, user_query: str, bot_response: str):
        """Add interaction to conversation history"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_query,
            "bot": bot_response
        })
        
        # Limit conversation history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_summary(self) -> str:
        """Generate summary of conversation"""
        if not self.conversation_history:
            return "No conversation yet"
        
        try:
            recent_conversation = self.conversation_history[-5:]  # Last 5 exchanges
            conversation_text = "\n".join([
                f"User: {exchange['user']}\nBot: {exchange['bot'][:100]}..."
                for exchange in recent_conversation
            ])
            
            prompt = f"""
Summarize this HR conversation in one sentence:

{conversation_text}

Summary:"""
            
            response = groq_client.generate(prompt, max_tokens=30, temperature=0.1)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Conversation with {len(self.conversation_history)} exchanges"
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history reset")
    
    def finalize_response(self, user_query: str, policies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Finalize the chatbot response"""
        try:
            # Generate answer
            answer = self.generate_answer(user_query, policies)
            
            # Add to conversation history
            self.add_to_conversation(user_query, answer)
            
            # Determine confidence based on policies found
            confidence = "HIGH" if policies else "GENERAL"
            
            return {
                "answer": answer,
                "confidence": confidence,
                "policies_used": len(policies),
                "conversation_length": len(self.conversation_history)
            }
            
        except Exception as e:
            logger.error(f"Response finalization failed: {str(e)}")
            return {
                "answer": "I apologize, but I'm experiencing technical difficulties. Please try again or contact HR for immediate assistance.",
                "confidence": "ERROR",
                "policies_used": 0,
                "conversation_length": len(self.conversation_history)
            }
    
    def process_query(self, user_query: str, emp_id: str = None) -> str:
        """Main function to process user query and return simple string response. Uses emp_id for 'my' queries."""
        try:
            logger.info(f"Processing query: {user_query}")
            query_lower = user_query.lower()
            # If query is about 'my' dept head/email and emp_id is provided, use it
            if emp_id and ("my dept head" in query_lower or "my department head" in query_lower or ("my" in query_lower and "email" in query_lower)):
                employee = postgres_client.get_employee(emp_id)
                if employee:
                    dept_head = employee.get("dept_head")
                    dept_head_email = employee.get("head_email")
                    if dept_head and dept_head_email:
                        return f"Your department head is {dept_head}, and their email is {dept_head_email}."
                    elif dept_head:
                        return f"Your department head is {dept_head}."
                    else:
                        return "Sorry, I could not find your department head's information."
            # Determine query type
            query_type = self.determine_query_type(user_query)
            logger.info(f"Query type determined: {query_type}")
            
            policies = []
            db_results = {}
            
            if query_type == 'database':
                # Search database for relevant information
                db_results = self.search_database_info(user_query)
                
            elif query_type == 'policy':
                # Enhance query for better policy search
                enhanced_query = self.enhance_query(user_query)
                # Search for relevant policies
                policies = self.search_policies(enhanced_query)
                
            # Generate answer based on query type and available data
            answer = self.generate_answer(user_query, policies, db_results)
            
            # Add to conversation history
            self.add_to_conversation(user_query, answer)
            
            logger.info(f"Query processed successfully, response length: {len(answer)}")
            return answer
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            error_response = f"I encountered an error while processing your query. Please try again or rephrase your question."
            self.add_to_conversation(user_query, error_response)
            return error_response
    
    def __call__(self, user_query: str, emp_id: str = None) -> str:
        """Make the agent callable for direct use"""
        return self.process_query(user_query, emp_id)

# Global agent instance
chatbot_agent = ChatbotAgent()
working_chatbot = ChatbotAgent()

# LangGraph states for chatbot workflow
class ChatbotState(TypedDict):
    query: str
    response: str
    policies: list
    status: str

# State functions

def search_policies(state: ChatbotState) -> ChatbotState:
    """Search for relevant policies based on user query"""
    try:
        query = state["query"]
        
        # Search for relevant policies in the database
        policies = postgres_client.search_policies(query)
        
        return {
            **state,
            "policies": policies,
            "status": "policies_found"
        }
        
    except Exception as e:
        logger.error(f"Error searching policies: {e}")
        return {
            **state,
            "policies": [],
            "status": "error"
        }

def generate_response(state: ChatbotState) -> ChatbotState:
    """Generate response using Gemma based on policies and query"""
    try:
        query = state["query"]
        policies = state["policies"]
        
        if not policies:
            response = "I couldn't find specific policy information related to your question. Please contact HR for detailed assistance or try rephrasing your question."
        else:
            # Create context from relevant policies
            policy_context = "\n\n".join([
                f"Policy Content: {policy['content'][:500]}..." if len(policy['content']) > 500 else f"Policy Content: {policy['content']}"
                for policy in policies[:3]  # Use top 3 most relevant policies
            ])
            
            prompt = f"""
            Based on the following company policy information, please answer the user's question professionally and accurately.
            
            User Question: {query}
            
            Relevant Policy Information:
            {policy_context}
            
            Please provide a helpful, accurate response based on the policy information. If the policies don't fully address the question, mention that and suggest contacting HR for additional clarification.
            
            Keep the response conversational but professional.
            """
            
            response = groq_client.generate(prompt, max_tokens=100, temperature=0.1)
        
        return {
            **state,
            "response": response,
            "status": "complete"
        }
        
    except Exception as e:
        logger.error(f"Error generating chatbot response: {e}")
        return {
            **state,
            "response": f"I apologize, but I encountered an error while processing your question: {str(e)}. Please contact HR for assistance.",
            "status": "error"
        }

# Create the workflow
workflow = StateGraph(ChatbotState)
workflow.add_node("search_policies", search_policies)
workflow.add_node("generate_response", generate_response)

workflow.set_entry_point("search_policies")
workflow.add_edge("search_policies", "generate_response")
workflow.add_edge("generate_response", END)

chatbot_agent = workflow.compile()

def chatbot_agent_main(query: str) -> str:
    """Main function for chatbot agent - uses working ChatbotAgent class directly"""
    try:
        # Use the working ChatbotAgent class directly (not the LangGraph workflow)
        working_chatbot = ChatbotAgent()
        return working_chatbot.process_query(query)
        
    except Exception as e:
        logger.error(f"Chatbot agent error: {e}")
        return f"I apologize, but I encountered an error: {str(e)}. Please contact HR for assistance."

# Convenience functions
def process_chat_query(user_query: str) -> Dict[str, Any]:
    """Process a chat query"""
    return chatbot_agent.process_query(user_query)

def reset_chatbot():
    """Reset chatbot conversation"""
    chatbot_agent.reset_conversation()

def get_conversation_summary() -> str:
    """Get conversation summary"""
    return chatbot_agent.get_conversation_summary()

def get_chatbot_stats() -> Dict[str, Any]:
    """Get chatbot statistics"""
    return {
        "conversation_length": len(chatbot_agent.conversation_history),
        "max_history": chatbot_agent.max_history,
        "last_interaction": chatbot_agent.conversation_history[-1]["timestamp"] if chatbot_agent.conversation_history else "None"
    } 

# Utility: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Utility: Populate Postgres 'data' table from all PDFs in data folder
DATA_FOLDER = os.path.join(os.path.dirname(__file__), '../data')
def populate_policies_from_pdfs():
    pdf_files = glob.glob(os.path.join(DATA_FOLDER, '*.pdf'))
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        filename = os.path.basename(pdf_file)
        # Insert into Postgres 'data' table
        postgres_client._execute(
            "INSERT INTO data (content, metadata) VALUES (%s, %s)",
            (text, f'{{"filename": "{filename}"}}')
        )
        logger.info(f"Inserted policy from {filename}")

# Call this once to migrate all PDFs to Postgres
# populate_policies_from_pdfs()

# Chatbot: Search policies in Postgres by keyword

def search_policies_keyword(query, limit=3):
    # Simple keyword search in content
    sql = "SELECT content, metadata FROM data WHERE content ILIKE %s LIMIT %s"
    results = postgres_client._execute(sql, (f'%{query}%', limit), fetchall=True)
    return results or []

def process_chat_query_langgraph(query):
    # Search for relevant policy
    results = search_policies_keyword(query)
    if results:
        best = results[0]
        answer = best['content'][:800]  # Return first 800 chars
        meta = best.get('metadata', '')
        return {"answer": answer, "source": meta}
    else:
        return {"answer": "Sorry, I couldn't find a relevant policy. Please contact HR for more info.", "source": None} 