import json
import importlib
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
from pathlib import Path
from loguru import logger

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from utils.gemma_client import gemma_client
from utils.postgres_client import postgres_client
from utils.groq_client import groq_client

class WorkflowState(TypedDict):
    """State for LangGraph workflow"""
    emp_id: str
    query: str
    intent: str
    agents_to_execute: List[str]
    agent_results: Dict[str, Any]
    shared_context: Dict[str, Any]
    workflow_type: str
    can_proceed: bool
    final_response: str
    error: Optional[str]

class DynamicAgentWrapper:
    """Dynamic wrapper for agents loaded from configuration"""
    
    def __init__(self, agent_config: Dict[str, Any]):
        self.config = agent_config
        self.name = agent_config["name"]
        self.capabilities = agent_config["capabilities"]
        self.triggers = agent_config["triggers"]
        self.agent_instance = None
        self._load_agent_instance()
        
    def _load_agent_instance(self):
        """Dynamically load agent instance from module path"""
        try:
            module_path = self.config["functions"]["module_path"]
            module = importlib.import_module(module_path)
            
            # Try to get the agent class (assume class name matches config name)
            agent_class_name = self.name
            if hasattr(module, agent_class_name):
                agent_class = getattr(module, agent_class_name)
                self.agent_instance = agent_class()
                logger.info(f"✅ Loaded agent: {self.name}")
            else:
                logger.warning(f"⚠️ Agent class {agent_class_name} not found in {module_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load agent {self.name}: {str(e)}")
            
    def execute(self, query: str, emp_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute agent with dynamic configuration"""
        if not self.agent_instance:
            return {
                "success": False,
                "message": f"Agent {self.name} not properly loaded",
                "error": "Agent initialization failed"
            }
        try:
            context = context or {}
            process_function_name = self.config["functions"]["process_function"]
            if hasattr(self.agent_instance, process_function_name):
                process_function = getattr(self.agent_instance, process_function_name)
                # Call with appropriate parameters based on agent type
                if "attendance" in self.name.lower():
                    operation = self._parse_attendance_operation(query)
                    overtime_query = context.get("overtime_query", "")
                    result = process_function(emp_id, operation, overtime_query)
                elif "leave" in self.name.lower():
                    preview_only = context.get("preview_only", False)
                    result = process_function(query, emp_id, preview_only)
                elif "complaint" in self.name.lower():
                    preview_only = context.get("preview_only", False)
                    result = process_function(query, emp_id, preview_only)
                elif "employee" in self.name.lower():
                    result = process_function(query, emp_id)
                elif "chatbot" in self.name.lower():
                    result = process_function(query, emp_id)
                else:
                    # Generic function call for new agent types
                    result = process_function(query, emp_id, context)
                return self._standardize_result(result)
            else:
                return {
                    "success": False,
                    "message": f"Process function {process_function_name} not found in {self.name}",
                    "error": "Function not found"
                }
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            return {
                "success": False,
                "message": f"Agent execution failed: {str(e)}",
                "error": str(e)
            }
    
    def _parse_attendance_operation(self, query: str) -> str:
        """Parse attendance operation using LLM for intelligent understanding"""
        try:
            prompt = f"""
You are an attendance operation classifier. Analyze the query and determine the intended attendance action.

Query: "{query}"

Available operations:
- checkin: For checking in, signing in, clocking in, starting work
- checkout: For checking out, signing out, clocking out, ending work  
- overtime: For overtime calculations, OT queries, work hour summaries

Examples:
- "check in" → checkin
- "checkin" → checkin
- "sign in" → checkin
- "clock in" → checkin
- "start work" → checkin
- "begin shift" → checkin
- "check out" → checkout
- "checkout" → checkout
- "check me out" → checkout
- "check you out" → checkout
- "sign out" → checkout
- "clock out" → checkout
- "end work" → checkout
- "finish shift" → checkout
- "overtime" → overtime
- "ot" → overtime
- "weekly overtime" → overtime
- "monthly overtime" → overtime
- "work hours" → overtime
- "attendance summary" → overtime

Respond with ONLY the operation name (checkin, checkout, or overtime). No other text.
"""
            
            response = groq_client.generate(prompt, max_tokens=10)
            operation = response.strip().lower()
            
            # Validate the response
            if operation in ["checkin", "checkout", "overtime"]:
                return operation
            else:
                # Fallback to simple keyword matching if LLM response is invalid
                query_lower = query.lower()
                if any(word in query_lower for word in ["overtime", "ot"]):
                    return "overtime"
                elif any(word in query_lower for word in ["out", "sign out", "clock out"]):
                    return "checkout"
                else:
                    return "checkin"  # Default to checkin
                    
        except Exception as e:
            logger.error(f"LLM attendance parsing failed: {str(e)}")
            # Fallback to simple keyword matching
            query_lower = query.lower()
            if any(word in query_lower for word in ["overtime", "ot"]):
                return "overtime"
            elif any(word in query_lower for word in ["out", "sign out", "clock out"]):
                return "checkout"
            else:
                return "checkin"  # Default to checkin
    
    def _standardize_result(self, result: Any) -> Dict[str, Any]:
        """Standardize agent result format"""
        if isinstance(result, dict):
            if "success" in result:
                return result
            else:
                return {
                    "success": True,
                    "message": result.get("message", str(result)),
                    "data": result
                }
        elif isinstance(result, str):
            return {
                "success": True,
                "message": result,
                "data": {"response": result}
            }
        else:
            return {
                "success": True,
                "message": str(result),
                "data": {"response": result}
            }

class LangGraphOrchestrator:
    """LangGraph-based dynamic orchestrator"""
    
    def __init__(self, config_path: str = "agents/agents.json"):
        self.config_path = config_path
        self.agents = {}
        self.config = None
        self.workflow_graph = None
        self._load_configuration()
        self._initialize_agents()
        self._build_workflow_graph()
        
    def _load_configuration(self):
        """Load agent configuration from JSON file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return
                
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                
            logger.info(f"📋 Loaded configuration from {self.config_path}")
            logger.info(f"📊 Found {len(self.config.get('agents', {}))} agent configurations")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            self.config = {"agents": {}}
    
    def _initialize_agents(self):
        """Initialize all agents from configuration"""
        if not self.config or "agents" not in self.config:
            logger.error("No agent configuration found")
            return
            
        for agent_key, agent_config in self.config["agents"].items():
            try:
                # Create dynamic wrapper
                wrapper = DynamicAgentWrapper(agent_config)
                
                # Store agent
                self.agents[agent_key] = wrapper
                
                logger.info(f"✅ Initialized agent: {agent_key} ({agent_config['name']})")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize agent {agent_key}: {str(e)}")
        
        logger.info(f"🚀 LangGraph Orchestrator initialized with {len(self.agents)} agents")
    
    def _build_workflow_graph(self):
        """Build LangGraph workflow"""
        try:
            workflow = StateGraph(dict)
            
            # Add nodes
            workflow.add_node("analyze_intent", self._analyze_intent_node)
            workflow.add_node("execute_agents", self._execute_agents_node)
            workflow.add_node("compile_response", self._compile_response_node)
            
            # Set entry point
            workflow.set_entry_point("analyze_intent")
            
            # Linear flow
            workflow.add_edge("analyze_intent", "execute_agents")
            workflow.add_edge("execute_agents", "compile_response")
            workflow.add_edge("compile_response", END)
            
            self.workflow_graph = workflow.compile()
            logger.info("🔗 LangGraph workflow built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build LangGraph workflow: {e}")
            # Create a simple workflow without LangGraph
            self.workflow_graph = None
            logger.info("⚠️ Using simple workflow without LangGraph")
    
    def _analyze_intent_node(self, state):
        print("[LangGraph] Entering analyze_intent_node", state)
        try:
            query = state.get("query", "")
            intent_classification = self._analyze_intent(query)
            
            if intent_classification.startswith("SINGLE:"):
                agent_name = intent_classification.split(":", 1)[1].strip()
                state["intent"] = intent_classification
                state["agents_to_execute"] = [agent_name] if agent_name in self.agents else []
                state["workflow_type"] = "single_agent"
                
            elif intent_classification.startswith("MULTI:"):
                agent_list = intent_classification.split(":", 1)[1].strip().split(",")
                agent_list = [agent.strip() for agent in agent_list if agent.strip() in self.agents]
                state["intent"] = intent_classification
                state["agents_to_execute"] = agent_list
                state["workflow_type"] = "multi_agent"
                
            elif intent_classification.startswith("COMPLEX:"):
                tasks = self.parse_complex_query(query)
                agent_list = [task.get("agent", "") for task in tasks if task.get("agent") in self.agents]
                state["intent"] = intent_classification
                state["agents_to_execute"] = agent_list
                state["workflow_type"] = "sequential_workflow"
                
            else:
                # Use trigger-based routing from agents.json
                best_agent = self._find_best_agent_by_triggers(query)
                state["intent"] = f"SINGLE:{best_agent}"
                state["agents_to_execute"] = [best_agent] if best_agent else []
                state["workflow_type"] = "single_agent"
            
            state["can_proceed"] = len(state["agents_to_execute"]) > 0
            state["agent_results"] = {}
            state["shared_context"] = {}
            state["error"] = None
            
        except Exception as e:
            state["error"] = f"Intent analysis failed: {str(e)}"
            state["can_proceed"] = False
        print("[LangGraph] Exiting analyze_intent_node", state)
        return dict(state)
    
    def _execute_agents_node(self, state):
        print("[LangGraph] Entering execute_agents_node", state)
        try:
            query = state.get("query", "")
            emp_id = state.get("emp_id", "")
            
            if state.get("workflow_type") == "sequential_workflow":
                # Parse complex query to get specific tasks
                tasks = self.parse_complex_query(query)
                state["shared_context"]["tasks"] = tasks  # Store tasks for response compilation
                
                for i, task in enumerate(tasks):
                    agent_name = task.get("agent")
                    sub_query = task.get("sub_query", query)
                    condition = task.get("condition")
                    
                    # Check if this task has a condition and evaluate it
                    if condition and i > 0:
                        # Get previous task result to evaluate condition
                        prev_task_key = f"{list(state['agent_results'].keys())[-1]}"
                        prev_result = state["agent_results"].get(prev_task_key, {})
                        
                        # Evaluate condition based on previous result
                        should_execute = self._evaluate_condition(condition, prev_result, state["shared_context"])
                        
                        if not should_execute:
                            logger.info(f"⏭️ Skipping conditional task {i+1}: {agent_name} (condition not met)")
                            state["agent_results"][f"{agent_name}_{i}"] = {
                                "success": True,
                                "message": f"Task skipped - condition not met: {condition}",
                                "skipped": True,
                                "condition": condition
                            }
                            continue
                    
                    if agent_name in self.agents:
                        agent_wrapper = self.agents[agent_name]
                        result = agent_wrapper.execute(sub_query, emp_id, state["shared_context"])
                        # Store result with task index to avoid overwriting
                        state["agent_results"][f"{agent_name}_{i}"] = result
                        
                        if result.get("success"):
                            state["shared_context"][f"{agent_name}_{i}_output"] = result.get("data", {})
                        
                        logger.info(f"✅ Executed sequential agent {i+1}: {agent_name} with sub_query: {sub_query[:50]}...")
                    else:
                        state["agent_results"][f"{agent_name}_{i}"] = {
                            "success": False,
                            "message": f"Agent {agent_name} not found",
                            "error": "Agent not available"
                        }
            else:
                # Execute all agents in parallel for single/multi agent workflows
                for agent_name in state.get("agents_to_execute", []):
                    if agent_name in self.agents:
                        agent_wrapper = self.agents[agent_name]
                        result = agent_wrapper.execute(query, emp_id, state["shared_context"])
                        state["agent_results"][agent_name] = result
                        
                        if result.get("success"):
                            state["shared_context"][f"{agent_name}_output"] = result.get("data", {})
                        
                        logger.info(f"✅ Executed agent: {agent_name}")
                    else:
                        state["agent_results"][agent_name] = {
                            "success": False,
                            "message": f"Agent {agent_name} not found",
                            "error": "Agent not available"
                        }
            
            state["can_proceed"] = True
            state["error"] = None
            
        except Exception as e:
            state["error"] = f"Agent execution failed: {str(e)}"
            state["can_proceed"] = False
        print("[LangGraph] Exiting execute_agents_node", state)
        return dict(state)
    
    def _compile_response_node(self, state):
        print("[LangGraph] Entering compile_response_node", state)
        try:
            if state.get("workflow_type") == "single_agent":
                # Single agent response
                agent_name = state.get("agents_to_execute", [""])[0]
                result = state["agent_results"].get(agent_name, {})
                state["final_response"] = result.get("message", "No response generated")
                
            elif state.get("workflow_type") == "multi_agent":
                # Multi-agent response compilation
                compiled_messages = []
                for agent_name in state.get("agents_to_execute", []):
                    result = state["agent_results"].get(agent_name, {})
                    if result.get("success"):
                        agent_title = agent_name.title()
                        compiled_messages.append(f"**{agent_title} Agent Response:**\n{result.get('message', '')}")
                
                state["final_response"] = "\n\n".join(compiled_messages) if compiled_messages else "No successful responses"
                
            elif state.get("workflow_type") == "sequential_workflow":
                # Sequential workflow response
                tasks = state["shared_context"].get("tasks", [])
                formatted_steps = []
                for i, task in enumerate(tasks):
                    agent_name = task.get("agent")
                    result_key = f"{agent_name}_{i}"
                    result = state["agent_results"].get(result_key, {})
                    condition = task.get("condition")
                    
                    step_num = i + 1
                    total = len(tasks)
                    agent_title = agent_name.title()
                    
                    # Handle conditional tasks
                    if condition:
                        header = f"**Step {step_num}/{total} - {agent_title} Agent (Conditional)**"
                        if result.get("skipped"):
                            content = f"⏭️ **Skipped:** {result.get('message', 'Condition not met')}"
                            formatted_steps.append(f"{header}\n\n{content}")
                        else:
                            content = result.get("message", "")
                            formatted_steps.append(f"{header}\n\n{content}")
                    else:
                        header = f"**Step {step_num}/{total} - {agent_title} Agent**"
                        
                        if result.get("success"):
                            content = result.get("message", "")
                            formatted_steps.append(f"{header}\n\n{content}")
                        else:
                            error_msg = result.get("error", "Task failed")
                            formatted_steps.append(f"{header}\n\n❌ Error: {error_msg}")
                
                state["final_response"] = "\n\n".join(formatted_steps)
            
            else:
                state["final_response"] = "Workflow type not supported"
            
            state["can_proceed"] = True
            state["error"] = None
            
        except Exception as e:
            state["error"] = f"Response compilation failed: {str(e)}"
            state["can_proceed"] = False
        print("[LangGraph] Exiting compile_response_node", state)
        return dict(state)
    
    def _find_best_agent_by_triggers(self, query: str) -> str:
        """Find the best agent using triggers from agents.json"""
        query_lower = query.lower()
        best_agent = None
        best_score = 0
        
        for agent_key, agent_config in self.config.get("agents", {}).items():
            triggers = agent_config.get("triggers", [])
            score = sum(1 for trigger in triggers if trigger.lower() in query_lower)
            if score > best_score:
                best_score = score
                best_agent = agent_key
        
        return best_agent
    
    def _analyze_intent(self, query: str) -> str:
        """Analyze intent using LLM with dynamic agent list"""
        try:
            # Get available agents from configuration
            available_agents = list(self.config.get("agents", {}).keys())
            agent_list = ", ".join(available_agents)
            
            prompt = f"""
You are an intent classifier. Analyze the query and respond ONLY with the exact format specified.

Available agents: {agent_list}

Query: "{query}"

IMPORTANT: Respond with ONLY one of these exact formats (no other text):
- SINGLE:agent_name
- MULTI:agent1,agent2
- COMPLEX:sequential

Rules:
1. If the query is about a company policy, act, regulation, rule, or any HR policy (e.g., "whistleblower policy", "minimum wage act", "maternity policy", "leave policy"), always route to SINGLE:chatbot.
2. If query is about attendance actions ("check in", "check out", "sign in", "sign out", "clock in", "clock out", "overtime", "ot", "attendance summary"), always route to SINGLE:attendance.
3. If query is about filing a complaint, issue, or request, route to SINGLE:complaint.
4. If query has conditional logic (e.g., "check balance AND IF more than 2 days apply for leave") → COMPLEX:sequential
5. If query has multiple independent tasks that can run in parallel (e.g., "apply for leave and file a complaint") → MULTI:agent1,agent2
6. If query has multiple dependent tasks (e.g., "check balance and apply leave") → COMPLEX:sequential
7. If query has 1 task → SINGLE:agent_name

Examples:
- "what is the whistleblower policy" → SINGLE:chatbot
- "minimum wage act" → SINGLE:chatbot
- "overtime" → SINGLE:attendance
- "weekly overtime" → SINGLE:attendance
- "attendance summary" → SINGLE:attendance
- "check my attendance" → SINGLE:attendance
- "check in" → SINGLE:attendance
- "check me out" → SINGLE:attendance
- "apply for leave and file a complaint" → MULTI:leave,complaint
- "check balance and request laptop" → COMPLEX:sequential
- "check my leave balance AND IF it's more than 2 days apply for leave" → COMPLEX:sequential
- "check balance AND IF sufficient apply for leave" → COMPLEX:sequential
- "I have an issue with my laptop" → SINGLE:complaint
- "file a complaint about my manager" → SINGLE:complaint
- "i need new laptop" → SINGLE:complaint
- "request a new laptop" → SINGLE:complaint
- "my laptop is broken" → SINGLE:complaint
- "need IT support" → SINGLE:complaint
- "hardware request" → SINGLE:complaint
- "i want to request a wheelchair" → SINGLE:complaint
- "request wheelchair from office" → SINGLE:complaint
- "need wheelchair my leg is broken" → SINGLE:complaint
- "request equipment from office" → SINGLE:complaint
- "need medical equipment" → SINGLE:complaint
- "request assistive device" → SINGLE:complaint
- "i want to request" → SINGLE:complaint
- "need help with" → SINGLE:complaint
- "report an issue" → SINGLE:complaint
- "file a request" → SINGLE:complaint

Response (ONLY the format, no explanation):
"""
            # Fallback: if query is exactly 'overtime', 'weekly overtime', 'monthly overtime', or 'attendance summary', force attendance
            if query.strip().lower() in ["overtime", "weekly overtime", "monthly overtime", "attendance summary"]:
                return "SINGLE:attendance"
            
            response = groq_client.generate(prompt, max_tokens=20)
            response = response.strip()
            
            # Validate response format
            if response.startswith(("SINGLE:", "MULTI:", "COMPLEX:")):
                return response
            else:
                # Force complex workflow for multi-task queries
                if any(keyword in query.lower() for keyword in ["and also", "and", "also", ","]):
                    return "COMPLEX:sequential"
                else:
                    # Use trigger-based fallback
                    best_agent = self._find_best_agent_by_triggers(query)
                    return f"SINGLE:{best_agent}" if best_agent else "SINGLE:chatbot"
        
        except Exception as e:
            logger.error(f"Intent analysis failed: {str(e)}")
            # Use trigger-based fallback
            best_agent = self._find_best_agent_by_triggers(query)
            return f"SINGLE:{best_agent}" if best_agent else "SINGLE:chatbot"
    
    def parse_complex_query(self, query: str) -> List[Dict[str, Any]]:
        """Parse complex query into sequential tasks using pure LLM logic"""
        try:
            # Get available agents from configuration
            available_agents = list(self.config.get("agents", {}).keys())
            agent_list = ", ".join(available_agents)
            
            prompt = f"""
You are a task parser. Parse this complex query into sequential tasks with proper condition extraction.

Query: "{query}"

Available agents: {agent_list}

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON array - no explanations, no markdown, no code blocks
2. Extract conditions from natural language and convert to logical format
3. Handle "if its more than X days available" → "leaves_left > X"
4. Handle "if sufficient" → "leaves_left > 0"
5. Handle "if more than X hours" → "overtime_hours > X"
6. Handle "if below X%" → "attendance_percentage < X"

EXAMPLES:

Query: "check my leave balance and if its more than 20 days available apply for leave from 21to to 31st july"
[
    {{"agent": "leave", "sub_query": "check my leave balance", "task_type": "balance_check", "condition": null}},
    {{"agent": "leave", "sub_query": "apply for leave from 21to to 31st july", "task_type": "leave_application", "condition": "leaves_left > 20"}}
]

Query: "check balance AND IF sufficient apply for leave from 25th to 30th July"
[
    {{"agent": "leave", "sub_query": "check my leave balance", "task_type": "balance_check", "condition": null}},
    {{"agent": "leave", "sub_query": "apply for leave from 25th to 30th July", "task_type": "leave_application", "condition": "leaves_left > 0"}}
]

Query: "check my overtime AND IF it's more than 10 hours request compensation"
[
    {{"agent": "attendance", "sub_query": "check my overtime", "task_type": "overtime_check", "condition": null}},
    {{"agent": "complaint", "sub_query": "request compensation", "task_type": "compensation_request", "condition": "overtime_hours > 10"}}
]

Query: "check my attendance AND IF it's below 80% file a complaint"
[
    {{"agent": "attendance", "sub_query": "check my attendance", "task_type": "attendance_check", "condition": null}},
    {{"agent": "complaint", "sub_query": "file a complaint about attendance", "task_type": "attendance_complaint", "condition": "attendance_percentage < 80"}}
]

Query: "I need leave from 25th to 30th July and a new laptop"
[
    {{"agent": "leave", "sub_query": "I need leave from 25th to 30th July", "task_type": "leave_application", "condition": null}},
    {{"agent": "complaint", "sub_query": "I need a new laptop", "task_type": "hardware_request", "condition": null}}
]

Return ONLY the JSON array. No other text.
"""
            
            response = groq_client.generate(prompt, max_tokens=400, temperature=0.1)
            logger.info(f"[Orchestrator] LLM split response: {response}")
            
            # Clean the response to extract only JSON
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
            
            # Try to parse JSON response
            try:
                import json
                tasks = json.loads(json_str)
                if isinstance(tasks, list) and len(tasks) > 0:
                    # Validate that all agents exist
                    valid_tasks = []
                    for task in tasks:
                        if task.get("agent") in self.agents:
                            valid_tasks.append(task)
                        else:
                            logger.warning(f"Agent {task.get('agent')} not found, skipping task")
                    logger.info(f"[Orchestrator] Final split tasks: {valid_tasks}")
                    return valid_tasks
                else:
                    logger.error("LLM returned empty or invalid task list")
                    return []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response was: {response}")
                return []
        except Exception as e:
            logger.error(f"Complex query parsing failed: {str(e)}")
            return []

    def _evaluate_condition(self, condition: str, prev_result: Dict[str, Any], shared_context: Dict[str, Any]) -> bool:
        """Evaluate condition using LLM for intelligent understanding of any condition"""
        try:
            if not condition:
                return True
                
            # Extract data from previous result
            data = prev_result.get("data", {})
            message = prev_result.get("message", "")
            
            # Create a comprehensive context for the LLM
            context_info = {
                "previous_result_data": data,
                "previous_result_message": message,
                "shared_context": shared_context
            }
            
            prompt = f"""
You are a condition evaluator. Analyze the condition and the previous task result to determine if the condition is met.

CONDITION: "{condition}"

PREVIOUS TASK RESULT:
- Data: {data}
- Message: {message}
- Shared Context: {shared_context}

TASK: Evaluate if the condition is met based on the previous result. Consider:
1. Leave balances (e.g., "leaves_left > 2", "sufficient balance")
2. Attendance data (e.g., "overtime > 10 hours", "attendance > 80%")
3. Complaint status (e.g., "priority is HIGH", "category is IT")
4. Any other logical conditions

EXAMPLES:
- Condition: "leaves_left > 2" + Data: {{"leaves_left": 5}} → TRUE
- Condition: "leaves_left > 2" + Data: {{"leaves_left": 1}} → FALSE
- Condition: "sufficient balance" + Message: "5 days remaining" → TRUE
- Condition: "overtime > 10" + Data: {{"overtime_hours": 15}} → TRUE
- Condition: "priority is HIGH" + Data: {{"priority": "MEDIUM"}} → FALSE

Respond with ONLY "TRUE" or "FALSE" (no other text).
"""
            
            response = groq_client.generate(prompt, max_tokens=10, temperature=0.1)
            response = response.strip().upper()
            
            if response == "TRUE":
                logger.info(f"✅ Condition '{condition}' evaluated as TRUE")
                return True
            elif response == "FALSE":
                logger.info(f"❌ Condition '{condition}' evaluated as FALSE")
                return False
            else:
                logger.warning(f"⚠️ Invalid LLM response for condition '{condition}': {response}")
                # Fallback: if we can't understand the condition, skip the task
                return False
                
        except Exception as e:
            logger.error(f"LLM condition evaluation failed: {str(e)}")
            # Fallback: if LLM fails, skip the task
            return False
    

    
    def process_request(self, emp_id: str, query: str) -> Dict[str, Any]:
        """Process request through LangGraph workflow"""
        try:
            start_time = datetime.now()
            # Check if LangGraph workflow is available
            if self.workflow_graph:
                # Build initial state as a plain dict with all required keys
                initial_state = {
                    "emp_id": emp_id,
                    "query": query,
                    "intent": "",
                    "agents_to_execute": [],
                    "agent_results": {},
                    "shared_context": {},
                    "workflow_type": "",
                    "can_proceed": True,
                    "final_response": "",
                    "error": None
                }
                logger.info(f"[LangGraph] Initial state for workflow: {initial_state}")
                # Execute LangGraph workflow
                final_state = self.workflow_graph.invoke(initial_state)
                logger.info(f"[LangGraph] Final state after workflow: {final_state}")
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                return {
                    "success": not bool(final_state.get("error")),
                    "message": final_state.get("final_response", "No response generated"),
                    "status": "completed" if not final_state.get("error") else "error",
                    "confidence": 0.9,
                    "processing_time": processing_time,
                    "agents_executed": list(final_state.get("agent_results", {}).keys()),
                    "task_type": final_state.get("workflow_type", "unknown"),
                    "error": final_state.get("error")
                }
            else:
                return {
                    "success": False,
                    "message": "LangGraph workflow is not initialized. Cannot process request.",
                    "status": "error",
                    "confidence": 0.0,
                    "processing_time": 0,
                    "agents_executed": [],
                    "task_type": "error",
                    "error": "LangGraph workflow not available"
                }
        except Exception as e:
            logger.error(f"LangGraph workflow execution failed: {str(e)}")
            return {
                "success": False,
                "message": f"❌ Execution error: {str(e)}",
                "status": "error",
                "confidence": 0.0,
                "processing_time": 0,
                "agents_executed": [],
                "task_type": "error",
                "error": str(e)
            }
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        return list(self.agents.keys())
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific agent"""
        if agent_name not in self.agents:
            return {"error": "Agent not found"}
            
        agent = self.agents[agent_name]
        return {
            "name": agent.name,
            "capabilities": agent.capabilities,
            "triggers": agent.triggers,
            "loaded": agent.agent_instance is not None,
            "config": agent.config
        }

# Global LangGraph orchestrator instance  
langgraph_orchestrator = LangGraphOrchestrator()

# API functions
def process_dynamic_request_full(emp_id: str, query: str) -> Dict[str, Any]:
    """Process request through LangGraph orchestrator and return full structured response"""
    result = langgraph_orchestrator.process_request(emp_id, query)
    return result

def get_dynamic_orchestrator_stats() -> Dict[str, Any]:
    """Get LangGraph orchestrator statistics"""
    return {
        "available_agents": len(langgraph_orchestrator.agents),
        "agent_types": langgraph_orchestrator.get_available_agents(),
        "configuration_file": langgraph_orchestrator.config_path,
        "last_initialized": datetime.now().isoformat(),
        "langgraph_enabled": langgraph_orchestrator.workflow_graph is not None
    }

logger.info("🚀 LangGraph Orchestrator Ready - Pure LangGraph Agentic Workflows Enabled") 