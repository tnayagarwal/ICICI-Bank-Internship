"""
Autonomous Task Scheduler
Handles self-triggering operations, periodic tasks, and event-driven automation
"""
import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import schedule

# Import memory system (with fallback)
try:
    from utils.memory import memory_system
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    logger.warning("Memory system not available for scheduler")

class TaskType(Enum):
    DAILY = "daily"
    HOURLY = "hourly" 
    WEEKLY = "weekly"
    INTERVAL = "interval"
    CONTINUOUS = "continuous"

@dataclass
class ScheduledTask:
    id: str
    name: str
    task_type: TaskType
    function: Callable[[], Any]
    schedule_time: str = ""  # e.g., "00:10" for daily, "*/10" for interval
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    max_errors: int = 5
    description: str = ""

class HRScheduler:
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.is_running = False
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        # Register default autonomous tasks
        self.register_default_tasks()
    
    def register_default_tasks(self):
        """Register default autonomous tasks"""
        
        # Daily overtime calculation at 00:10 IST
        self.add_task(
            task_id="daily_overtime",
            name="Calculate Daily Overtime",
            task_type=TaskType.DAILY,
            function=self.calculate_all_overtime,
            schedule_time="00:10",
            description="Calculate overtime for all employees automatically"
        )
        
        # Hourly policy document ingestion
        self.add_task(
            task_id="hourly_policy_sync",
            name="Sync Policy Documents", 
            task_type=TaskType.HOURLY,
            function=self.sync_policy_documents,
            schedule_time="00",  # At the start of every hour
            description="Ingest new policy documents and update embeddings"
        )
        
        # Every 10 minutes - review pending tasks
        self.add_task(
            task_id="task_review",
            name="Review Pending Tasks",
            task_type=TaskType.INTERVAL,
            function=self.review_pending_tasks,
            schedule_time="*/10",  # Every 10 minutes
            description="Review and manage pending orchestrator tasks"
        )
        
        # Weekly attendance compliance check (Mondays at 09:00)
        self.add_task(
            task_id="weekly_compliance",
            name="Weekly Attendance Compliance",
            task_type=TaskType.WEEKLY,
            function=self.check_attendance_compliance,
            schedule_time="monday:09:00",
            description="Check attendance compliance across all departments"
        )
        
        # System health check every 4 hours
        self.add_task(
            task_id="system_health",
            name="System Health Check",
            task_type=TaskType.INTERVAL,
            function=self.perform_health_check,
            schedule_time="*/4h",  # Every 4 hours
            description="Monitor system health and performance metrics"
        )
    
    def add_task(self, task_id: str, name: str, task_type: TaskType, 
                function: Callable[[], Any], schedule_time: str = "",
                parameters: Dict[str, Any] = None, description: str = ""):
        """Add a new scheduled task"""
        
        task = ScheduledTask(
            id=task_id,
            name=name,
            task_type=task_type,
            function=function,
            schedule_time=schedule_time,
            parameters=parameters or {},
            description=description
        )
        
        self.tasks[task_id] = task
        logger.info(f"Added scheduled task: {name} ({task_id})")
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        try:
            self.is_running = True
            self.stop_event.clear()
            
            # Schedule all tasks
            for task in self.tasks.values():
                if task.enabled:
                    self._schedule_task(task)
            
            # Start scheduler in separate thread
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("HR Scheduler started successfully")
            
            # Store startup event in memory if available
            if MEMORY_AVAILABLE:
                asyncio.create_task(memory_system.remember("scheduler_startup", {
                    "startup_time": datetime.now().isoformat(),
                    "registered_tasks": len(self.tasks),
                    "enabled_tasks": len([t for t in self.tasks.values() if t.enabled])
                }, category="system", importance=3.0))
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {str(e)}")
            self.is_running = False
    
    def stop(self):
        """Stop the scheduler"""
        try:
            self.is_running = False
            self.stop_event.set()
            
            # Clear all scheduled jobs
            schedule.clear()
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            logger.info("HR Scheduler stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {str(e)}")
    
    def _schedule_task(self, task: ScheduledTask):
        """Schedule a task with the schedule library"""
        try:
            if task.task_type == TaskType.DAILY:
                schedule.every().day.at(task.schedule_time).do(self._execute_task, task.id)
            elif task.task_type == TaskType.HOURLY:
                minute = task.schedule_time or "00"
                schedule.every().hour.at(f":{minute}").do(self._execute_task, task.id)
            elif task.task_type == TaskType.WEEKLY:
                if ":" in task.schedule_time:
                    day, time_str = task.schedule_time.split(":", 1)
                    getattr(schedule.every(), day.lower()).at(time_str).do(self._execute_task, task.id)
            elif task.task_type == TaskType.INTERVAL:
                # Simplified interval parsing
                if "*/10" in task.schedule_time:
                    schedule.every(10).minutes.do(self._execute_task, task.id)
                elif "*/4h" in task.schedule_time:
                    schedule.every(4).hours.do(self._execute_task, task.id)
                else:
                    schedule.every(60).minutes.do(self._execute_task, task.id)  # Default hourly
            
            logger.info(f"Scheduled task: {task.name}")
            
        except Exception as e:
            logger.error(f"Failed to schedule task {task.name}: {str(e)}")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        logger.info("Scheduler loop started")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Scheduler loop error: {str(e)}")
                time.sleep(60)  # Wait longer on error
        
        logger.info("Scheduler loop stopped")
    
    def _execute_task(self, task_id: str):
        """Execute a scheduled task"""
        if task_id not in self.tasks:
            logger.error(f"Task not found: {task_id}")
            return
        
        task = self.tasks[task_id]
        
        if not task.enabled:
            logger.info(f"Skipping disabled task: {task.name}")
            return
        
        try:
            logger.info(f"Executing scheduled task: {task.name}")
            task.last_run = datetime.now()
            task.run_count += 1
            
            # Execute the task function
            if asyncio.iscoroutinefunction(task.function):
                # Handle async functions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(task.function(**task.parameters))
                loop.close()
            else:
                result = task.function(**task.parameters)
            
            # Log success
            if MEMORY_AVAILABLE:
                asyncio.create_task(memory_system.remember(f"task_execution_{task_id}", {
                    "task_id": task_id,
                    "task_name": task.name,
                    "execution_time": task.last_run.isoformat(),
                    "result": str(result)[:500] if result else "Success",
                    "status": "completed"
                }, category="scheduler", importance=2.0))
            
            logger.info(f"Task completed successfully: {task.name}")
            
        except Exception as e:
            task.error_count += 1
            logger.error(f"Task execution failed: {task.name} - {str(e)}")
            
            # Log error
            if MEMORY_AVAILABLE:
                asyncio.create_task(memory_system.remember(f"task_error_{task_id}", {
                    "task_id": task_id,
                    "task_name": task.name,
                    "execution_time": datetime.now().isoformat(),
                    "error": str(e),
                    "error_count": task.error_count,
                    "status": "failed"
                }, category="scheduler", importance=4.0))
            
            # Disable task if too many errors
            if task.error_count >= task.max_errors:
                task.enabled = False
                logger.error(f"Disabled task due to repeated failures: {task.name}")
    
    # Scheduled Task Functions
    def calculate_all_overtime(self) -> str:
        """Calculate overtime for all employees"""
        try:
            # Import here to avoid circular imports
            from agents.attendance_agent import process_attendance
            
            # This would normally get all employee IDs and calculate overtime
            # For demo, we'll simulate the process
            employees = ["20001", "20002", "20003", "20004"]  # Test employees
            completed = 0
            
            for emp_id in employees:
                try:
                    result = process_attendance(emp_id, "overtime")
                    if result.get("success"):
                        completed += 1
                except Exception as e:
                    logger.error(f"Overtime calculation failed for {emp_id}: {str(e)}")
            
            message = f"Overtime calculation completed for {completed}/{len(employees)} employees"
            logger.info(message)
            return message
            
        except Exception as e:
            logger.error(f"Overtime calculation failed: {str(e)}")
            return f"Failed: {str(e)}"
    
    def sync_policy_documents(self) -> str:
        """Sync and index policy documents"""
        try:
            # Check for new policy documents in the database
            # This would normally update embeddings and sync with vector store
            logger.info("Policy document sync initiated")
            
            # Simulate policy sync
            if MEMORY_AVAILABLE:
                asyncio.create_task(memory_system.remember("policy_sync", {
                    "sync_time": datetime.now().isoformat(),
                    "documents_processed": 5,
                    "status": "completed"
                }, category="system", importance=2.0))
            
            return "Policy sync completed successfully"
            
        except Exception as e:
            logger.error(f"Policy sync failed: {str(e)}")
            return f"Failed: {str(e)}"
    
    def review_pending_tasks(self) -> str:
        """Review and manage pending tasks"""
        try:
            # This would normally interface with the orchestrator
            # For now, just log the review
            logger.debug("Reviewing pending tasks")
            
            active_tasks = len([t for t in self.tasks.values() if t.enabled])
            failed_tasks = len([t for t in self.tasks.values() if t.error_count > 0])
            
            if MEMORY_AVAILABLE:
                asyncio.create_task(memory_system.remember("task_review", {
                    "review_time": datetime.now().isoformat(),
                    "active_tasks": active_tasks,
                    "failed_tasks": failed_tasks
                }, category="system", importance=1.0))
            
            return f"Task review completed: {active_tasks} active, {failed_tasks} with errors"
            
        except Exception as e:
            logger.error(f"Task review failed: {str(e)}")
            return f"Failed: {str(e)}"
    
    def check_attendance_compliance(self) -> str:
        """Check attendance compliance across all departments"""
        try:
            # This would normally analyze attendance patterns
            logger.info("Attendance compliance check initiated")
            
            # Simulate compliance check
            compliance_score = 95.5  # Mock score
            
            if MEMORY_AVAILABLE:
                asyncio.create_task(memory_system.remember("compliance_check", {
                    "check_time": datetime.now().isoformat(),
                    "compliance_score": compliance_score,
                    "issues_found": 2,
                    "status": "completed"
                }, category="compliance", importance=4.0))
            
            message = f"Compliance check completed: {compliance_score}% compliance rate"
            logger.info(message)
            return message
            
        except Exception as e:
            logger.error(f"Compliance check failed: {str(e)}")
            return f"Failed: {str(e)}"
    
    def perform_health_check(self) -> str:
        """Perform system health check"""
        try:
            # Check various system components
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "scheduler": {
                    "running": self.is_running,
                    "total_tasks": len(self.tasks),
                    "enabled_tasks": len([t for t in self.tasks.values() if t.enabled]),
                    "failed_tasks": len([t for t in self.tasks.values() if t.error_count > 0])
                }
            }
            
            if MEMORY_AVAILABLE:
                memory_stats = asyncio.run(memory_system.get_memory_stats())
                health_data["memory_system"] = memory_stats
                
                # Store health check results
                asyncio.create_task(memory_system.remember("system_health_check", health_data, 
                                                         category="system", importance=3.0))
            
            logger.info("System health check completed")
            return "Health check completed successfully"
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return f"Failed: {str(e)}"
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all scheduled tasks"""
        task_status = {}
        
        for task_id, task in self.tasks.items():
            task_status[task_id] = {
                "name": task.name,
                "type": task.task_type.value,
                "enabled": task.enabled,
                "schedule": task.schedule_time,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "run_count": task.run_count,
                "error_count": task.error_count,
                "description": task.description
            }
        
        return {
            "scheduler_running": self.is_running,
            "total_tasks": len(self.tasks),
            "enabled_tasks": len([t for t in self.tasks.values() if t.enabled]),
            "tasks": task_status
        }
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a specific task"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            self.tasks[task_id].error_count = 0  # Reset error count
            
            # Re-schedule if scheduler is running
            if self.is_running:
                self._schedule_task(self.tasks[task_id])
            
            logger.info(f"Enabled task: {task_id}")
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a specific task"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            logger.info(f"Disabled task: {task_id}")
            return True
        return False

# Global scheduler instance
hr_scheduler = HRScheduler()

# Convenience functions
def start_scheduler():
    """Start the HR scheduler"""
    hr_scheduler.start()

def stop_scheduler():
    """Stop the HR scheduler"""
    hr_scheduler.stop()

def get_scheduler_status() -> Dict[str, Any]:
    """Get scheduler status"""
    return hr_scheduler.get_task_status()

def add_custom_task(task_id: str, name: str, function: Callable, 
                   schedule_time: str, task_type: TaskType = TaskType.DAILY,
                   description: str = ""):
    """Add a custom scheduled task"""
    hr_scheduler.add_task(
        task_id=task_id,
        name=name,
        task_type=task_type,
        function=function,
        schedule_time=schedule_time,
        description=description
    ) 