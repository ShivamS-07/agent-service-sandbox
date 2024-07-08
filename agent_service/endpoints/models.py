import datetime
import enum
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from prefect.client.schemas.objects import StateType
from pydantic import BaseModel, Field

from agent_service.io_types.graph import GraphOutput
from agent_service.io_types.table import TableOutput
from agent_service.io_types.text import TextOutput
from agent_service.planner.planner_types import PlanStatus
from agent_service.types import Message
from agent_service.utils.date_utils import get_now_utc


####################################################################################################
# CreateAgent
####################################################################################################
class CreateAgentResponse(BaseModel):
    success: bool
    allow_retry: bool
    agent_id: Optional[str] = None  # only set if success is True


####################################################################################################
# DeleteAgent
####################################################################################################


class DeleteAgentResponse(BaseModel):
    success: bool


####################################################################################################
# UpdateAgent
####################################################################################################
class UpdateAgentRequest(BaseModel):
    agent_name: str


class UpdateAgentResponse(BaseModel):
    success: bool


####################################################################################################
# Agent automation enabled flag
####################################################################################################
class EnableAgentAutomationRequest(BaseModel):
    agent_id: str


class EnableAgentAutomationResponse(BaseModel):
    success: bool
    next_run: Optional[datetime.datetime] = None


class DisableAgentAutomationRequest(BaseModel):
    agent_id: str


class DisableAgentAutomationResponse(BaseModel):
    success: bool


####################################################################################################
# GetAllAgents
####################################################################################################
class AgentMetadata(BaseModel):
    agent_id: str
    user_id: Optional[str]
    agent_name: str
    created_at: datetime.datetime
    last_updated: datetime.datetime

    last_run: Optional[datetime.datetime] = None
    next_run: Optional[datetime.datetime] = None

    latest_notification_string: Optional[str] = None
    unread_notification_count: int = 0
    automation_enabled: bool = False

    def to_agent_row(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }


class GetAllAgentsResponse(BaseModel):
    agents: List[AgentMetadata]


####################################################################################################
# ChatWithAgent
####################################################################################################
class ChatWithAgentRequest(BaseModel):
    agent_id: str
    prompt: str
    is_first_prompt: bool = False


class ChatWithAgentResponse(BaseModel):
    success: bool
    allow_retry: bool
    name: Optional[str] = None


####################################################################################################
# GetChatHistory
####################################################################################################
class GetChatHistoryResponse(BaseModel):
    messages: List[Message]  # sorted by message_time ASC


####################################################################################################
# GetAgentWorklogBoard
####################################################################################################
class Status(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"

    @classmethod
    def from_prefect_state(cls, prefect_state: Optional[StateType]) -> "Status":
        if prefect_state == StateType.RUNNING:
            return cls.RUNNING
        elif prefect_state == StateType.COMPLETED:
            return cls.COMPLETE
        elif prefect_state in (StateType.FAILED, StateType.CRASHED):
            return cls.ERROR
        elif prefect_state in (StateType.CANCELLING, StateType.CANCELLED):
            return cls.CANCELLED
        else:
            return cls.NOT_STARTED


class PlanRunTaskLog(BaseModel):
    """Referring to a single log item under a task. A task can have multiple logs."""

    log_id: str
    log_message: str  # user-faced log item under each task
    created_at: datetime.datetime
    has_output: bool = False


class PlanRunTask(BaseModel):
    """Referring to a **started** task within a Prefect run. A task can have multiple worklogs."""

    task_id: str
    task_name: str
    status: Status
    start_time: Optional[datetime.datetime]  # if None then it's not started yet
    end_time: Optional[datetime.datetime]
    logs: List[PlanRunTaskLog]  # sorted by start_time ASC, could be empty
    has_output: bool = True


class PlanRun(BaseModel):
    """Referring to a **started** Prefect run which can have multiple tasks."""

    plan_run_id: str  # the actual run ID from Prefect
    status: Status
    plan_id: str  # which execution plan is this associated with
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime]
    tasks: List[PlanRunTask]  # sorted by start_time ASC
    shared: bool = False


class PlanTemplateTask(BaseModel):
    task_id: str
    task_name: str


class ExecutionPlanTemplate(BaseModel):
    """A template for the next run including the tasks that will be run."""

    plan_id: str
    upcoming_plan_run_id: Optional[str] = None
    tasks: List[PlanTemplateTask]


class GetAgentWorklogBoardResponse(BaseModel):
    run_history: List[PlanRun]  # sorted by time ASC, all the runs that have happened
    execution_plan_template: Optional[ExecutionPlanTemplate] = None
    latest_plan_status: Optional[str] = None


####################################################################################################
# GetAgentTaskOutput
####################################################################################################
class GetAgentTaskOutputResponse(BaseModel):
    output: Optional[Union[TextOutput, GraphOutput, TableOutput]] = Field(
        discriminator="output_type"
    )


####################################################################################################
# GetAgentOutput
####################################################################################################
class AgentOutput(BaseModel):
    agent_id: str
    plan_id: str  # which execution plan is this associated with
    plan_run_id: str  # which run is the output generated from
    is_intermediate: bool  # whether this is an intermediate output or the final output
    output: Union[TextOutput, GraphOutput, TableOutput] = Field(discriminator="output_type")
    created_at: datetime.datetime
    shared: bool = False


class GetAgentOutputResponse(BaseModel):
    # it'll be only intermediate outputs OR the final outputs, sorted by time ASC
    outputs: List[AgentOutput]


class GetPlanRunOutputResponse(BaseModel):
    outputs: List[AgentOutput]
    agent_name: str


####################################################################################################
# AgentEvents
####################################################################################################
class EventType(str, enum.Enum):
    MESSAGE = "message"
    OUTPUT = "output"
    NEW_PLAN = "new_plan"
    PLAN_STATUS = "plan_status"
    EXECUTION_STATUS = "execution_status"
    TASK_STATUS = "task_status"
    TASK_LOG = "task_log"


class MessageEvent(BaseModel):
    event_type: Literal[EventType.MESSAGE] = EventType.MESSAGE
    message: Message


class OutputEvent(BaseModel):
    event_type: Literal[EventType.OUTPUT] = EventType.OUTPUT
    output: List[AgentOutput]


class NewPlanEvent(BaseModel):
    event_type: Literal[EventType.NEW_PLAN] = EventType.NEW_PLAN
    plan: ExecutionPlanTemplate


class PlanStatusEvent(BaseModel):
    event_type: Literal[EventType.PLAN_STATUS] = EventType.PLAN_STATUS
    status: PlanStatus


class TaskStatus(BaseModel):
    status: Status
    task_id: str
    task_name: str
    has_output: bool
    logs: List[PlanRunTaskLog]


class TaskStatusEvent(BaseModel):
    event_type: Literal[EventType.TASK_STATUS] = EventType.TASK_STATUS
    plan_run_id: str
    tasks: List[TaskStatus]


class ExecutionStatusEvent(BaseModel):
    event_type: Literal[EventType.EXECUTION_STATUS] = EventType.EXECUTION_STATUS
    status: Status
    plan_run_id: str
    plan_id: str


class AgentEvent(BaseModel):
    agent_id: str
    event: Union[
        MessageEvent,
        OutputEvent,
        NewPlanEvent,
        PlanStatusEvent,
        TaskStatusEvent,
        ExecutionStatusEvent,
    ] = Field(discriminator="event_type")
    timestamp: datetime.datetime = Field(default_factory=get_now_utc)


####################################################################################################
# Plan Run Sharing
####################################################################################################
class SharePlanRunRequest(BaseModel):
    plan_run_id: str


class SharePlanRunResponse(BaseModel):
    success: bool


class UnsharePlanRunRequest(BaseModel):
    plan_run_id: str


class UnsharePlanRunResponse(BaseModel):
    success: bool


####################################################################################################
# Notifications
####################################################################################################


class MarkNotificationsAsReadRequest(BaseModel):
    agent_id: str
    timestamp: Optional[datetime.datetime] = None


class MarkNotificationsAsReadResponse(BaseModel):
    success: bool


class NotificationEventType(str, enum.Enum):
    NOTIFY_MESSAGE = "notify_message"


class NotifyMessageEvent(BaseModel):
    event_type: Literal[NotificationEventType.NOTIFY_MESSAGE] = NotificationEventType.NOTIFY_MESSAGE
    agent_id: str
    unread_count: int
    latest_notification_string: Optional[str] = None


class NotificationEvent(BaseModel):
    user_id: str
    event: Union[NotifyMessageEvent] = Field(discriminator="event_type")
    timestamp: datetime.datetime = Field(default_factory=get_now_utc)


####################################################################################################
# Feature Flags / LD
####################################################################################################


class GetSecureUserResponse(BaseModel):
    hash: str
    context: Dict[str, Any]


####################################################################################################
# Debug info
####################################################################################################
class Tooltips(BaseModel):
    plan_selections: str
    all_generated_plans: str
    worker_sqs_log: str
    tool_calls: str


class Debug(BaseModel):
    agent_owner_id: Optional[str]
    plan_selections: List[Dict[str, Any]]
    all_generated_plans: List[Dict[str, Any]]
    worker_sqs_log: Dict[str, Any]
    tool_calls: Dict[str, Any]


class GetAgentDebugInfoResponse(BaseModel):
    tooltips: Tooltips
    debug: Debug
