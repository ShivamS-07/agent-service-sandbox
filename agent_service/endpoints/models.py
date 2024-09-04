import datetime
import enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from prefect.client.schemas.objects import StateType
from pydantic import BaseModel, Field

from agent_service.io_types.citations import CitationDetailsType
from agent_service.io_types.graph import GraphOutput
from agent_service.io_types.table import TableOutput
from agent_service.io_types.text import TextOutput
from agent_service.planner.planner_types import PlanStatus, RunMetadata
from agent_service.types import Message
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.prompt_template import PromptTemplate
from agent_service.utils.scheduling import AgentSchedule
from agent_service.utils.sidebar_sections import SidebarSection


####################################################################################################
# CreateAgent
####################################################################################################
class CreateAgentRequest(BaseModel):
    is_draft: bool = False


class CreateAgentResponse(BaseModel):
    success: bool
    allow_retry: bool
    agent_id: Optional[str] = None  # only set if success is True


class UpdateAgentDraftStatusRequest(BaseModel):
    is_draft: bool = False


class UpdateAgentDraftStatusResponse(BaseModel):
    success: bool


####################################################################################################
# DeleteAgent
####################################################################################################


class TerminateAgentRequest(BaseModel):
    plan_id: Optional[str] = None
    plan_run_id: Optional[str] = None


class TerminateAgentResponse(BaseModel):
    success: bool


class DeleteAgentResponse(BaseModel):
    success: bool


class RestoreAgentResponse(BaseModel):
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
# Agent scheduling
####################################################################################################
class SetAgentScheduleRequest(BaseModel):
    agent_id: str
    user_schedule_description: str


class SetAgentScheduleResponse(BaseModel):
    agent_id: str
    schedule: AgentSchedule
    success: bool = True
    error_msg: Optional[str] = None


####################################################################################################
# GetAllAgents, GetAgent
####################################################################################################
class AgentMetadata(BaseModel):
    agent_id: str
    user_id: Optional[str]
    agent_name: str
    created_at: datetime.datetime
    last_updated: datetime.datetime
    deleted: bool = False

    last_run: Optional[datetime.datetime] = None
    next_run: Optional[datetime.datetime] = None
    output_last_updated: Optional[datetime.datetime] = None

    latest_notification_string: Optional[str] = None
    unread_notification_count: int = 0
    automation_enabled: bool = False

    schedule: Optional[AgentSchedule] = None
    cost_info: Optional[List[Dict[str, Any]]] = None
    section_id: Optional[str] = None

    is_draft: Optional[bool] = False

    def to_agent_row(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "is_draft": self.is_draft,
        }


class Section(BaseModel):
    name: str
    id: str


class GetAllAgentsResponse(BaseModel):
    sections: List[SidebarSection]
    agents: List[AgentMetadata]


####################################################################################################
# Agent Custom Notification Criteria
####################################################################################################
class CustomNotification(BaseModel):
    custom_notification_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    notification_prompt: str
    created_at: datetime.datetime = Field(default_factory=get_now_utc)
    auto_generated: bool = False


class CreateCustomNotificationRequest(BaseModel):
    agent_id: str
    notification_prompt: str


class CustomNotificationStatusResponse(BaseModel):
    custom_notification_id: str
    success: bool


####################################################################################################
# Agent Notification List
####################################################################################################
class NotificationEmailsResponse(BaseModel):
    emails: List[str]


class UpdateNotificationEmailsRequest(BaseModel):
    agent_id: str
    emails: List[str]


class UpdateNotificationEmailsResponse(BaseModel):
    success: bool
    bad_emails: Optional[List[str]]


class RemoveNotificationEmailsRequest(BaseModel):
    agent_id: str
    email: str


class RemoveNotificationEmailsResponse(BaseModel):
    success: bool


class AgentNotificationEmail(BaseModel):
    email: str
    user_id: str
    agent_id: str


class NotificationUser(BaseModel):
    user_id: str
    username: str
    name: str
    email: str


class AgentNotificationBody(BaseModel):
    summary_title: str
    summary_body: str


class AgentNotificationData(BaseModel):
    agent_name: str
    agent_id: str
    plan_run_id: str
    # the user id of the agent owner
    agent_owner: str
    notification_body: AgentNotificationBody


class AgentSubscriptionMessage(BaseModel):
    message_type: str = "agent_notification_event"
    user_ids: List[str]
    agent_data: List[AgentNotificationData]


####################################################################################################
# Agent Feedback
####################################################################################################
class SetAgentFeedBackRequest(BaseModel):
    agent_id: str
    plan_id: str
    plan_run_id: str
    output_id: str
    widget_title: str
    rating: float
    feedback_comment: Optional[str]


class SetAgentFeedBackResponse(BaseModel):
    success: bool


class AgentFeedback(BaseModel):
    agent_id: str
    plan_id: str
    plan_run_id: str
    output_id: str
    widget_title: str
    rating: float
    feedback_comment: Optional[str]
    feedback_user_id: str


class GetAgentFeedBackRequest(BaseModel):
    agent_id: str
    plan_id: str
    plan_run_id: str
    output_id: str


class GetAgentFeedBackResponse(BaseModel):
    agent_feedback: Optional[AgentFeedback]
    success: bool


####################################################################################################
# ChatWithAgent
####################################################################################################
class ChatWithAgentRequest(BaseModel):
    agent_id: str
    prompt: str
    is_first_prompt: bool = False
    # If true, DO NOT ask GPT for a response for this message, ONLY insert into the DB
    skip_agent_response: bool = False
    canned_prompt_id: Optional[str] = None


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
class Status(enum.StrEnum):
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


class PlanRunStatusInfo(BaseModel):
    # Used internally only
    plan_run_id: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    status: Optional[Status]


class TaskRunStatusInfo(BaseModel):
    # Used internally only
    task_id: str
    plan_run_id: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    status: Optional[Status]


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
    run_description: Optional[str] = None


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
    output_id: str
    plan_id: str  # which execution plan is this associated with
    plan_run_id: str  # which run is the output generated from
    is_intermediate: bool  # whether this is an intermediate output or the final output
    output: Union[TextOutput, GraphOutput, TableOutput] = Field(discriminator="output_type")
    created_at: datetime.datetime
    shared: bool = False
    # metadata from the run, currently only populated by certain db functions
    run_metadata: Optional[RunMetadata] = None
    live_plan_output: bool = False


class GetAgentOutputResponse(BaseModel):
    # it'll be only intermediate outputs OR the final outputs, sorted by time ASC
    outputs: List[AgentOutput]
    run_summary_long: Optional[str | TextOutput] = None
    run_summary_short: Optional[str] = None
    newly_updated_outputs: List[str] = Field(default_factory=list)


class GetPlanRunOutputResponse(BaseModel):
    outputs: List[AgentOutput]
    agent_name: str


####################################################################################################
# Citation Details
####################################################################################################
class GetCitationDetailsResponse(BaseModel):
    details: Optional[CitationDetailsType]


####################################################################################################
# AgentEvents
####################################################################################################
class EventType(enum.StrEnum):
    MESSAGE = "message"
    OUTPUT = "output"
    NEW_PLAN = "new_plan"
    PLAN_STATUS = "plan_status"
    EXECUTION_STATUS = "execution_status"
    TASK_STATUS = "task_status"
    TASK_LOG = "task_log"


class Event(BaseModel):
    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)


class MessageEvent(Event):
    event_type: Literal[EventType.MESSAGE] = EventType.MESSAGE
    message: Message


class OutputEvent(Event):
    event_type: Literal[EventType.OUTPUT] = EventType.OUTPUT
    output: List[AgentOutput]


class NewPlanEvent(Event):
    event_type: Literal[EventType.NEW_PLAN] = EventType.NEW_PLAN
    plan: ExecutionPlanTemplate


class PlanStatusEvent(Event):
    event_type: Literal[EventType.PLAN_STATUS] = EventType.PLAN_STATUS
    status: PlanStatus


class TaskStatus(BaseModel):
    status: Status
    task_id: str
    task_name: str
    has_output: bool
    logs: List[PlanRunTaskLog]


class TaskStatusEvent(Event):
    event_type: Literal[EventType.TASK_STATUS] = EventType.TASK_STATUS
    plan_run_id: str
    tasks: List[TaskStatus]


class ExecutionStatusEvent(Event):
    event_type: Literal[EventType.EXECUTION_STATUS] = EventType.EXECUTION_STATUS
    status: Status
    plan_run_id: str
    plan_id: str
    # Only populated if the plan is finished
    run_summary_long: Optional[str | TextOutput] = None
    run_summary_short: Optional[str] = None
    newly_updated_outputs: List[str] = Field(default_factory=list)


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


class MarkNotificationsAsUnreadRequest(BaseModel):
    agent_id: str
    message_id: str


class MarkNotificationsAsReadResponse(BaseModel):
    success: bool


class MarkNotificationsAsUnreadResponse(BaseModel):
    success: bool


class NotificationEventType(enum.StrEnum):
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
# File uploads
####################################################################################################
class UploadFileResponse(BaseModel):
    success: bool = True


####################################################################################################
# Debug info
####################################################################################################
class Tooltips(BaseModel):
    create_execution_plans: str
    run_execution_plans: str


class Debug(BaseModel):
    run_execution_plans: Dict[str, Any]
    create_execution_plans: Dict[str, Any]
    cost_info: Dict[str, Any]
    agent_owner_id: Optional[str]
    gpt_service_info: Dict[str, Any]


class GetAgentDebugInfoResponse(BaseModel):
    debug: Debug
    tooltips: Tooltips


class ToolPromptInfo(BaseModel):
    gpt_model: str

    main_prompt_name: str
    # the example prompt to GPT (if there are multiple calls, choose the most expensive one)
    main_prompt_example: str

    sys_prompt_name: str  # empty str when it's not set
    sys_prompt_example: str

    gpt_response_example: str

    num_calls: int
    total_num_input_tokens: int
    total_num_output_tokens: int
    total_cost_usd: float
    duration_seconds: float  # end time of last call - start time of first call


class PlanRunToolDebugInfo(BaseModel):
    tool_id: str  # task_id internally
    tool_name: str  # python function name
    tool_description: str  # the prompt to GPT for what this tool does
    tool_comment: str  # the short comment from GPT for what this tool does

    """
    - `arg_names`: the `args` inside execution plans that tells you the names of the arguments so
        you can know the dependencies of the tools inside the plan
    - `args`: the actual values of arguments (in serialized format) that are passed to the tool, if
        None but `arg_names` is not None then the tool hasn't been run yet
    - `output_variable_name`: the name of the output variable that the tool produces, which can
        potentially be the input to another tool
    """
    arg_names: Dict[str, Any]
    args: Optional[Dict[str, Any]]
    output_variable_name: str

    # output: Any  # `IOType` type but in JSON format
    start_time_utc: Optional[datetime.datetime]  # None if not started/ended
    end_time_utc: Optional[datetime.datetime]
    duration_seconds: Optional[float]

    prompt_infos: List[ToolPromptInfo]  # empty if no gpt calls made


class GetPlanRunDebugInfoResponse(BaseModel):
    plan_run_tools: List[PlanRunToolDebugInfo]


class GetDebugToolArgsResponse(BaseModel):
    args: Dict[str, Any]


class GetDebugToolResultResponse(BaseModel):
    result: Any


class ToolMetadata(BaseModel):
    tool_name: str  # python function name
    tool_description: str  # the prompt to GPT for what this tool does
    tool_header: str  # function header, e.g. `def my_tool(x: int) -> int:`
    category: str  # `ToolCategory`


class GetToolLibraryResponse(BaseModel):
    tool_category_map: Dict[str, str]  # name to description
    tools: List[ToolMetadata]


####################################################################################################
# Memory Item
####################################################################################################


class MemoryItem(BaseModel):
    id: str
    name: str
    type: str
    time_created: datetime.datetime
    time_updated: datetime.datetime
    # TODO: add field - need to grab from db
    # ownerId: str


class ListMemoryItemsResponse(BaseModel):
    success: bool
    items: List[MemoryItem]


class GetAutocompleteItemsRequest(BaseModel):
    text: str


class GetAutocompleteItemsResponse(BaseModel):
    success: bool
    items: List[MemoryItem]


class GetMemoryContentResponse(BaseModel):
    output: Optional[Union[TextOutput, TableOutput]] = Field(discriminator="output_type")


class DeleteMemoryResponse(BaseModel):
    success: bool


class RenameMemoryRequest(BaseModel):
    id: str
    type: str
    new_name: str


class RenameMemoryResponse(BaseModel):
    success: bool


####################################################################################################
# User
####################################################################################################


class Account(BaseModel):
    user_id: str
    email: str
    username: str
    name: str


class UpdateUserRequest(BaseModel):
    name: str
    username: str
    email: str


class UpdateUserResponse(BaseModel):
    success: bool


class GetAccountInfoResponse(BaseModel):
    account: Account


class GetTeamAccountsResponse(BaseModel):
    accounts: List[Account]


# Regression Test Run Info
####################################################################################################
class GetTestSuiteRunInfoResponse(BaseModel):
    test_suite_run_info: Dict[str, Any]


class GetTestSuiteRunsResponse(BaseModel):
    test_suite_runs: List[Dict[str, Any]]


class GetTestCaseInfoResponse(BaseModel):
    test_case_info: Dict[str, Any]


class GetTestCasesResponse(BaseModel):
    test_cases: List[Dict[str, Any]]


class CannedPrompt(BaseModel):
    id: str
    prompt: str


class GetCannedPromptsResponse(BaseModel):
    canned_prompts: List[CannedPrompt]


####################################################################################################
# Document conversion
####################################################################################################
class MediaType(enum.StrEnum):
    DOCX = "docx"
    TXT = "txt"


class ConvertMarkdownRequest(BaseModel):
    content: str
    format: MediaType


####################################################################################################
# Sidebar Organization
####################################################################################################


class CreateSectionRequest(BaseModel):
    name: str


class CreateSectionResponse(BaseModel):
    section_id: str


class DeleteSectionRequest(BaseModel):
    section_id: str


class DeleteSectionResponse(BaseModel):
    success: bool


class RenameSectionRequest(BaseModel):
    section_id: str
    new_name: str


class RenameSectionResponse(BaseModel):
    success: bool


class SetAgentSectionRequest(BaseModel):
    new_section_id: Optional[str]
    agent_id: str


class SetAgentSectionResponse(BaseModel):
    success: bool


class RearrangeSectionRequest(BaseModel):
    new_index: int
    section_id: str


class RearrangeSectionResponse(BaseModel):
    success: bool


####################################################################################################
# Prompt Templates
####################################################################################################
class GetPromptTemplatesResponse(BaseModel):
    prompt_templates: List[PromptTemplate]
