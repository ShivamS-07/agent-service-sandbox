import datetime
import enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_service.io_types.graph import GraphOutput, GraphType
from agent_service.io_types.table import TableOutput, TableTransformation
from agent_service.io_types.text import TextOutput
from agent_service.planner.planner_types import ExecutionPlan, PlanStatus, RunMetadata
from agent_service.types import MemoryType, Message
from agent_service.utils.date_utils import get_now_utc
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


class DeleteAgentsRequest(BaseModel):
    agent_ids: List[str]


class DeleteAgentsResponse(BaseModel):
    success: bool


####################################################################################################
# UpdateAgent
####################################################################################################
class UpdateAgentRequest(BaseModel):
    agent_name: str


class UpdateAgentResponse(BaseModel):
    success: bool


####################################################################################################
# Agent Help
####################################################################################################
class AgentHelpRequest(BaseModel):
    is_help_requested: bool
    send_chat_message: bool = False
    resolved_by_cs: bool = False


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
# Agent user settings
####################################################################################################


class AgentUserSettingsSetRequest(BaseModel):
    include_web_results: Optional[bool] = None


####################################################################################################
# GetAllAgents, GetAgent
####################################################################################################
class AgentMetadata(BaseModel):
    created_from_template: bool = False
    copied_from_agent_id: Optional[str] = None
    created_while_spoofed: bool = False
    # Populated if the agent was created while spoofed. This is the user ID of
    # the "real" user, not the spoofed user.
    real_user_id: Optional[str] = None


class AgentInfo(BaseModel):
    agent_id: str
    user_id: Optional[str]
    agent_name: str
    created_at: datetime.datetime
    last_updated: datetime.datetime
    deleted: bool = False

    last_run: Optional[datetime.datetime] = None
    next_run: Optional[datetime.datetime] = None
    output_last_updated: Optional[datetime.datetime] = None
    last_automation_run: Optional[datetime.datetime] = None

    latest_notification_string: Optional[str] = None
    unread_notification_count: int = 0
    automation_enabled: bool = False

    schedule: Optional[AgentSchedule] = None
    cost_info: Optional[List[Dict[str, Any]]] = None
    section_id: Optional[str] = None

    is_draft: Optional[bool] = False
    agent_metadata: Optional[AgentMetadata] = None
    help_requested: bool = False

    def to_agent_row(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "is_draft": self.is_draft,
            "agent_metadata": (
                self.agent_metadata.model_dump_json()
                if self.agent_metadata
                else AgentMetadata().model_dump_json()
            ),
        }


class Section(BaseModel):
    name: str
    id: str


class GetAllAgentsResponse(BaseModel):
    sections: List[SidebarSection]
    agents: List[AgentInfo]


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
    notification_key: str


class RemoveNotificationEmailsResponse(BaseModel):
    success: bool


class NotificationUser(BaseModel):
    user_id: str
    username: str
    name: str
    email: str


####################################################################################################
# Agent Email
####################################################################################################


class AgentNotificationBody(BaseModel):
    summary_title: str
    summary_body: str


class AgentNotificationData(BaseModel):
    agent_name: str
    agent_id: str
    email_subject: str
    plan_run_id: str
    # the user id of the agent owner
    agent_owner: str
    notification_body: AgentNotificationBody


class AgentSubscriptionMessage(BaseModel):
    message_type: str = "agent_notification_event"
    user_id_email_pairs: List[tuple[str, str]]
    agent_data: List[AgentNotificationData]


class AgentNotificationEmail(BaseModel):
    email: str
    user_id: str | None
    agent_id: str


class OnboardingEmailMessage(BaseModel):
    message_type: str = "onboarding_email_event"
    user_name: str
    user_id: str
    email: str
    hubspot_email_id: int


class ForwardingEmailMessage(BaseModel):
    message_type: str = "forwarding_email_event"
    notification_key: str
    recipient_email: str


class PlanRunFinishEmailMessage(BaseModel):
    message_type: str = "plan_run_finish_email_event"
    agent_id: str
    agent_owner: str
    agent_name: str
    short_summary: str
    output_titles: str


class HelpRequestResolvedEmailMessage(BaseModel):
    message_type: str = "help_request_resolved_email_event"
    agent_id: str
    agent_owner: str
    agent_name: str
    short_summary: str
    output_titles: str


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


class GetUnreadUpdatesSummaryResponse(BaseModel):
    summary_mmessage: str


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
    total_message_count: Optional[int] = None
    start_index: Optional[int] = None


####################################################################################################
# OutputPreview
####################################################################################################
class OutputType(enum.StrEnum):
    TEXT = "text"
    TABLE = "table"
    LINE_GRAPH = GraphType.LINE
    PIE_GRAPH = GraphType.PIE
    BAR_GRAPH = GraphType.BAR


class OutputPreview(BaseModel):
    title: str
    output_type: OutputType


####################################################################################################
# GetAgentWorklogBoard
####################################################################################################
class Status(enum.StrEnum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"
    NO_RESULTS_FOUND = "NO_RESULTS_FOUND"


class PlanRunInfo(BaseModel):
    agent_id: str
    plan_id: str
    plan_run_id: str
    created_at: datetime.datetime
    is_shared: bool = False
    status: Status = Status.COMPLETE


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
    percentage_complete: Optional[float] = None


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
    preview: Optional[List[OutputPreview]] = None


class PlanTemplateTask(BaseModel):
    task_id: str
    task_name: str


class ExecutionPlanTemplate(BaseModel):
    """A template for the next run including the tasks that will be run."""

    plan_id: str
    upcoming_plan_run_id: Optional[str] = None
    tasks: List[PlanTemplateTask]
    preview: Optional[List[OutputPreview]] = None


class GetAgentWorklogBoardResponse(BaseModel):
    run_history: List[PlanRun]  # sorted by time ASC, all the runs that have happened
    total_plan_count: Optional[int] = None
    start_index: Optional[int] = None
    execution_plan_template: Optional[ExecutionPlanTemplate] = None
    latest_plan_status: Optional[str] = None


class QuickThoughts(BaseModel):
    summary: TextOutput


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
    task_id: Optional[str] = None  # None for backward compatibility
    dependent_task_ids: List[str] = Field(default_factory=list)
    parent_task_ids: List[str] = Field(default_factory=list)
    is_locked: bool = False

    transformation_id: Optional[str] = None
    is_transformation_local: Optional[bool] = None


class GetAgentOutputResponse(BaseModel):
    # it'll be only intermediate outputs OR the final outputs, sorted by time ASC
    outputs: List[AgentOutput]
    run_summary_long: Optional[str | TextOutput] = None
    run_summary_short: Optional[str] = None
    newly_updated_outputs: List[str] = Field(default_factory=list)
    quick_thoughts: Optional[QuickThoughts] = None


class GetPlanRunOutputResponse(BaseModel):
    outputs: List[AgentOutput]
    agent_name: str


class DeleteAgentOutputRequest(BaseModel):
    plan_id: str
    output_ids: List[str]
    task_ids: List[str]


class DeleteAgentOutputResponse(BaseModel):
    success: bool = True


class LockAgentOutputRequest(BaseModel):
    plan_id: str
    output_ids: List[str]
    task_ids: List[str]


class LockAgentOutputResponse(BaseModel):
    success: bool = True


class UnlockAgentOutputRequest(BaseModel):
    plan_id: str
    output_ids: List[str]
    task_ids: List[str]


class UnlockAgentOutputResponse(BaseModel):
    success: bool = True


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
    AGENT_NAME = "agent_name"
    QUICK_THOUGHTS = "quick_thoughts"
    TASK_LOG_PROGRESS_BAR = "task_log_progress_bar"


class Event(BaseModel):
    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)


class MessageEvent(Event):
    event_type: Literal[EventType.MESSAGE] = EventType.MESSAGE
    message: Message


class AgentNameEvent(Event):
    event_type: Literal[EventType.AGENT_NAME] = EventType.AGENT_NAME
    agent_name: str


class OutputEvent(Event):
    event_type: Literal[EventType.OUTPUT] = EventType.OUTPUT
    output: List[AgentOutput]


class AgentQuickThoughtsEvent(Event):
    event_type: Literal[EventType.QUICK_THOUGHTS] = EventType.QUICK_THOUGHTS
    quick_thoughts: QuickThoughts


class TaskLogProgressBarEvent(Event):
    event_type: Literal[EventType.TASK_LOG_PROGRESS_BAR] = EventType.TASK_LOG_PROGRESS_BAR
    percentage: float
    log_id: Optional[str]
    created_at: datetime.datetime
    log_message: Optional[str]
    task_id: Optional[str]


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
        AgentNameEvent,
        AgentQuickThoughtsEvent,
        TaskLogProgressBarEvent,
    ] = Field(discriminator="event_type")
    timestamp: datetime.datetime = Field(default_factory=get_now_utc)


####################################################################################################
# UpdateAgentWidgetName
####################################################################################################
class UpdateAgentWidgetNameRequest(BaseModel):
    output_id: str
    new_widget_title: str


class UpdateAgentWidgetNameResponse(BaseModel):
    success: bool


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
    NOTIFY_HELP_STATUS = "notify_help_status"


class NotifyMessageEvent(BaseModel):
    event_type: Literal[NotificationEventType.NOTIFY_MESSAGE] = NotificationEventType.NOTIFY_MESSAGE
    agent_id: str
    unread_count: int
    latest_notification_string: Optional[str] = None


class NotifyHelpStatusEvent(BaseModel):
    event_type: Literal[NotificationEventType.NOTIFY_HELP_STATUS] = (
        NotificationEventType.NOTIFY_HELP_STATUS
    )
    agent_id: str
    is_help_requested: bool


class NotificationEvent(BaseModel):
    user_id: str
    event: Union[NotifyMessageEvent, NotifyHelpStatusEvent] = Field(discriminator="event_type")
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


class GetDebugToolArgsResponse(BaseModel):
    args: Dict[str, Any]


class GetDebugToolResultResponse(BaseModel):
    result: Any


####################################################################################################
# View Plan Tools
####################################################################################################
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


class ToolArgInfo(BaseModel):
    arg_name: str
    arg_value: Any
    arg_type_name: str
    required: bool
    is_editable: bool  # type of edited value should be consistent with `arg_type_name`
    # empty if no dependency. {'task_id': ..., 'task_name': ...}
    tasks_to_depend_on: List[Dict[str, str]]


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
    arg_list: List[ToolArgInfo]
    output_variable_name: str

    # output: Any  # `IOType` type but in JSON format
    start_time_utc: Optional[datetime.datetime]  # None if not started/ended
    end_time_utc: Optional[datetime.datetime]
    duration_seconds: Optional[float]

    prompt_infos: List[ToolPromptInfo]  # empty if no gpt calls made


class GetPlanRunDebugInfoResponse(BaseModel):
    plan_run_tools: List[PlanRunToolDebugInfo]


####################################################################################################
# Modify Tool Args
####################################################################################################
class ArgToModify(BaseModel):
    tool_id: str
    arg_name: str
    arg_value: Any  # BE validate if the type is correct


class ModifyPlanRunArgsRequest(BaseModel):
    args_to_modify: List[ArgToModify]

    def __str__(self) -> str:
        outputs = [
            f"{arg.tool_id=}, {arg.arg_name=}, {arg.arg_value=}" for arg in self.args_to_modify
        ]
        return "\n".join(outputs)


class ValidateArgError(BaseModel):
    tool_id: str
    arg_name: str
    error: str

    def __str__(self) -> str:
        return f"{self.tool_id=}, {self.arg_name=}, {self.error=}"


class ModifyPlanRunArgsResponse(BaseModel):
    errors: List[ValidateArgError] = Field(default_factory=list)


####################################################################################################
# Tool Library
####################################################################################################
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
    memory_type: Optional[MemoryType] = None


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
# Variables and Data
####################################################################################################
class AvailableVariable(BaseModel):
    """
    This is only a subset of all available fields in the metadata - returned for view only.
    """

    id: str
    name: str
    description: str
    alternate_names: List[str]
    is_global: bool
    is_preset: bool
    hierarchical_category_id: str


class GetAvailableVariablesResponse(BaseModel):
    variables: List[AvailableVariable]


class GetVariableCoverageRequest(BaseModel):
    """
    Request model for getting variable coverage
    """

    universe_id: Optional[str] = None
    feature_ids: Optional[List[str]] = None


class GetVariableCoverageResponse(BaseModel):
    """
    Response model for getting feature coverage
    feature_id -> coverage
    """

    coverages: Dict[str, float]


class VariableHierarchyNode(BaseModel):
    """
    Flat representation of the hierarchy tree view.
    A null parent indicates a top level node
    """

    category_id: str
    category_name: str
    parent_category_id: Optional[str]


class GetVariableHierarchyResponse(BaseModel):
    flat_nodes: List[VariableHierarchyNode]


class ExperimentalGetFormulaDataRequest(BaseModel):
    markdown_formula: str
    gbi_ids: List[int]
    from_date: datetime.date
    to_date: datetime.date


class ExperimentalGetFormulaDataResponse(BaseModel):
    output: TableOutput
    output_graph: GraphOutput


####################################################################################################
# Custom Documents
####################################################################################################


class CustomDocumentListing(BaseModel):
    file_id: str
    name: str
    full_path: str
    base_path: str
    type: str
    size: int
    is_dir: bool
    listing_status: str
    upload_time: Optional[datetime.datetime] = None
    author: str
    author_org: str
    publication_time: Optional[datetime.datetime] = None
    company_name: str
    spiq_company_id: int


class ListCustomDocumentsResponse(BaseModel):
    documents: List[CustomDocumentListing]


class CheckCustomDocumentUploadQuotaResponse(BaseModel):
    authorized_s3_bucket: Optional[str] = None
    authorized_s3_prefix: Optional[str] = None
    total_quota_size: int
    remaining_quota_size: int
    message: Optional[str] = None


class AddCustomDocumentsResponse(BaseModel):
    success: bool
    added_listings: Optional[List[CustomDocumentListing]] = None


class DeleteCustomDocumentsRequest(BaseModel):
    file_paths: List[str]


class DeleteCustomDocumentsResponse(BaseModel):
    success: bool


class GetCustomDocumentFileRequest(BaseModel):
    for_preview: bool


class GetCustomDocumentFileResponse(BaseModel):
    is_preview: bool
    file_name: str
    file_type: str
    content: bytes


class CustomDocumentSummaryChunk(BaseModel):
    chunk_id: str
    headline: str
    summary: str
    long_summary: str


class GetCustomDocumentFileInfoResponse(BaseModel):
    file_id: str
    status: str
    file_type: str
    file_size: int
    author: str
    author_org: str
    upload_time: datetime.datetime
    publication_time: datetime.datetime
    company_name: str
    spiq_company_id: int
    chunks: List[CustomDocumentSummaryChunk]
    file_paths: List[str]


####################################################################################################
# Stock Search
####################################################################################################
class GetOrderedSecuritiesRequest(BaseModel):
    # use camelCase to match the GQL definition
    searchText: str
    preferEtfs: bool = False
    includeDepositary: bool = False
    includeForeign: bool = False
    order: List[str] = ["volume"]
    priorityCountry: Optional[str] = None
    priorityExchange: Optional[str] = None
    priorityCurrency: Optional[str] = None
    maxItems: int = 0


class MasterSecuritySector(BaseModel):
    id: int
    name: str
    topParentName: Optional[str]


class MasterSecurity(BaseModel):
    class MasterSecurityCountryInfo(BaseModel):
        name: Optional[str] = None
        isoCountryCode: Optional[str] = None

    # use camelCase to match the GQL definition
    gbiId: int
    symbol: Optional[str] = None
    isin: Optional[str] = None
    name: Optional[str] = None
    currency: Optional[str] = None
    country: Optional[str] = None
    primaryExchange: Optional[str] = None
    gics: Optional[int] = None
    assetType: Optional[str] = None
    securityType: Optional[str] = None
    from_: Optional[str] = Field(
        alias="from", default=None
    )  # `from` is a reserved keyword -> `from x import y`
    to: Optional[str] = None
    sector: Optional[MasterSecuritySector] = None
    isPrimaryTradingItem: Optional[bool] = None
    hasRecommendations: Optional[bool] = None
    # only request countryInfo if needed, it is batch mapped
    countryInfo: Optional[MasterSecurityCountryInfo] = None


class GetOrderedSecuritiesResponse(BaseModel):
    securities: List[MasterSecurity]


class GetCompanyDescriptionResponse(BaseModel):
    description: Optional[str]


class GetHistoricalPricesRequest(BaseModel):
    gbi_id: int
    start_date: datetime.date
    end_date: datetime.date


class GetHistoricalPricesResponse(BaseModel):
    class HistoricalPrice(BaseModel):
        date: datetime.date
        field: str
        value: Optional[float]

    prices: Optional[List[HistoricalPrice]] = None


class GetMarketDataResponse(BaseModel):
    class MarketData(BaseModel):
        gbiId: int
        date: str
        field: str
        value: Optional[float]

    market_data: Optional[List[MarketData]] = None


class GetDividendYieldResponse(BaseModel):
    dividend_yield: Optional[float] = None


class GetPriceDataResponse(BaseModel):
    class PriceData(BaseModel):
        gbiId: int
        lastClosePrice: Optional[float]
        marketCap: Optional[float]
        percentWeight: Optional[float]
        periodLowPrice: Optional[float]
        periodHighPrice: Optional[float]
        pricePercentile: Optional[float]

    price_data: Optional[PriceData] = None


class GetRealTimePriceResponse(BaseModel):
    class RealTimePrice(BaseModel):
        gbiId: int
        latestPrice: Optional[float]
        lastClosePrice: Optional[float]
        lastUpdate: Optional[datetime.datetime]
        lastClosePriceUpdate: Optional[datetime.datetime]

    price: Optional[dict] = None


class GetSecurityResponse(BaseModel):
    class SimpleSecurity(BaseModel):
        gbiId: int
        symbol: Optional[str]
        name: Optional[str]
        isin: Optional[str]
        country: Optional[str]
        currency: Optional[str]
        primaryExchange: Optional[str]
        sector: Optional[MasterSecuritySector]
        securityType: Optional[str]

    security: Optional[SimpleSecurity]


class GetNewsTopicsRequest(BaseModel):
    gbi_id: int
    delta_horizon: str
    show_hypotheses: bool = False


class GetNewsTopicsResponse(BaseModel):
    class NewsTopic(BaseModel):
        class DailyNewsCount(BaseModel):
            date: datetime.date
            newsCount: int

        class NewsItem(BaseModel):
            newsId: str
            headline: str
            isTopSource: bool
            publishedAt: datetime.datetime
            source: str
            url: str

        dailyNewsCounts: List[DailyNewsCount]
        newsItems: List[NewsItem]
        originalTopicImpact: Optional[int]
        originalTopicPolarity: Optional[str]
        topicId: str
        topicLabel: str
        topicDescription: str
        topicImpact: int
        topicRating: float
        topicPolarity: str
        topicStatus: str
        previousTopicPolarity: Optional[str]
        isCrossCompanyTopic: bool
        topicStockRationale: str

    class SentimentHistory(BaseModel):
        date: datetime.date
        sentimentScore: float

    topics: Optional[List[NewsTopic]]
    sentimentHistory: Optional[List[SentimentHistory]]


class GetSecurityProsConsResponse(BaseModel):
    class ProCon(BaseModel):
        summary: str
        details: str

    pros: Optional[List[ProCon]]
    cons: Optional[List[ProCon]]


class GetEarningsSummaryResponse(BaseModel):
    class EarningsReport(BaseModel):
        class EarningsReportDetail(BaseModel):
            class EarningsReportReference(BaseModel):
                class EarningsReportReferenceLine(BaseModel):
                    highlights: List[str]
                    paragraphs: List[str]

                valid: bool
                referenceLines: List[EarningsReportReferenceLine]
                justification: str

            header: str
            detail: str
            sentiment: str
            isAligned: bool
            references: EarningsReportReference

        date: datetime.datetime
        title: str
        details: List[EarningsReportDetail]
        highlights: str
        qaDetails: List[EarningsReportDetail]
        qaHighlights: str
        quarter: int
        year: int

    reports: Optional[List[EarningsReport]]


class GetExecutiveEarningsSummaryResponse(BaseModel):
    class SummaryChange(BaseModel):
        header: str
        detail: str

    changes: Optional[List[SummaryChange]]


class GetEarningsStatusResponse(BaseModel):
    class EarningsStatus(BaseModel):
        providerDataPresent: Optional[bool]
        providerLatestYear: Optional[int]
        providerLatestQuarter: Optional[int]
        earningsSummaryAutogenerated: Optional[bool]
        latestSummaryGenerated: Optional[datetime.datetime]
        latestSummaryStatus: Optional[str]

    earnings: Optional[EarningsStatus]


class GetUpcomingEarningsResponse(BaseModel):
    class UpcomingEarnings(BaseModel):
        date: str
        earningsTimestamp: Optional[datetime.datetime]
        quarter: int
        year: int

    earnings: Optional[UpcomingEarnings]


class GetNewsSummaryRequest(BaseModel):
    gbi_id: int
    delta_horizon: str
    show_hypotheses: bool = False


class GetNewsSummaryResponse(BaseModel):
    class SourceCount(BaseModel):
        sourceId: str
        sourceName: str
        domainUrl: str
        count: int
        deltaCount: int
        isTopSource: bool
        sentiment: float

    sentiment: float
    summary: Optional[str]
    sourceCounts: List[SourceCount]


class GetStockRelatedAgentsResponse(BaseModel):
    class RelatedAgent(BaseModel):
        agent_id: str
        agent_name: str
        run_description: Optional[str]
        last_updated: datetime.datetime

    agents: List[RelatedAgent]


####################################################################################################
# ETF
####################################################################################################


class GetEtfSummaryResponse(BaseModel):
    class EtfSummary(BaseModel):
        class EtfDataItem(BaseModel):
            id: str
            name: str

        class EtfLowHighPrice(BaseModel):
            low: float
            high: float

        activeUntilDate: Optional[str] = None
        average3MDailyVolume: Optional[float] = None
        benchmarks: Optional[List[EtfDataItem]] = None
        consensusPriceTarget: Optional[float] = None
        description: Optional[str] = None
        dividendFrequency: Optional[EtfDataItem] = None
        dividendYield: Optional[float] = None
        inceptionDate: Optional[str] = None
        lastClosePrice: Optional[float] = None
        lastDay: Optional[EtfLowHighPrice] = None
        lastYear: Optional[EtfLowHighPrice] = None
        markets: Optional[List[EtfDataItem]] = None
        netExpenseRatio: Optional[float] = None
        performanceAsOfDate: Optional[str] = None
        size: Optional[float] = None
        spiqSectors: Optional[List[EtfDataItem]] = None
        styles: Optional[List[EtfDataItem]] = None

    summary: Optional[EtfSummary] = None


class GetEtfHoldingsResponse(BaseModel):
    class EtfHoldings(BaseModel):
        class Holding(BaseModel):
            gbiId: int
            security: Optional[MasterSecurity] = None
            weight: Optional[float]

        asOfDate: datetime.date
        holdings: Optional[List[Holding]] = None

    holdings_data: Optional[EtfHoldings] = None


class GetEtfAllocationsResponse(BaseModel):
    class EtfAllocations(BaseModel):
        class MarketCapAllocation(BaseModel):
            class MarketCapAllocationItem(BaseModel):
                class MarketCapItem(BaseModel):
                    label: str
                    maxMarketCap: float
                    minMarketCap: float

                marketCap: MarketCapItem
                weight: float

            asOfDate: datetime.date
            etfGbiId: int
            allocations: Optional[List[MarketCapAllocationItem]] = None

        class SectorRegionAllocation(BaseModel):
            class SectorAllocationItem(BaseModel):
                class SectorItem(BaseModel):
                    name: str

                sector: SectorItem
                sectorId: int
                endGrossWeight: float
                endShortWeight: float
                endLongWeight: float
                endNetWeight: float

            class RegionAllocationItem(BaseModel):
                class CountryItem(BaseModel):
                    name: str

                country: CountryItem
                isoCountryCode: str
                endGrossWeight: float
                endShortWeight: float
                endLongWeight: float
                endNetWeight: float

            startDate: datetime.date
            endDate: datetime.date
            isoCurrency: str
            sectorAllocations: Optional[List[SectorAllocationItem]] = None
            regionAllocations: Optional[List[RegionAllocationItem]] = None

        marketCapAllocations: Optional[MarketCapAllocation] = None
        sectorRegionAllocations: Optional[SectorRegionAllocation] = None

    allocations: Optional[EtfAllocations] = None


class GetEtfHoldingsStatsResponse(BaseModel):
    class HoldingStats(BaseModel):
        class IndividualStat(BaseModel):
            class StatSecurity(BaseModel):
                gbiId: int
                weight: Optional[float] = None
                security: Optional[MasterSecurity] = None

            dividendYield: Optional[float] = None
            security: Optional[StatSecurity] = None

        class OverallStat(BaseModel):
            marketCap: Optional[float] = None
            priceToBook: Optional[float] = None
            priceToCashFlow: Optional[float] = None
            priceToEarnings: Optional[float] = None
            salesToEv: Optional[float] = None

        etfGbiId: int
        asOfDate: datetime.date
        individualStatistics: Optional[List[IndividualStat]] = None
        overallStatistics: Optional[OverallStat] = None

    stats: Optional[HoldingStats] = None


class GetEtfSimilarEtfsResponse(BaseModel):
    class EtfSimilarEtf(BaseModel):
        etfId: int
        security: Optional[MasterSecurity] = None
        overallSimilarityScore: Optional[float] = None
        riskSimilarityScore: Optional[float] = None
        sectorSimilarityScore: Optional[float] = None
        factorSimilarityScore: Optional[float] = None
        priceSimilarityScore: Optional[float] = None
        expenseRatio: Optional[float] = None

    similar_etfs: Optional[List[EtfSimilarEtf]] = None


####################################################################################################
# User
####################################################################################################


class Account(BaseModel):
    user_id: str
    email: str
    username: str
    name: str
    organization_id: Optional[str] = None


class UpdateUserRequest(BaseModel):
    name: str
    username: str
    email: str


class GetUsersRequest(BaseModel):
    user_ids: List[str]


class UpdateUserResponse(BaseModel):
    success: bool


class GetAccountInfoResponse(BaseModel):
    account: Account


class GetTeamAccountsResponse(BaseModel):
    accounts: List[Account]


class GetUsersResponse(BaseModel):
    accounts: List[Account]


class UserHasAccessResponse(BaseModel):
    success: bool


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
    agent_ids: List[str]


class SetAgentSectionResponse(BaseModel):
    success: bool


class RearrangeSectionRequest(BaseModel):
    new_index: int
    section_id: str


class RearrangeSectionResponse(BaseModel):
    success: bool


class RetryPlanRunRequest(BaseModel):
    agent_id: str
    plan_run_id: str


class RetryPlanRunResponse(BaseModel):
    success: bool = True


####################################################################################################
# Prompt Templates
####################################################################################################
class PromptTemplate(BaseModel):
    template_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    prompt: str
    category: str
    created_at: datetime.datetime
    plan: ExecutionPlan
    cadence_tag: str
    llm_recommended: bool = False
    notification_criteria: Optional[List[str]] = None
    user_id: Optional[str] = None
    organization_ids: Optional[List[str]] = None
    recommended_company_ids: Optional[List[str]] = None
    preview: Optional[List[OutputPreview]] = None
    description_embedding: Optional[List[float]] = None


class UserOrganization(BaseModel):
    organization_id: str
    organization_name: str


class CreatePromptTemplateRequest(BaseModel):
    name: str
    description: str
    prompt: str
    category: str
    plan_run_id: str
    cadence_tag: str
    organization_ids: Optional[List[str]] = None
    recommended_company_ids: Optional[List[str]] = None
    notification_criteria: Optional[List[str]] = None


class CreatePromptTemplateResponse(BaseModel):
    template_id: str
    user_id: str
    name: str
    description: str
    description_embedding: List[float]
    prompt: str
    category: str
    created_at: datetime.datetime
    plan_run_id: str
    cadence_tag: str
    organization_ids: Optional[List[str]] = None
    recommended_company_ids: Optional[List[str]] = None
    notification_criteria: Optional[List[str]] = None


class UpdatePromptTemplateRequest(BaseModel):
    template_id: str
    name: str
    description: str
    prompt: str
    category: str
    plan: ExecutionPlan
    cadence_tag: str
    notification_criteria: Optional[List[str]] = None
    organization_ids: Optional[List[str]] = None
    recommended_company_ids: Optional[List[str]] = None


class UpdatePromptTemplateResponse(BaseModel):
    prompt_template: PromptTemplate


class GetPromptTemplatesResponse(BaseModel):
    prompt_templates: List[PromptTemplate]


class CopyAgentToUsersRequest(BaseModel):
    src_agent_id: str
    dst_user_ids: List[str]
    dst_agent_name: Optional[str] = None


class CopyAgentToUsersResponse(BaseModel):
    user_id_to_new_agent_id_map: Dict[str, str]


class GenTemplatePlanResponse(BaseModel):
    plan: Optional[ExecutionPlan] = None
    preview: Optional[List[OutputPreview]] = None


class GenTemplatePlanRequest(BaseModel):
    template_prompt: str


class RunTemplatePlanResponse(BaseModel):
    agent_id: str


class RunTemplatePlanRequest(BaseModel):
    template_prompt: str
    plan: ExecutionPlan
    is_draft: bool = False
    notification_criteria: Optional[List[str]] = None
    cadence_description: Optional[str] = None


class DeletePromptTemplateRequest(BaseModel):
    template_id: str


class DeletePromptTemplateResponse(BaseModel):
    template_id: str


class GetCompaniesResponse(BaseModel):
    companies: List[UserOrganization]


class GenPromptTemplateFromPlanResponse(BaseModel):
    prompt_str: str


class GenPromptTemplateFromPlanRequest(BaseModel):
    plan_run_id: str
    agent_id: str


class FindTemplatesRelatedToPromptResponse(BaseModel):
    prompt_templates: Optional[List[PromptTemplate]] = None


class FindTemplatesRelatedToPromptRequest(BaseModel):
    query: str


################################################
# Quality Tool Classes
################################################
class AgentQC(BaseModel):
    agent_qc_id: Optional[str] = None
    agent_id: str
    user_id: str
    agent_status: Optional[str] = None
    query_order: Optional[int] = 0
    agent_name: Optional[str] = None
    plan_id: Optional[str] = None
    query: Optional[str] = None
    cs_reviewer: Optional[str] = None
    eng_reviewer: Optional[str] = None
    prod_reviewer: Optional[str] = None
    follow_up: Optional[str] = None
    score_rating: Optional[int] = None
    priority: Optional[str] = None
    use_case: Optional[str] = None
    problem_area: Optional[str] = None
    cs_failed_reason: Optional[str] = None
    cs_attempt_reprompting: Optional[str] = None
    cs_expected_output: Optional[str] = None
    cs_notes: Optional[str] = None
    canned_prompt_id: Optional[str] = None
    eng_failed_reason: Optional[str] = None
    eng_solution: Optional[str] = None
    eng_solution_difficulty: Optional[int] = None
    jira_link: Optional[str] = None
    slack_link: Optional[str] = None
    fullstory_link: Optional[str] = None
    duplicate_agent: Optional[str] = None
    created_at: datetime.datetime
    last_updated: datetime.datetime
    cognito_username: Optional[str] = None
    agent_feedbacks: List[AgentFeedback] = []
    cs_reviewed: Optional[bool] = None
    eng_reviewed: Optional[bool] = None
    prod_reviewed: Optional[bool] = None
    owner_name: Optional[str] = None
    owner_organization_name: Optional[str] = None
    prod_priority: Optional[str] = None
    prod_notes: Optional[str] = None
    is_spoofed: Optional[bool] = False
    qc_status: Optional[str] = None


class HorizonCriteriaOperator(enum.StrEnum):
    equal = "="
    greater_than = ">"
    less_than = "<"
    not_equal = "!="
    ilike = "ILIKE"
    between = "BETWEEN"
    in_operator = "IN"
    equal_any = "=ANY"


class ScoreRating(enum.IntEnum):
    BROKEN = 0
    BAD = 3
    NEUTRAL = 5
    GOOD = 9


class HorizonCriteria(BaseModel):
    column: str
    operator: HorizonCriteriaOperator
    arg1: Any
    arg2: Optional[Any]


class Pagination(BaseModel):
    page_index: int
    page_size: int


class SearchAgentQCRequest(BaseModel):
    filter_criteria: List[HorizonCriteria]
    search_criteria: List[HorizonCriteria]
    pagination: Optional[Pagination] = None


class SearchAgentQCResponse(BaseModel):
    agent_qcs: List[AgentQC]
    total_agent_qcs: int


class AddAgentQCRequest(BaseModel):
    agent_id: str
    user_id: str


class AddAgentQCResponse(BaseModel):
    success: bool
    agent_qc: Optional[AgentQC]


class UpdateAgentQCRequest(BaseModel):
    agent_qc: AgentQC


class UpdateAgentQCResponse(BaseModel):
    success: bool


class QueryWithBreakdown(BaseModel):
    use_case: Optional[str]
    score_rating: Optional[int]


class AgentSnapshot(BaseModel):
    week_start: datetime.datetime
    week_end: datetime.datetime
    live_agents: int
    created_agents: int


class GetQueryHistoricalAgentsRequest(BaseModel):
    start_date: Optional[datetime.datetime] = None
    end_date: Optional[datetime.datetime] = None


class HistoricalAgentsSnapshot(BaseModel):
    live: Tuple[datetime.datetime, int]
    non_live: Tuple[datetime.datetime, int]


class GetAgentsQCRequest(BaseModel):
    start_dt: Optional[datetime.datetime] = None
    end_dt: Optional[datetime.datetime] = None
    live_only: bool = True


class AgentQCInfo(BaseModel):
    agent_id: str
    agent_name: str
    user_id: str
    user_name: str
    user_org_id: str
    user_org_name: str
    user_is_internal: bool
    help_requested: bool

    most_recent_plan_run_id: str
    most_recent_plan_run_status: Status
    last_run_start: datetime.datetime

    last_successful_run: Optional[datetime.datetime] = None
    run_count_by_status: Dict[Status, int] = {}


class GetLiveAgentsQCResponse(BaseModel):
    agent_infos: List[AgentQCInfo]


# Request model for creating a Jira ticket
class CreateJiraTicketRequest(BaseModel):
    project_key: str
    summary: str
    description: str
    parent: Optional[str] = None
    priority: Optional[str] = None
    labels: List[str] = ["automation", "python"]
    issue_type: Optional[str] = "Task"
    assignee: Optional[str] = None
    additional_fields: Optional[Dict[str, Any]] = None


# Response model for the created Jira ticket
class CreateJiraTicketResponse(BaseModel):
    ticket_id: str
    ticket_url: str
    success: bool


class JiraTicketCriteria(BaseModel):
    project_key: str  # The key of the project in which to create the issue
    summary: str  # A short summary of the issue
    description: str  # Detailed description of the issue
    issue_type: Optional[str] = "Task"  # Type of issue, e.g., Bug, Task, Story
    assignee: Optional[str] = None  # Username of the user assigned to the issue
    priority: Optional[str] = None  # Priority of the issue, e.g., High, Medium, Low
    parent: Optional[str] = None  # Parent issue to which this issue belongs (Epics)
    story_points: Optional[float] = None  # Story points for Agile projects
    # below are unsupported
    reporter: Optional[str] = None  # Username of the reporter of the issue
    labels: Optional[List[str]] = None  # List of labels for the issue
    components: Optional[List[str]] = None  # Components involved, e.g., Backend, UI
    versions: Optional[List[str]] = None  # Versions affected by the issue
    fix_versions: Optional[List[str]] = None  # Versions in which the issue is fixed
    due_date: Optional[str] = None  # Due date in YYYY-MM-DD format
    environment: Optional[str] = None  # Environment where the issue occurs
    timetracking: Optional[Dict[str, str]] = None  # Time tracking info
    sprint_id: Optional[int] = None  # ID of the sprint to link the issue to
    custom_fields: Optional[Dict[str, Any]] = None  # Additional custom fields as key-value pairs


####################################################################################################
# Transformation
####################################################################################################
class TransformTableOutputRequest(BaseModel):
    agent_id: str  # for authorization purpose
    plan_id: str
    plan_run_id: str  # record which run this transformation applies to originally
    task_id: str
    # whether the transformation applies to this run only, or all following runs
    is_transformation_local: bool
    transformation: TableTransformation
    # when `is_transformation_local` is False, `effective_from` is required
    effective_from: Optional[datetime.datetime] = None


class TransformTableOutputResponse(BaseModel):
    transformation_id: str


class SuccessResponse(BaseModel):
    success: bool


class UpdateTransformationSettingsRequest(BaseModel):
    agent_id: str
    transformation_id: str
    is_transformation_local: bool
    effective_from: Optional[datetime.datetime] = None
