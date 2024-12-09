from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext


class SetupScheduleInput(ToolArgs):
    pass


@tool(
    description=(
        "This function will create or modify a schedule for "
        "a task to be performed at a later time or on "
        "at regular intervals"
    ),
    category=ToolCategory.AUTOMATION,
    tool_registry=default_tool_registry(),
    is_visible=True,
    enabled=False,
)
async def set_schedule(args: SetupScheduleInput, context: PlanRunContext) -> str:
    # This is a mock
    await tool_log(
        log="Setting up a schedule for this task...",
        context=context,
    )

    # should this be a tool log or a message?
    # message = Message(BaseModel):
    # agent_id: str = Field(default_factory=lambda: str(uuid4()))  # default is for testing only
    # message: IOType
    # is_user_message: bool
    await tool_log(
        log="Next run set for tomorrow at 8:00 AM",
        context=context,
    )
    return ""


class DeleteScheduleInput(ToolArgs):
    pass


@tool(
    description=("This function will delete a scheduled task"),
    category=ToolCategory.AUTOMATION,
    tool_registry=default_tool_registry(),
    is_visible=True,
    enabled=False,
)
async def delete_schedule(args: DeleteScheduleInput, context: PlanRunContext) -> str:
    # This is a mock
    await tool_log(
        log="Deleting the schedule for this task task...",
        context=context,
    )
    await tool_log(
        log="If you change your mind just ask me to setup a schedule again",
        context=context,
    )

    return ""


class PauseScheduleInput(ToolArgs):
    pass


@tool(
    description=("This function will temporarily pause a scheduled task"),
    category=ToolCategory.AUTOMATION,
    tool_registry=default_tool_registry(),
    is_visible=True,
    enabled=False,
)
async def pause_schedule(args: PauseScheduleInput, context: PlanRunContext) -> str:
    # This is a mock
    await tool_log(
        log="Temporarily pausing the schedule for this task...",
        context=context,
    )
    await tool_log(
        log="Next run set for 3 weeks from thursday",
        context=context,
    )

    return ""


class SetNotificationCriteria(ToolArgs):
    pass


@tool(
    description=(
        "This function will setup or modify the trigger event for a notification. "
        "Supported triggers are: one time future reminders, "
        "regular periods such as daily/weekly/monthly schedule, "
        "or when some user specified event takes place."
    ),
    category=ToolCategory.AUTOMATION,
    tool_registry=default_tool_registry(),
    is_visible=True,
    enabled=False,
)
async def set_notification(args: SetNotificationCriteria, context: PlanRunContext) -> str:
    # This is a mock
    await tool_log(
        log="Setting up notification to your specifications...",
        context=context,
    )

    return ""


class DeleteNotificationCriteria(ToolArgs):
    pass


@tool(
    description=("This function will remove a previously setup notification."),
    category=ToolCategory.AUTOMATION,
    tool_registry=default_tool_registry(),
    is_visible=True,
    enabled=False,
)
async def delete_notification(args: DeleteNotificationCriteria, context: PlanRunContext) -> str:
    # This is a mock
    await tool_log(
        log="Removing that notification for you...",
        context=context,
    )

    return ""


class NotifyUserInput(ToolArgs):
    pass


@tool(
    description=("This function will notify user of important events."),
    category=ToolCategory.AUTOMATION,
    tool_registry=default_tool_registry(),
    is_visible=False,
    enabled=False,
)
async def notify_user(args: NotifyUserInput, context: PlanRunContext) -> str:
    # This is a mock
    await tool_log(
        log="Something cool happened, you better check it out! [link]",
        context=context,
    )

    return ""
