from agent_service.endpoints.models import Status


class AgentExecutionError(Exception):
    result_status = Status.ERROR


class AgentRetryError(Exception):
    pass


class NonRetriableError(AgentExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def get_message_for_llm(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


class EmptyInputError(NonRetriableError):
    """
    Use this error if a tool is given empty input. E.g. if a summary tool gets
    an empty list of texts.
    NOTE: We need both this and EmptyOutputError because there may be some cases
    where the outputs of multiple tools are fed into a single tool (e.g. a
    summarizer). In that case, we only want to error if ALL of the prior tool
    outputs were empty.
    """

    result_status = Status.NO_RESULTS_FOUND


class EmptyOutputError(NonRetriableError):
    """
    NOTE: Read above exception description before using this one

    Use this error if a tool results in an empty output. Especially useful for
    filter tools, etc. Note that this should only be raised if the workflow
    CANNOT continue after the error. In general, prefer an EmptyInputError for
    flexibility, since there may be cases where multiple outputs are merged, in
    which case a single output being empty is acceptable.
    """

    result_status = Status.NO_RESULTS_FOUND
