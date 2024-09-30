from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar, Union

from agent_service.planner.errors import AgentCancelledError
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.postgres import Postgres, SyncBoostedPG

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


async def raise_exc_if_cancelled(
    func_name: str, db: Union[AsyncDB, Postgres], context: PlanRunContext
) -> None:
    raise_exc = False
    if isinstance(db, AsyncDB):
        res = await gather_with_concurrency(
            [
                db.is_agent_deleted(agent_id=context.agent_id),
                db.is_cancelled(ids_to_check=[context.plan_id, context.plan_run_id]),
            ]
        )
        raise_exc = any(res)
    elif isinstance(db, Postgres):
        raise_exc = db.is_cancelled(
            ids_to_check=[context.plan_id, context.plan_run_id]
        ) or db.is_agent_deleted(agent_id=context.agent_id)

    if raise_exc:
        raise AgentCancelledError(
            f"{context.agent_id=}, {context.plan_id=}, {context.plan_run_id=} is cancelled. "
            f"Skipping the execution of function <{func_name}>"
        )


def check_cancelled_decorator(async_func: F) -> F:
    """
    A decorator to check if the agent, plan or plan run is cancelled before executing the function.
    This can effectively be used to kill an agent while running inside a tool that has a few steps
    To use the decorator properly, make sure your function signature includes `context`,
    and use keyword arguments for other parameters, e.g.

    @check_cancelled_decorator
    async def example_func(context: PlanRunContext, param1: str) -> None: ...

    example_func(context=context, param1="test")

    It won't check/raise exception if `context.skip_db_commit` is True or
    `context.run_tasks_without_prefect` is True (mostly in local runs)
    DO NOT use it to wrap tools
    """

    @wraps(async_func)
    async def wrapper(context: PlanRunContext, *args: Any, **kwargs: Any):  # type: ignore
        if not context.skip_db_commit or not context.run_tasks_without_prefect:
            # do not check in local
            await raise_exc_if_cancelled(
                func_name=async_func.__qualname__,
                db=AsyncDB(SyncBoostedPG(skip_commit=context.skip_db_commit)),
                context=context,
            )
        return await async_func(context, *args, **kwargs)

    return wrapper  # type: ignore


if __name__ == "__main__":
    import asyncio

    @check_cancelled_decorator
    async def test_func(context: PlanRunContext, param1: str) -> int:
        print("test_func")
        return 0

    async def main() -> None:
        context = PlanRunContext.get_dummy(user_id="")
        await test_func(param1="test", context=context)  # nothing

        context.skip_db_commit = False
        context.run_tasks_without_prefect = False
        try:
            await test_func(param1="test", context=context)
        except AgentCancelledError as e:
            print(e)

    asyncio.run(main())
