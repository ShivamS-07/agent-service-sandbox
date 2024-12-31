import asyncio
import datetime
import functools
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Dict,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
)

from agent_service.utils.progress_bar_args import FrontendProgressBar, ProgressBarArgs

T = TypeVar("T")
P = ParamSpec("P")
ANY_TYPE = TypeVar("ANY_TYPE")

MAX_CONCURRENCY = 4


def sync(async_fn: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    """
    Wrapper class around an async function to expose a synchronous version by running
    it in its own event loop.

    The intended use for this is to interface between an async function and a
    synchronous caller for example at the API layer, do not use this in an
    already async function.

    NOTE: this cannot be called as a synchronous method inside an async context!
    We will need to asyncify the chain of calls all the way down to where it is used.
    """

    @functools.wraps(async_fn)
    def wrapper(*args, **kwargs) -> T:  # type: ignore
        return asyncio.run(async_fn(*args, **kwargs))

    return wrapper


async def gather_with_stop(tasks: Collection[Awaitable], stop_count: int) -> List[Any]:
    successful = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            successful.append(result)
        if len(successful) >= stop_count:
            break
    return successful


async def get_consensus(  # type: ignore
    function: Callable,
    *args,
    experts: int = 3,
    min_agreement: int = 2,
    default: Optional[Any] = None,
    target: Optional[Any] = None,
) -> Optional[Any]:
    # allows us to get a consensus of multiple nondetermistic (e.g. GPT) calls
    tasks = []
    for _ in range(experts):
        tasks.append(function(*args))
    results = await asyncio.gather(*tasks)
    counts = Counter(results)
    if (
        target is not None
    ):  # if there's a specific result we are looking for, allows for sensible non-majority logic
        if counts.get(target, 0) >= min_agreement:
            return target
    else:
        top, top_count = counts.most_common(1)[0]
        if top_count >= min_agreement:
            return top
    return default


async def to_awaitable(val: T) -> T:
    return val


# credit: https://stackoverflow.com/questions/48483348/how-to-limit-concurrency-with-python-asyncio
async def gather_with_concurrency(
    tasks: Collection[Awaitable],
    n: int = MAX_CONCURRENCY,
    return_exceptions: bool = False,
    progress_bar_args: Optional[ProgressBarArgs] = None,
) -> Any:
    n = min(n, len(tasks))  # no greater than number of tasks
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task: Awaitable) -> Any:
        async with semaphore:
            return await task

    semaphored_tasks = (sem_task(task) for task in tasks)

    if progress_bar_args:
        log_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow()
        result = await FrontendProgressBar.gather(
            *semaphored_tasks,
            desc=progress_bar_args.desc,
            context=progress_bar_args.context,
            log_id=log_id,
            created_at=created_at,
        )
        await FrontendProgressBar.all_done(
            context=progress_bar_args.context,
            desc=progress_bar_args.desc,
            log_id=log_id,
            created_at=created_at,
        )
        return result
    else:
        return await asyncio.gather(*semaphored_tasks, return_exceptions=return_exceptions)


async def gather_dict_as_completed(
    task_to_id: Dict[Awaitable[T], ANY_TYPE], n: Optional[int] = None
) -> AsyncIterable[Tuple[Optional[ANY_TYPE], Optional[T]]]:
    """Run multiple coroutines concurrently with limited semaphore and yield the results as soon as
    they are done. Note this is MORE efficient than `gather_with_concurrency` as you can process
    the tasks whenever they are done, instead of waiting for all of them to finish. However, the
    order of the results is in the order of completion, not the order of the input.

    To use it, do this:
    async for task_id, result in gather_dict_as_completed(task_to_id, n):
        # do something with the result

    Args:
        task_to_id (Dict[Awaitable[T], ANY_TYPE]): A dictionary of future to id mapping.
    `id` could be anything that later helps you identify the order/result.
        n (Optional[int]): number of coroutines to run concurrently. None means run
    all of them concurrently.

    Returns:
        AsyncIterable[Tuple[Optional[ANY_TYPE], Optional[T]]]: An async generator that yields
    the results of the coroutines as soon as it's done. When the task fails, it will yield None.
    """
    if n is None:
        n = len(task_to_id)
    else:
        n = min(n, len(task_to_id))

    semaphore = asyncio.Semaphore(n)

    async def sem_task(task_id: ANY_TYPE, task: Awaitable[T]) -> Tuple[Any, T]:
        async with semaphore:
            result = await task
            return task_id, result

    wrapped_tasks = [sem_task(task_id, task) for task, task_id in task_to_id.items()]
    for coro in asyncio.as_completed(wrapped_tasks):
        try:
            task_id, result = await coro
            yield (task_id, result)
        except Exception as e:
            print(f"Error in gather_dict_as_completed: {e}")
            yield (None, None)


def async_wrap(func: Callable[P, T]) -> Callable[P, Coroutine[Any, Any, T]]:
    """A decorator to wrap a synchronous function to async.
    2 ways to use it:
        1) Put it above func like @async_wrap
        2) Directly wrap the function like async_func = async_wrap(func)
    To call wrapped function:
        1) await wrapped_func
        2) Put it into coroutines: await gather_with_concurrency(coroutines, N)
    Reference:
    https://stackoverflow.com/questions/43241221/how-can-i-wrap-a-synchronous-function
    -in-an-async-coroutine

    Asyncio Doc about `run_in_executor`: basically still creates threads defaultly to
    implement asynchronous
    https://docs.python.org/3/library/asyncio-eventloop.html#executing-code-in-thread
    -or-process-pools
    """

    @functools.wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs) -> T:  # type: ignore
        pfunc = functools.partial(func, *args, **kwargs)
        if loop is None:
            loop = asyncio.get_event_loop()
        if executor is not None:
            return await loop.run_in_executor(executor, pfunc)

        # explicitly create a new thread for it and cleanup once it's done
        # default max_workers = min(32, os.cpu_count() + 4)
        # spawn only `1` thread if not specified to avoid threading overhead
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, pfunc)

    return run  # type: ignore


async def identity(x: Any) -> Any:
    return x


def run_async_background(coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    """Run an async function in the background without waiting for it to complete.
    Be careful! If the coroutine doesn't finish before the program exits, it will be terminated

    Args:
        coro (Coroutine[Any, Any, T]): An async function object, e.g `func(a, b, c)`

    Returns:
        asyncio.Task: Return the task object. If you need it to be awaited/blocking, use `await task`
    in the program
    """
    return asyncio.create_task(coro)
