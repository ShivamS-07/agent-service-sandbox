import asyncio
import functools
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    List,
    Optional,
    TypeVar,
)

T = TypeVar("T")

MAX_CONCURRENCY = 4


def sync(async_fn: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
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
async def gather_with_concurrency(tasks: Collection[Awaitable], n: int = MAX_CONCURRENCY) -> Any:
    n = min(n, len(tasks))  # no greater than number of tasks
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task: Awaitable) -> Any:
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


def async_wrap(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
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
