import time
from loguru import logger


def timeit(method):
    """Decorator to time the execution of a function."""

    def timed(*args, **kw):
        start_time = time.time()
        logger.info(f"\nStarting execution of {method.__name__}.")
        result = method(*args, **kw)
        end_time = time.time()
        n_seconds = int(end_time - start_time)
        if n_seconds < 60:
            logger.info(f"\n{method.__name__} : {n_seconds}s to execute")
        elif 60 < n_seconds < 3600:
            logger.info(
                f"\n{method.__name__} : {n_seconds // 60}min {n_seconds % 60}s to execute"
            )
        else:
            logger.info(
                f"\n{method.__name__} : {n_seconds // 3600}h {n_seconds % 3600 // 60}min {n_seconds // 3600 % 60}s to execute"
            )
        return result

    return timed


def get_formatted_time():
    return time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
