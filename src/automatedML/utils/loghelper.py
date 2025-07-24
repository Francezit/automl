import logging
import uuid
import time
import json


def call_function_begin_logger(nameFunction: str, context_name: str = None, fun=None, arg: str = None, option: tuple = None, includeOption: bool = True):
    if fun is None:
        fun = logging.info
    if arg is None:
        arg = ""
    st = time.time()

    if context_name is not None:
        id = f"{context_name}_{nameFunction}"
    else:
        id = f"{nameFunction}_{uuid.uuid4().hex}"

    msg = f"[Call function {id}] {arg}"
    fun(msg)
    if option is not None and includeOption:
        try:
            msg = f"[Option function {id}] {json.dumps(option)}"
            fun(msg)
        except:
            pass
    return {"id": id, "st": st}


def trace_function_logger(logId: dict, msg: str, fun=None):
    if fun is None:
        fun = logging.info

    id = logId["id"]
    msg = f"[Trace function {id}] {msg}"
    fun(msg)


def call_function_end_logger(logId: dict, fun=None):
    if fun is None:
        fun = logging.info

    et = time.time()
    id = logId["id"]
    dt = et - logId["st"]
    msg = f"[Call function {id}] processed in {str(dt)}s"
    fun(msg)


def call_function_end_error_logger(logId: dict, err: str, fun=None):
    if fun is None:
        fun = logging.info

    et = time.time()
    id = logId["id"]
    dt = et - logId["st"]
    msg = f"[Call function {id}] processed in {str(dt)}s but an error occured: {err}"
    fun(msg)


__all__ = [
    "call_function_begin_logger", "trace_function_logger",
    "call_function_end_logger", "call_function_end_error_logger"
]
