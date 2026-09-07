"""Utility functions for method caching and coordinate data parsing."""

import copy
from functools import wraps
import inspect
from collections import OrderedDict
import threading
from pathlib import Path
from typing import Any, Callable, TypeVar, overload, cast
import numpy as np
import yaml

T = TypeVar("T", bound=Callable[..., Any])


def _make_hashable(obj: Any) -> Any:
    """Recursively convert unhashable types into hashable ones."""
    if isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(i) for i in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    if hasattr(obj, "model_dump"):
        return _make_hashable(obj.model_dump())
    try:
        hash(obj)
        return obj
    except TypeError:
        return id(obj)


def deep_copy_shape_or_builder(obj: Any, memo: Any = None) -> Any:
    """Safely deep copy build123d objects, handling frames and context tokens in Builders."""
    if memo is None:
        memo = {}

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    match obj:
        case _ if hasattr(obj, "__dict__") and any(
            k in obj.__dict__ for k in ("_python_frame", "parent_frame", "_reset_tok", "workplanes_context")
        ):
            copied_obj = copy.copy(obj)
            memo[obj_id] = copied_obj
            copied_obj.__dict__ = {}
            for k, v in obj.__dict__.items():
                if k in ("_python_frame", "parent_frame", "_reset_tok", "workplanes_context"):
                    copied_obj.__dict__[k] = None
                else:
                    copied_obj.__dict__[k] = deep_copy_shape_or_builder(v, memo)
            return copied_obj
        case list():
            copied_list: list[Any] = []
            memo[obj_id] = copied_list
            copied_list.extend(deep_copy_shape_or_builder(x, memo) for x in obj)
            return copied_list
        case tuple():
            copied_tuple = tuple(deep_copy_shape_or_builder(x, memo) for x in obj)
            memo[obj_id] = copied_tuple
            return copied_tuple
        case dict():
            copied_dict: dict[Any, Any] = {}
            memo[obj_id] = copied_dict
            for k, v in obj.items():
                copied_dict[deep_copy_shape_or_builder(k, memo)] = deep_copy_shape_or_builder(v, memo)
            return copied_dict
        case set():
            copied_set: set[Any] = set()
            memo[obj_id] = copied_set
            copied_set.update(deep_copy_shape_or_builder(x, memo) for x in obj)
            return copied_set

    try:
        res = copy.deepcopy(obj, memo)
        memo[obj_id] = res
        return res
    except Exception:
        try:
            res = copy.copy(obj)
            memo[obj_id] = res
            return res
        except Exception:
            memo[obj_id] = obj
            return obj


@overload
def method_cache(func: Callable[..., Any]) -> Callable[..., Any]: ...


@overload
def method_cache(
    *, maxsize: int = 128, deepcopy: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


_method_cache_lock = threading.Lock()


def method_cache(func: Callable[..., Any] | None = None, *, maxsize: int = 128, deepcopy: bool = True) -> Any:
    """Create per-instance cache to avoid memory leaks and Pydantic @validate_call conflicts."""

    def decorator(f: T) -> T:
        @wraps(f)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            cache_attr = f"_cache_{f.__name__}"
            lock_attr = f"_lock_{f.__name__}"

            if not hasattr(self, lock_attr):
                with _method_cache_lock:
                    if not hasattr(self, lock_attr):
                        setattr(self, lock_attr, threading.Lock())

            lock = getattr(self, lock_attr)

            with lock:
                if not hasattr(self, cache_attr):
                    setattr(self, cache_attr, OrderedDict())
                cache = getattr(self, cache_attr)
                key = (_make_hashable(args), _make_hashable(kwargs))
                if key in cache:
                    cache.move_to_end(key)
                    return deep_copy_shape_or_builder(cache[key]) if deepcopy else cache[key]

            # Execute the function outside the lock to prevent deadlock if f() calls other cached methods
            result = f(self, *args, **kwargs)

            with lock:
                cache[key] = result
                if len(cache) > maxsize:
                    cache.popitem(last=False)
                return deep_copy_shape_or_builder(result) if deepcopy else result

        try:
            setattr(wrapper, "__signature__", inspect.signature(f))
        except (ValueError, TypeError):
            pass

        if hasattr(wrapper, "__wrapped__"):
            delattr(wrapper, "__wrapped__")

        return cast(T, wrapper)

    if func is None:
        return decorator
    return decorator(func)


def load_measurements(path: str) -> dict[int | str, np.ndarray]:
    """
    Load and parse the measurements YAML file and return processed numpy arrays.

    Example YAML (list format):
    ```yaml
    - [0, 0, 0]
    - [1, 2, 3]
    ```
    Returns: {1: array([0, 0, 0]), 2: array([1, 2, 3])}

    Example YAML (dict format):
    ```yaml
    p1: [0, 0, 0]
    p2: [1, 2, 3]
    ```
    Returns: {'p1': array([0, 0, 0]), 'p2': array([1, 2, 3])}

    Example YAML (nested section):
    ```yaml
    section_a:
      p1: [0, 0, 0]
      p2: [1, 2, 3]
    section_b:
      p3: [4, 5, 6]
    ```
    """
    file_path = Path(path)
    if not file_path.exists():
        return {}

    with open(file_path, "r") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML at {file_path}") from e

    p = {}
    if isinstance(data, list):
        for idx, item in enumerate(data):
            p[idx + 1] = np.array(item, dtype=float)
    elif isinstance(data, dict):
        for k, item in data.items():
            p[k] = np.array(item, dtype=float)
    return p
