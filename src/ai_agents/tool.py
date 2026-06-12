import abc
import asyncio
import dataclasses
import inspect
from asyncio import AbstractEventLoop
from collections.abc import Callable
from typing import get_type_hints, get_args, Optional, Type, Union, List, Annotated, TypeVar, Generic, Any, \
    Dict, Tuple

import jsonref  # type: ignore[import-untyped]
from jsonref import JsonRef  # type: ignore[import-untyped]
from pydantic import create_model, BaseModel, Field

RawParameters = Union[str, dict]
ToolType = TypeVar("ToolType")
FunctionInput = TypeVar("FunctionInput")
FunctionOutput = TypeVar("FunctionOutput", default=Any)
F = TypeVar("F", bound=Callable[..., Any])


@dataclasses.dataclass
class FunctionInputPayload:
    name: str
    arguments: RawParameters
    extras: Optional[Any] = None


@dataclasses.dataclass
class FunctionOutputPayload(Generic[FunctionOutput]):
    result: FunctionOutput
    extras: Optional[Any] = None


def _use_or_replace_loop(current_loop: Optional[AbstractEventLoop]) -> AbstractEventLoop:
    if current_loop is None or not current_loop.is_running():
        return asyncio.new_event_loop()
    return current_loop


class ToolCollection(abc.ABC, Generic[ToolType, FunctionInput, FunctionOutput]):
    def __init__(self, *tools: Callable):
        self._tools: Dict[str, Tuple[ToolMetadata, Callable]] = dict()
        for candidate_tool in tools:
            metadata = tool_metadata(candidate_tool)
            self._tools[metadata.name] = (metadata, candidate_tool)

    def __call__(self, name, params: RawParameters) -> FunctionOutput:
        _, callable_tool = self._tools[name]
        return call_with_params(callable_tool, params)

    @abc.abstractmethod
    def tools(self) -> List[ToolType]:
        """
        Return a list of tools that can be used by the LLM
        """
        ...

    @abc.abstractmethod
    def extract_parameters(self, inp: FunctionInput) -> FunctionInputPayload:
        """
        Extract the parameters from the input
        """
        ...

    async def invoke_fn_async(self, *functions: FunctionInput) -> Dict[str, FunctionOutputPayload[FunctionOutput]]:
        assert len(functions) > 0, "At least one function must be provided"

        async def run_as_async(_name, _params, _extras):
            return _name, FunctionOutputPayload(result=self(_name, _params), extras=_extras)

        async def run(_name, _params, _extras):
            return _name, FunctionOutputPayload(result=await self(_name, _params), extras=_extras)

        lambdas = list()
        for function in functions:
            params = self.extract_parameters(function)
            metadata, _ = self._tools[params.name]
            if not metadata.is_async:
                lambdas.append(run_as_async(params.name, params.arguments, params.extras))
            else:
                lambdas.append(run(params.name, params.arguments, params.extras))
        return dict(await asyncio.gather(*lambdas))

    def invoke_fn(self, *functions: FunctionInput, loop: Optional[AbstractEventLoop] = None) -> Dict[
        str, FunctionOutputPayload[FunctionOutput]]:
        with asyncio.Runner(loop_factory=lambda: _use_or_replace_loop(loop)) as runner:
            return runner.run(self.invoke_fn_async(*functions))


@dataclasses.dataclass
class ToolMetadata:
    name: str
    description: str
    model: Type[BaseModel]
    is_async: bool


def tool(
        name: Optional[str] = None,
        description: Optional[str] = None
) -> Callable[[F], F]:
    """
    Create a tool from a function
    :param name: The name of the tool, defaults to the function name
    :param description: The description of the tool, defaults to the function docstring
    """

    def add_tool_metadata(func: F) -> F:
        metadata_source: Callable[..., Any] = func

        fn_name = name or func.__name__
        fn_description = description or func.__doc__
        assert fn_description is not None, f"Function {fn_name} must have a description"
        fn_description = fn_description.strip()

        if not inspect.isfunction(func) and hasattr(func, "__call__"):
            metadata_source = func.__call__

        hints = get_type_hints(metadata_source, include_extras=True)

        # Handle any missing annotations
        for param_name, param_annotation in hints.items():
            args = get_args(param_annotation)
            if not args:
                hints[param_name] = Annotated[param_annotation, Field()]

        # The return type is not needed
        if "return" in hints:
            del hints["return"]

        # Attach the metadata to the function or class
        setattr(func, "__tool_metadata__", ToolMetadata(
            name=fn_name,
            description=fn_description,
            model=create_model(fn_name, __doc__=fn_description, **hints),
            is_async=inspect.iscoroutinefunction(metadata_source)
        ))
        return func

    return add_tool_metadata


class ToolNotDecoratedError(Exception):
    """Raised when a function without the @tool decorator is used"""
    pass


def tool_metadata(fn: Callable) -> ToolMetadata:
    try:
        return getattr(fn, "__tool_metadata__")
    except AttributeError as e:
        raise ToolNotDecoratedError(
            f"Function {fn.__name__} is not a tool, did you forget to add the @tool decorator?") from e


def call_with_params(fn, params: RawParameters):
    """
    Call the function with parameters as either a JSON string or a dictionary
    """
    metadata = tool_metadata(fn)
    validated_params = metadata.model.model_validate_json(params) if isinstance(params, str) else metadata.model.model_validate(
        params)
    param_values = {k: getattr(validated_params, k) for k in type(validated_params).model_fields.keys()}
    return fn(**param_values)


def input_schema(fn) -> JsonRef:
    """
    Get the JSON schema for the input parameters
    """

    def remove_title_recursive(obj):
        if not isinstance(obj, dict):
            return
        if "title" in obj:
            del obj["title"]
        if "properties" in obj:
            for prop in obj["properties"].values():
                remove_title_recursive(prop)

    schema = jsonref.replace_refs(tool_metadata(fn).model.model_json_schema(), lazy_load=False)
    if "$defs" in schema:
        del schema["$defs"]
    remove_title_recursive(schema)
    return schema
