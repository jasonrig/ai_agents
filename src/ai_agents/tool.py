import abc
import asyncio
import dataclasses
import inspect
from collections import abc as collections_abc
from asyncio import AbstractEventLoop
from types import MappingProxyType
from typing import Awaitable, Iterator, Literal, Mapping, TypeGuard, get_type_hints, get_args, Optional, Type, Union, List, Annotated, TypeVar, Generic, Any, \
    Dict, Tuple

import jsonref  # type: ignore[import-untyped]
from jsonref import JsonRef  # type: ignore[import-untyped]
from pydantic import create_model, BaseModel, Field

RawParameters = Union[str, dict]

# Collection generics: provider-specific tool shape, incoming payload type, and tool return type.
ToolType = TypeVar("ToolType")
FunctionInput = TypeVar("FunctionInput")
FunctionOutput = TypeVar("FunctionOutput", default=Any)

# Generic helper type variables for decorated callables and direct call_with_params() usage.
F = TypeVar("F", bound=collections_abc.Callable[..., Any])
ReturnType = TypeVar("ReturnType")

# Registered tools are either sync callables or async callables, and we keep that distinction in storage.
SyncToolCallable = collections_abc.Callable[..., FunctionOutput]
AsyncToolCallable = collections_abc.Callable[..., Awaitable[FunctionOutput]]
SyncToolEntry = Tuple["SyncToolMetadata", SyncToolCallable]
AsyncToolEntry = Tuple["AsyncToolMetadata", AsyncToolCallable]
ToolEntry = SyncToolEntry | AsyncToolEntry

# Hooks operate on raw parameters, resolved results, or the advertised LLM-visible tool entries.
PreToolCallHook = collections_abc.Callable[["ToolMetadata", RawParameters], RawParameters]
PostToolCallHook = collections_abc.Callable[["ToolMetadata", FunctionOutput], FunctionOutput]
OnToolListHook = collections_abc.Callable[[list[ToolEntry]], collections_abc.Iterable[ToolEntry]]


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


def _is_async_tool(
        candidate_tool: SyncToolCallable | AsyncToolCallable
) -> TypeGuard[AsyncToolCallable]:
    return inspect.iscoroutinefunction(candidate_tool)


def _is_awaitable_result(
        result: FunctionOutput | Awaitable[FunctionOutput]
) -> TypeGuard[Awaitable[FunctionOutput]]:
    return bool(inspect.isawaitable(result))


def _is_sync_result(
        result: FunctionOutput | Awaitable[FunctionOutput]
) -> TypeGuard[FunctionOutput]:
    return not bool(inspect.isawaitable(result))


class ToolCollection(collections_abc.Mapping[str, ToolEntry], abc.ABC, Generic[ToolType, FunctionInput, FunctionOutput]):
    def __init__(
            self,
            *tools: SyncToolCallable | AsyncToolCallable,
            pre_tool_call_hooks: Optional[collections_abc.Iterable[PreToolCallHook]] = None,
            post_tool_call_hooks: Optional[collections_abc.Iterable[PostToolCallHook]] = None,
            on_tool_list_hooks: Optional[collections_abc.Iterable[OnToolListHook]] = None,
    ):
        tools_by_name: Dict[str, ToolEntry] = dict()
        for candidate_tool in tools:
            metadata = tool_metadata(candidate_tool)
            if _is_async_tool(candidate_tool):
                tools_by_name[metadata.name] = (
                    AsyncToolMetadata(
                        name=metadata.name,
                        description=metadata.description,
                        model=metadata.model,
                    ),
                    candidate_tool
                )
            else:
                tools_by_name[metadata.name] = (
                    SyncToolMetadata(
                        name=metadata.name,
                        description=metadata.description,
                        model=metadata.model,
                    ),
                    candidate_tool
                )
        self._tools: Mapping[str, ToolEntry] = MappingProxyType(tools_by_name)
        self._pre_tool_call_hooks = tuple(pre_tool_call_hooks or ())
        self._post_tool_call_hooks = tuple(post_tool_call_hooks or ())
        self._on_tool_list_hooks = tuple(on_tool_list_hooks or ())

    def __call__(self, name, params: RawParameters) -> FunctionOutput | Awaitable[FunctionOutput]:
        metadata, callable_tool = self[name]
        hooked_params = self._apply_pre_tool_call_hooks(metadata, params)
        result = call_with_params(callable_tool, hooked_params)
        if _is_awaitable_result(result):
            return self._apply_post_tool_call_hooks_async(metadata, result)
        if _is_sync_result(result):
            return self._apply_post_tool_call_hooks(metadata, result)
        raise TypeError(f"Tool {name} returned an unsupported result type")

    def __getitem__(self, key: str) -> ToolEntry:
        return self._tools[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def _apply_pre_tool_call_hooks(self, metadata: "ToolMetadata", params: RawParameters) -> RawParameters:
        for hook in self._pre_tool_call_hooks:
            params = hook(metadata, params)
        return params

    def _apply_post_tool_call_hooks(self, metadata: "ToolMetadata", result: FunctionOutput) -> FunctionOutput:
        for hook in self._post_tool_call_hooks:
            result = hook(metadata, result)
        return result

    async def _apply_post_tool_call_hooks_async(
            self,
            metadata: "ToolMetadata",
            result: Awaitable[FunctionOutput]
    ) -> FunctionOutput:
        return self._apply_post_tool_call_hooks(metadata, await result)

    def _tools_for_llm(self) -> list[ToolEntry]:
        entries = list(self._tools.values())
        for hook in self._on_tool_list_hooks:
            entries = list(hook(entries))
        return entries

    def visible_tools(self) -> list["ToolMetadata"]:
        """
        Return metadata copies for tools currently visible to the LLM after tool-list hooks run.
        """
        return [dataclasses.replace(metadata) for metadata, _ in self._tools_for_llm()]

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

        async def run(
                _name: str,
                _params: RawParameters,
                _extras: Any
        ) -> tuple[str, FunctionOutputPayload[FunctionOutput]]:
            result = self(_name, _params)
            if _is_awaitable_result(result):
                return _name, FunctionOutputPayload(result=await result, extras=_extras)
            if _is_sync_result(result):
                return _name, FunctionOutputPayload(result=result, extras=_extras)
            raise TypeError(f"Tool {_name} returned an unsupported result type")

        lambdas: list[Awaitable[tuple[str, FunctionOutputPayload[FunctionOutput]]]] = []
        for function in functions:
            params = self.extract_parameters(function)
            lambdas.append(run(params.name, params.arguments, params.extras))
        return {name: payload for name, payload in await asyncio.gather(*lambdas)}

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


@dataclasses.dataclass
class SyncToolMetadata(ToolMetadata):
    is_async: Literal[False] = False


@dataclasses.dataclass
class AsyncToolMetadata(ToolMetadata):
    is_async: Literal[True] = True


def tool(
        name: Optional[str] = None,
        description: Optional[str] = None
) -> collections_abc.Callable[[F], F]:
    """
    Create a tool from a function
    :param name: The name of the tool, defaults to the function name
    :param description: The description of the tool, defaults to the function docstring
    """

    def add_tool_metadata(func: F) -> F:
        metadata_source: collections_abc.Callable[..., Any] = func

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


def tool_metadata(fn: collections_abc.Callable) -> ToolMetadata:
    try:
        return getattr(fn, "__tool_metadata__")
    except AttributeError as e:
        raise ToolNotDecoratedError(
            f"Function {fn.__name__} is not a tool, did you forget to add the @tool decorator?") from e


def call_with_params(fn: collections_abc.Callable[..., ReturnType], params: RawParameters) -> ReturnType:
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
