import abc
import asyncio
import dataclasses
import inspect
from asyncio import AbstractEventLoop
from typing import Callable, get_type_hints, get_args, Optional, Type, Union, List, Annotated, TypeVar, Generic, Any, \
    Dict, Tuple

import jsonref
from jsonref import JsonRef
from pydantic import create_model, BaseModel, Field

RawParameters = Union[str, dict]
ToolType = TypeVar("ToolType")
FunctionInput = TypeVar("FunctionInput")
FunctionOutput = TypeVar("FunctionOutput", default=Any)


@dataclasses.dataclass
class FunctionInputPayload:
    name: str
    arguments: RawParameters
    extras: Optional[Any] = None


@dataclasses.dataclass
class FunctionOutputPayload:
    result: FunctionOutput
    extras: Optional[Any] = None


def _use_or_replace_loop(current_loop: Optional[AbstractEventLoop]) -> AbstractEventLoop:
    if current_loop is None or not current_loop.is_running():
        return asyncio.new_event_loop()
    return current_loop


class AgentCollection(abc.ABC, Generic[ToolType, FunctionInput, FunctionOutput]):
    def __init__(self, *agents: Callable):
        self._agents: Dict[str, Tuple[AgentMetadata, Callable]] = dict()
        for a in agents:
            metadata = agent_metadata(a)
            self._agents[metadata.name] = (metadata, a)

    def __call__(self, name, params: RawParameters) -> FunctionOutput:
        _, agent = self._agents[name]
        return call_with_params(agent, params)

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

    async def invoke_fn_async(self, *functions: FunctionInput) -> Dict[str, FunctionOutputPayload]:
        assert len(functions) > 0, "At least one function must be provided"

        async def run_as_async(_name, _params, _extras):
            return _name, FunctionOutputPayload(result=self(_name, _params), extras=_extras)

        async def run(_name, _params, _extras):
            return _name, FunctionOutputPayload(result=await self(_name, _params), extras=_extras)

        lambdas = list()
        for function in functions:
            params = self.extract_parameters(function)
            metadata, _ = self._agents[params.name]
            if not metadata.is_async:
                lambdas.append(run_as_async(params.name, params.arguments, params.extras))
            else:
                lambdas.append(run(params.name, params.arguments, params.extras))
        return dict(await asyncio.gather(*lambdas))

    def invoke_fn(self, *functions: FunctionInput, loop: Optional[AbstractEventLoop] = None) -> Dict[
        str, FunctionOutputPayload]:
        with asyncio.Runner(loop_factory=lambda: _use_or_replace_loop(loop)) as runner:
            return runner.run(self.invoke_fn_async(*functions))


@dataclasses.dataclass
class AgentMetadata:
    name: str
    description: str
    model: Type[BaseModel]
    is_async: bool


def agent(
        name: Optional[str] = None,
        description: Optional[str] = None
):
    """
    Create an agent from a function
    :param name: The name of the agent, defaults to the function name
    :param description: The description of the agent, defaults to the function docstring
    """

    def add_agent_metadata(func: Callable):

        # "original_func" is the thing to which we will attach metadata,
        # "func" is the thing we will infer metadata from
        original_func = func

        fn_name = name or func.__name__
        fn_description = description or func.__doc__
        assert fn_description is not None, f"Function {fn_name} must have a description"
        fn_description = fn_description.strip()

        if not inspect.isfunction(func) and hasattr(func, "__call__"):
            func = func.__call__

        hints = get_type_hints(func, include_extras=True)

        # Handle any missing annotations
        for param_name, param_annotation in hints.items():
            args = get_args(param_annotation)
            if not args:
                hints[param_name] = Annotated[param_annotation, Field()]

        # The return type is not needed
        if "return" in hints:
            del hints["return"]

        # Attach the metadata to the function or class
        original_func.__agent_metadata__ = AgentMetadata(
            name=fn_name,
            description=fn_description,
            model=create_model(fn_name, __doc__=fn_description, **hints),
            is_async=inspect.iscoroutinefunction(func)
        )
        return original_func

    return add_agent_metadata


def agent_metadata(fn: Callable) -> AgentMetadata:
    assert hasattr(fn, "__agent_metadata__") and isinstance(fn.__agent_metadata__,
                                                            AgentMetadata), "Function is not an agent, did you forget to add the @agent decorator?"
    return getattr(fn, "__agent_metadata__")


def call_with_params(fn, params: RawParameters):
    """
    Call the function with parameters as either a JSON string or a dictionary
    """
    metadata = agent_metadata(fn)
    params = metadata.model.model_validate_json(params) if isinstance(params, str) else metadata.model.model_validate(
        params)
    params = {k: getattr(params, k) for k in params.model_fields.keys()}
    return fn(**params)


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

    schema = jsonref.replace_refs(agent_metadata(fn).model.model_json_schema(), lazy_load=False)
    if "$defs" in schema:
        del schema["$defs"]
    remove_title_recursive(schema)
    return schema
