import asyncio
from dataclasses import dataclass
from typing import Annotated, Any, List
from unittest import TestCase

from pydantic import Field

from ai_agents.agent import agent, call_with_params, input_schema, agent_metadata, AgentCollection, \
    FunctionInputPayload, FunctionOutputPayload


# Plain agent with no annotations
@agent(description="A greeting agent")
def say_hello(name: str):
    return f"Hello, {name}!"


@agent(description="An async greeting agent")
async def say_hello_async(name: str):
    await asyncio.sleep(0.1)
    return f"Hello, {name}!"


# Agent with annotations
@agent(description="A greeting agent")
def say_hello_annotated(name: Annotated[str, Field(description="The name of the person to greet")]):
    return f"Hello, {name}!"


# Agent with annotations and default value
@agent(description="A greeting agent")
def say_hello_default(name: Annotated[str, Field(description="The name of the person to greet", default="Alice")]):
    return f"Hello, {name}!"


# Agent with custom name and description defined in the decorator
@agent(name="greeting", description="A greeting agent")
def say_hello_custom(name: str):
    return f"Hello, {name}!"


# Agent with description in the docstring
@agent()
def say_hello_docstring(name: str):
    """
    A greeting agent
    """
    return f"Hello, {name}!"


# Class-based agent
@agent(description="A greeting agent")
class SayHello:
    def __call__(self, name: str):
        return f"Hello, {name}!"


# Class with agent function
class SayHelloClassWithAgentFunction:
    @agent(description="A greeting agent")
    def say_hello(self, name: str):
        return f"Hello, {name}!"


agents = [
    say_hello,
    say_hello_annotated,
    say_hello_default,
    say_hello_custom,
    say_hello_docstring,
    SayHello(),
    SayHelloClassWithAgentFunction().say_hello
]


class TestAgent(TestCase):

    def test_async_marked_correctly(self):
        self.assertTrue(agent_metadata(say_hello_async).is_async)
        self.assertFalse(agent_metadata(say_hello).is_async)

    def test_call_directly(self):
        for a in agents:
            metadata = agent_metadata(a)
            with self.subTest(metadata.name):
                result = a(name="Alice")
                self.assertEqual("Hello, Alice!", result)

    def test_call_with_json_string(self):
        for a in agents:
            metadata = agent_metadata(a)
            with self.subTest(metadata.name):
                result = call_with_params(a, '{"name": "Alice"}')
                self.assertEqual("Hello, Alice!", result)

    def test_call_with_object(self):
        for a in agents:
            metadata = agent_metadata(a)
            with self.subTest(metadata.name):
                result = call_with_params(a, {"name": "Alice"})
                self.assertEqual("Hello, Alice!", result)

    def test_call_default_value(self):
        result = call_with_params(say_hello_default, {})
        self.assertEqual("Hello, Alice!", result)

    def test_schema_plain(self):
        schema = input_schema(say_hello)
        self.assertEqual({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            },
            "description": "A greeting agent",
            "required": ["name"]
        }, schema)

    def test_schema_annotated(self):
        schema = input_schema(say_hello_annotated)
        self.assertEqual({
            "type": "object",
            "properties": {
                "name": {
                    "description": "The name of the person to greet",
                    "type": "string"
                }
            },
            "description": "A greeting agent",
            "required": ["name"]
        }, schema)

    def test_schema_custom_name_and_description(self):
        schema = input_schema(say_hello_custom)
        self.assertEqual({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            },
            "description": "A greeting agent",
            "required": ["name"]
        }, schema)

    def test_schema_docstring(self):
        schema = input_schema(say_hello_docstring)
        self.assertEqual({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            },
            "description": "A greeting agent",
            "required": ["name"]
        }, schema)


class TestAgentCollection(TestCase):
    @dataclass
    class DummyPayload:
        name: str
        payload: Any

    class DummyCollection(AgentCollection[Any, DummyPayload, str]):
        def tools(self) -> List[Any]:
            pass

        def extract_parameters(self, inp) -> FunctionInputPayload:
            return FunctionInputPayload(name=inp.name, arguments=inp.payload)

    def setUp(self):
        self.collection = TestAgentCollection.DummyCollection(say_hello, say_hello_async)

    def test_invoke_fn(self):
        single_fn_result = self.collection.invoke_fn(
            TestAgentCollection.DummyPayload(name="say_hello", payload={"name": "Alice"}))
        self.assertDictEqual({"say_hello": FunctionOutputPayload(result='Hello, Alice!', extras=None)},
                             single_fn_result)

        single_async_fn_result = self.collection.invoke_fn(
            TestAgentCollection.DummyPayload(name="say_hello_async", payload={"name": "Alice"}))
        self.assertDictEqual({"say_hello_async": FunctionOutputPayload(result='Hello, Alice!', extras=None)},
                             single_async_fn_result)

        multi_fn_result = self.collection.invoke_fn(
            TestAgentCollection.DummyPayload(name="say_hello", payload={"name": "Alice"}),
            TestAgentCollection.DummyPayload(name="say_hello_async", payload={"name": "Bob"})
        )
        self.assertDictEqual({"say_hello": FunctionOutputPayload(result='Hello, Alice!', extras=None),
                              "say_hello_async": FunctionOutputPayload(result='Hello, Bob!', extras=None)},
                             multi_fn_result)
