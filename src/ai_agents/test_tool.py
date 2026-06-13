import asyncio
from dataclasses import dataclass
from typing import Annotated, Any, List
from unittest import TestCase

from pydantic import Field

from ai_agents.tool import tool, call_with_params, input_schema, tool_metadata, ToolCollection, \
    FunctionInputPayload, FunctionOutputPayload, ToolMetadata, ToolNotDecoratedError


# Plain tool with no annotations
@tool(description="A greeting tool")
def say_hello(name: str):
    return f"Hello, {name}!"


@tool(description="An async greeting tool")
async def say_hello_async(name: str):
    await asyncio.sleep(0.1)
    return f"Hello, {name}!"


# Tool with annotations
@tool(description="A greeting tool")
def say_hello_annotated(name: Annotated[str, Field(description="The name of the person to greet")]):
    return f"Hello, {name}!"


# Tool with annotations and default value
@tool(description="A greeting tool")
def say_hello_default(name: Annotated[str, Field(description="The name of the person to greet", default="Alice")]):
    return f"Hello, {name}!"


# Tool with custom name and description defined in the decorator
@tool(name="greeting", description="A greeting tool")
def say_hello_custom(name: str):
    return f"Hello, {name}!"


# Tool with description in the docstring
@tool()
def say_hello_docstring(name: str):
    """
    A greeting tool
    """
    return f"Hello, {name}!"


def unannotated_function(name: str):
    return f"Hello, {name}!"


# Class-based tool
@tool(description="A greeting tool")
class SayHello:
    def __call__(self, name: str):
        return f"Hello, {name}!"


# Class with tool function
class SayHelloClassWithToolFunction:
    @tool(description="A greeting tool")
    def say_hello(self, name: str):
        return f"Hello, {name}!"


tools = [
    say_hello,
    say_hello_annotated,
    say_hello_default,
    say_hello_custom,
    say_hello_docstring,
    SayHello(),
    SayHelloClassWithToolFunction().say_hello
]


class TestTool(TestCase):

    def test_unannotated_function_throw_error(self):
        with self.assertRaises(ToolNotDecoratedError):
            tool_metadata(unannotated_function)

    def test_async_marked_correctly(self):
        self.assertTrue(tool_metadata(say_hello_async).is_async)
        self.assertFalse(tool_metadata(say_hello).is_async)

    def test_call_directly(self):
        for candidate_tool in tools:
            metadata = tool_metadata(candidate_tool)
            with self.subTest(metadata.name):
                result = candidate_tool(name="Alice")
                self.assertEqual("Hello, Alice!", result)

    def test_call_with_json_string(self):
        for candidate_tool in tools:
            metadata = tool_metadata(candidate_tool)
            with self.subTest(metadata.name):
                result = call_with_params(candidate_tool, '{"name": "Alice"}')
                self.assertEqual("Hello, Alice!", result)

    def test_call_with_object(self):
        for candidate_tool in tools:
            metadata = tool_metadata(candidate_tool)
            with self.subTest(metadata.name):
                result = call_with_params(candidate_tool, {"name": "Alice"})
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
            "description": "A greeting tool",
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
            "description": "A greeting tool",
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
            "description": "A greeting tool",
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
            "description": "A greeting tool",
            "required": ["name"]
        }, schema)


class TestToolCollection(TestCase):
    @dataclass
    class DummyPayload:
        name: str
        payload: Any

    class DummyCollection(ToolCollection[Any, DummyPayload, str]):
        def tools(self) -> List[Any]:
            return [metadata.name for metadata, _ in self._tools_for_llm()]

        def extract_parameters(self, inp) -> FunctionInputPayload:
            return FunctionInputPayload(name=inp.name, arguments=inp.payload)

    def setUp(self):
        self.collection = TestToolCollection.DummyCollection(say_hello, say_hello_async)

    def test_invoke_fn(self):
        single_fn_result = self.collection.invoke_fn(
            TestToolCollection.DummyPayload(name="say_hello", payload={"name": "Alice"}))
        self.assertDictEqual({"say_hello": FunctionOutputPayload(result='Hello, Alice!', extras=None)},
                             single_fn_result)

        single_async_fn_result = self.collection.invoke_fn(
            TestToolCollection.DummyPayload(name="say_hello_async", payload={"name": "Alice"}))
        self.assertDictEqual({"say_hello_async": FunctionOutputPayload(result='Hello, Alice!', extras=None)},
                             single_async_fn_result)

        multi_fn_result = self.collection.invoke_fn(
            TestToolCollection.DummyPayload(name="say_hello", payload={"name": "Alice"}),
            TestToolCollection.DummyPayload(name="say_hello_async", payload={"name": "Bob"})
        )
        self.assertDictEqual({"say_hello": FunctionOutputPayload(result='Hello, Alice!', extras=None),
                              "say_hello_async": FunctionOutputPayload(result='Hello, Bob!', extras=None)},
                             multi_fn_result)

    def test_mapping_lookup(self):
        metadata, callable_tool = self.collection["say_hello"]
        self.assertEqual("say_hello", metadata.name)
        self.assertIs(say_hello, callable_tool)

    def test_mapping_views(self):
        self.assertEqual(["say_hello", "say_hello_async"], list(self.collection.keys()))
        self.assertEqual(["say_hello", "say_hello_async"], [metadata.name for metadata, _ in self.collection.values()])
        self.assertEqual(
            [("say_hello", "say_hello"), ("say_hello_async", "say_hello_async")],
            [(name, metadata.name) for name, (metadata, _) in self.collection.items()]
        )

    def test_mapping_is_read_only(self):
        with self.assertRaises(TypeError):
            self.collection["say_goodbye"] = (tool_metadata(say_hello), say_hello)  # type: ignore[index]

    def test_pre_tool_call_hook_can_modify_arguments(self):
        collection = TestToolCollection.DummyCollection(
            say_hello,
            pre_tool_call_hooks=[
                lambda metadata, params: {"name": "Bob"} if metadata.name == "say_hello" else params
            ],
        )

        result = collection.invoke_fn(
            TestToolCollection.DummyPayload(name="say_hello", payload={"name": "Alice"}))

        self.assertEqual("Hello, Bob!", result["say_hello"].result)

    def test_post_tool_call_hook_can_modify_result(self):
        collection = TestToolCollection.DummyCollection(
            say_hello,
            post_tool_call_hooks=[
                lambda metadata, result: f"{result} Hooked" if metadata.name == "say_hello" else result
            ],
        )

        result = collection.invoke_fn(
            TestToolCollection.DummyPayload(name="say_hello", payload={"name": "Alice"}))

        self.assertEqual("Hello, Alice! Hooked", result["say_hello"].result)

    def test_post_tool_call_hook_applies_to_async_tools(self):
        collection = TestToolCollection.DummyCollection(
            say_hello_async,
            post_tool_call_hooks=[
                lambda metadata, result: f"{result} Hooked" if metadata.name == "say_hello_async" else result
            ],
        )

        result = collection.invoke_fn(
            TestToolCollection.DummyPayload(name="say_hello_async", payload={"name": "Alice"}))

        self.assertEqual("Hello, Alice! Hooked", result["say_hello_async"].result)

    def test_on_tool_list_hook_can_filter_tools(self):
        collection = TestToolCollection.DummyCollection(
            say_hello,
            say_hello_async,
            on_tool_list_hooks=[
                lambda entries: [entry for entry in entries if entry[0].name != "say_hello_async"]
            ],
        )

        self.assertEqual(["say_hello"], collection.tools())
        self.assertEqual(["say_hello"], [metadata.name for metadata in collection.visible_tools()])

    def test_on_tool_list_hook_can_reorder_tools(self):
        collection = TestToolCollection.DummyCollection(
            say_hello,
            say_hello_async,
            on_tool_list_hooks=[
                lambda entries: reversed(entries)
            ],
        )

        self.assertEqual(["say_hello_async", "say_hello"], collection.tools())
        self.assertEqual(["say_hello_async", "say_hello"], [metadata.name for metadata in collection.visible_tools()])

    def test_visible_tools_exposes_metadata(self):
        collection = TestToolCollection.DummyCollection(say_hello, say_hello_async)

        visible_tools = collection.visible_tools()

        self.assertTrue(all(isinstance(metadata, ToolMetadata) for metadata in visible_tools))
        self.assertEqual(["say_hello", "say_hello_async"], [metadata.name for metadata in visible_tools])
        self.assertEqual(["A greeting tool", "An async greeting tool"], [metadata.description for metadata in visible_tools])
        self.assertEqual([False, True], [metadata.is_async for metadata in visible_tools])
        self.assertIs(tool_metadata(say_hello).model, visible_tools[0].model)

    def test_visible_tools_returns_detached_metadata(self):
        collection = TestToolCollection.DummyCollection(say_hello, say_hello_async)

        visible_tools = collection.visible_tools()
        visible_tools[0].name = "renamed"
        visible_tools[0].description = "Changed description"

        self.assertEqual(["say_hello", "say_hello_async"], collection.tools())
        self.assertEqual(["say_hello", "say_hello_async"], [metadata.name for metadata in collection.visible_tools()])

    def test_tools_for_llm_is_filtered_snapshot(self):
        collection = TestToolCollection.DummyCollection(
            say_hello,
            say_hello_async,
            on_tool_list_hooks=[
                lambda entries: [entry for entry in entries if entry[0].name != "say_hello"]
            ],
        )

        entries = collection._tools_for_llm()

        self.assertEqual(["say_hello_async"], [metadata.name for metadata, _ in entries])
