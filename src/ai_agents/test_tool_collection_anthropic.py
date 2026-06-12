import os
from types import SimpleNamespace
from typing import Annotated
from unittest import TestCase, skipIf

from pydantic import Field

from ai_agents.tool import tool, FunctionOutputPayload

skip_anthropic_live_tests = skipIf(
    os.environ.get("AI_AGENTS_SKIP_ANTHROPIC_LIVE_TESTS"),
    "AI_AGENTS_SKIP_ANTHROPIC_LIVE_TESTS is set",
)


@tool()
def say_hello(name: Annotated[str, Field(description="Name of the person to greet")]):
    """
    Use this function to say hello to someone
    """
    return f"Hello, {name}!"


@tool()
def say_goodbye(name: Annotated[str, Field(description="Name of the person to farewell")]):
    """
    Use this function to say goodbye to someone
    """
    return f"Goodbye, {name}!"


class TestToolCollectionAnthropic(TestCase):
    def setUp(self):
        try:
            import anthropic
            from anthropic.types import ToolUseBlock
            from ai_agents.tool_collection_anthropic import ToolCollectionAnthropic
        except ImportError:
            self.skipTest("Anthropic SDK not installed")
        self.anthropic = anthropic
        self.ToolUseBlock = ToolUseBlock
        self.collection = ToolCollectionAnthropic(say_hello, say_goodbye)

    def test_tools(self):
        all_tools = sorted(self.collection.tools(), key=lambda x: x["name"])
        self.assertEqual(len(all_tools), 2)
        self.assertDictEqual({'description': 'Use this function to say goodbye to someone', 'input_schema': {
            'properties': {'name': {'description': 'Name of the person to farewell', 'type': 'string'}},
            'required': ['name'], 'type': 'object'}, 'name': 'say_goodbye'}, all_tools[0])
        self.assertDictEqual({'description': 'Use this function to say hello to someone', 'input_schema': {
            'properties': {'name': {'description': 'Name of the person to greet', 'type': 'string'}},
            'required': ['name'], 'type': 'object'}, 'name': 'say_hello'}, all_tools[1])

        self.assertEqual("Hello, Alice!", self.collection("say_hello", {"name": "Alice"}))
        self.assertEqual("Goodbye, Alice!", self.collection("say_goodbye", {"name": "Alice"}))

    def test_invoke_fn_with_tool_use_payload(self):
        tool_use = SimpleNamespace(id="toolu_123", name="say_hello", input={"name": "Alice"})

        result = self.collection.invoke_fn(tool_use)

        self.assertEqual({
            "say_hello": FunctionOutputPayload(
                result="Hello, Alice!",
                extras={"tool_call_id": "toolu_123"},
            )
        }, result)

    @skip_anthropic_live_tests
    def test_anthropic(self):
        client = self.anthropic.Anthropic()
        tools = self.collection.tools()
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            tools=tools,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Say hello to Alice"
                }
            ]
        )
        fn_output = self.collection.invoke_fn(next(filter(lambda x: isinstance(x, self.ToolUseBlock), message.content)))
        self.assertIsNotNone(fn_output["say_hello"].extras["tool_call_id"])
        self.assertEqual("Hello, Alice!", fn_output["say_hello"].result)

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            tools=tools,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Say bye to Alice"
                }
            ]
        )
        fn_output = self.collection.invoke_fn(next(filter(lambda x: isinstance(x, self.ToolUseBlock), message.content)))
        self.assertIsNotNone(fn_output["say_goodbye"].extras["tool_call_id"])
        self.assertEqual("Goodbye, Alice!", fn_output["say_goodbye"].result)
