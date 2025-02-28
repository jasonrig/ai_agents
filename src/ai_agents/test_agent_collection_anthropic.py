from typing import Annotated
from unittest import TestCase

from anthropic.types import ToolUseBlock
from pydantic import Field

from ai_agents.agent import agent


@agent()
def say_hello(name: Annotated[str, Field(description="Name of the person to greet")]):
    """
    Use this function to say hello to someone
    """
    return f"Hello, {name}!"


@agent()
def say_goodbye(name: Annotated[str, Field(description="Name of the person to farewell")]):
    """
    Use this function to say goodbye to someone
    """
    return f"Goodbye, {name}!"


class TestAgentCollectionAnthropic(TestCase):
    def setUp(self):
        try:
            import anthropic
            from ai_agents.agent_collection_anthropic import AgentCollectionAnthropic
        except ImportError:
            self.skipTest("Anthropic SDK not installed")
        self.client = anthropic.Anthropic()
        self.collection = AgentCollectionAnthropic(say_hello, say_goodbye)

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

    def test_anthropic(self):
        tools = self.collection.tools()
        try:
            message = self.client.messages.create(
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
        except TypeError:
            self.skipTest("Call to Anthropic API failed. Check your API key.")
        fn_output = self.collection.invoke_fn(next(filter(lambda x: isinstance(x, ToolUseBlock), message.content)))
        self.assertIsNotNone(fn_output["say_hello"].extras["tool_call_id"])
        self.assertEqual("Hello, Alice!", fn_output["say_hello"].result)

        message = self.client.messages.create(
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
        fn_output = self.collection.invoke_fn(next(filter(lambda x: isinstance(x, ToolUseBlock), message.content)))
        self.assertIsNotNone(fn_output["say_goodbye"].extras["tool_call_id"])
        self.assertEqual("Goodbye, Alice!", fn_output["say_goodbye"].result)
