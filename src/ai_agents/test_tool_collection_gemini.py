import os
from types import SimpleNamespace
from typing import Annotated
from unittest import TestCase

from pydantic import Field

from ai_agents.tool import tool, FunctionOutputPayload


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


class TestToolCollectionGemini(TestCase):
    def setUp(self):
        try:
            from google import genai
            from google.genai import types
            from ai_agents.tool_collection_gemini import ToolCollectionGemini
        except ImportError:
            self.skipTest("Gemini SDK not installed")
        self.genai = genai
        self.types = types
        self.collection = ToolCollectionGemini(say_hello, say_goodbye)

    def test_tools(self):
        all_tools = sorted(self.collection.tools(), key=lambda x: x.function_declarations[0].name)
        fn_names = [tool.function_declarations[0].name for tool in all_tools]
        self.assertEqual(["say_goodbye", "say_hello"], fn_names)

        fn_schema = [tool.function_declarations[0].parameters.properties for tool in all_tools]
        self.assertEqual("name", list(fn_schema[0].keys())[0])
        self.assertEqual("name", list(fn_schema[1].keys())[0])
        self.assertEqual("Name of the person to farewell", fn_schema[0]["name"].description)
        self.assertEqual("Name of the person to greet", fn_schema[1]["name"].description)

    def test_invoke_fn_with_function_call_payload(self):
        function_call = SimpleNamespace(name="say_hello", args={"name": "Alice"})

        result = self.collection.invoke_fn(function_call)

        self.assertEqual({"say_hello": FunctionOutputPayload(result="Hello, Alice!", extras=None)}, result)

    def test_gemini(self):
        if os.environ.get("AI_AGENTS_SKIP_GEMINI_LIVE_TESTS"):
            self.skipTest("AI_AGENTS_SKIP_GEMINI_LIVE_TESTS is set")

        client = self.genai.Client()
        model_config = self.types.GenerateContentConfig(
            tools=self.collection.tools(),
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say hello to Alice",
            config=model_config,
        )
        result = self.collection.invoke_fn(response.function_calls[0])
        self.assertEqual({"say_hello": FunctionOutputPayload(result="Hello, Alice!", extras=None)}, result)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say bye to Alice",
            config=model_config,
        )
        result = self.collection.invoke_fn(response.function_calls[0])
        self.assertEqual({"say_goodbye": FunctionOutputPayload(result="Goodbye, Alice!", extras=None)}, result)
