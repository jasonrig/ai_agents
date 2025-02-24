from typing import Annotated
from unittest import TestCase

from pydantic import Field

from src.ai_agents.agent import agent


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


class TestAgentCollectionGemini(TestCase):
    def setUp(self):
        self.client = None
        try:
            from google import genai
            from google.genai import types
            from src.ai_agents.agent_collection_gemini import AgentCollectionGemini
        except ImportError:
            self.skipTest("Gemini SDK not installed")
        self.collection = AgentCollectionGemini(say_hello, say_goodbye)
        self.client = None
        try:
            self.client = genai.Client()
            self.model_config = types.GenerateContentConfig(
                tools=self.collection.tools(),
            )
        except ValueError:
            pass

    def test_tools(self):
        all_tools = sorted(self.collection.tools(), key=lambda x: x.function_declarations[0].name)
        fn_names = [tool.function_declarations[0].name for tool in all_tools]
        self.assertEqual(["say_goodbye", "say_hello"], fn_names)

        fn_schema = [tool.function_declarations[0].parameters.properties for tool in all_tools]
        self.assertEqual("name", list(fn_schema[0].keys())[0])
        self.assertEqual("name", list(fn_schema[1].keys())[0])
        self.assertEqual("Name of the person to farewell", fn_schema[0]["name"].description)
        self.assertEqual("Name of the person to greet", fn_schema[1]["name"].description)

    def test_gemini(self):
        if self.client is None:
            self.skipTest("No GenAI client available. Did you set the GOOGLE_API_KEY environment variable?")

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say hello to Alice",
            config=self.model_config,
        )
        result = self.collection.invoke_fn(response.function_calls[0])
        self.assertEqual({"say_hello": "Hello, Alice!"}, result)

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say bye to Alice",
            config=self.model_config,
        )
        result = self.collection.invoke_fn(response.function_calls[0])
        self.assertEqual({"say_goodbye": "Goodbye, Alice!"}, result)
