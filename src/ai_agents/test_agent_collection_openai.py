from typing import Annotated
from unittest import TestCase

from openai import OpenAIError
from pydantic import Field, BaseModel

from ai_agents.agent import agent


class Name(BaseModel):
    first_name: str
    last_name: str


@agent()
def say_hello(name: Annotated[Name, Field(description="Name of the person to greet")]):
    """
    Use this function to say hello to someone
    """
    return f"Hello, {name.first_name} {name.last_name}!"


@agent()
def say_goodbye(name: Annotated[Name, Field(description="Name of the person to farewell")]):
    """
    Use this function to say goodbye to someone
    """
    return f"Goodbye, {name.first_name} {name.last_name}!"


class TestAgentCollectionOpenAI(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxDiff = None

    def setUp(self):
        try:
            import openai
            from ai_agents.agent_collection_openai import AgentCollectionOpenAI
        except ImportError:
            self.skipTest("OpenAI SDK not installed")
        self.client = None
        try:
            self.client = openai.Client()
        except OpenAIError:
            self.client = None
        self.collection = AgentCollectionOpenAI(say_hello, say_goodbye)

    def test_tools_non_strict(self):
        all_tools = sorted(self.collection.tools(strict=False), key=lambda x: x["function"]["name"])
        self.assertEqual(len(all_tools), 2)
        self.assertDictEqual({
            "type": "function",
            "function": {
                "name": "say_goodbye",
                "description": "Use this function to say goodbye to someone",
                "strict": False,
                "parameters": {
                    "properties": {
                        "name": {
                            "properties": {
                                "first_name": {
                                    "type": "string"
                                },
                                "last_name": {
                                    "type": "string"
                                }
                            },
                            "type": "object",
                            "required": ["first_name", "last_name"]
                        }
                    },
                    "type": "object",
                    "required": ["name"],
                }
            }
        }, all_tools[0])
        self.assertDictEqual({
            "type": "function",
            "function": {
                "name": "say_hello",
                "description": "Use this function to say hello to someone",
                "strict": False,
                "parameters": {
                    "properties": {
                        "name": {
                            "properties": {
                                "first_name": {
                                    "type": "string"
                                },
                                "last_name": {
                                    "type": "string"
                                }
                            },
                            "type": "object",
                            "required": ["first_name", "last_name"]
                        }
                    },
                    "type": "object",
                    "required": ["name"],
                }
            }
        }, all_tools[1])

    def test_tools_strict(self):
        all_tools = sorted(self.collection.tools(strict=True), key=lambda x: x["function"]["name"])
        self.assertEqual(len(all_tools), 2)
        self.assertDictEqual({
            "type": "function",
            "function": {
                "name": "say_goodbye",
                "description": "Use this function to say goodbye to someone",
                "strict": True,
                "parameters": {
                    "properties": {
                        "name": {
                            "properties": {
                                "first_name": {
                                    "type": "string"
                                },
                                "last_name": {
                                    "type": "string"
                                }
                            },
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["first_name", "last_name"]
                        }
                    },
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name"],
                }
            }
        }, all_tools[0])
        self.assertDictEqual({
            "type": "function",
            "function": {
                "name": "say_hello",
                "description": "Use this function to say hello to someone",
                "strict": True,
                "parameters": {
                    "properties": {
                        "name": {
                            "properties": {
                                "first_name": {
                                    "type": "string"
                                },
                                "last_name": {
                                    "type": "string"
                                }
                            },
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["first_name", "last_name"]
                        }
                    },
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name"],
                }
            }
        }, all_tools[1])

    def test_openai_non_strict(self):
        if self.client is None:
            self.skipTest("No OpenAI client available. Did you set the OPENAI_API_KEY environment variable?")
        tools = self.collection.tools(strict=False)
        message = self.client.chat.completions.create(
            model="gpt-4o",
            tools=tools,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Say hello to Alice Peterson"
                }
            ]
        )
        fn_output = self.collection.invoke_fn(message.choices[0].message.tool_calls[0])
        self.assertEqual({"say_hello": "Hello, Alice Peterson!"}, fn_output)

        message = self.client.chat.completions.create(
            model="gpt-4o",
            tools=tools,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Say bye to Alice Peterson"
                }
            ]
        )
        fn_output = self.collection.invoke_fn(message.choices[0].message.tool_calls[0])
        self.assertEqual({"say_goodbye": "Goodbye, Alice Peterson!"}, fn_output)

    def test_openai_strict(self):
        if self.client is None:
            self.skipTest("No OpenAI client available. Did you set the OPENAI_API_KEY environment variable?")
        tools = self.collection.tools(strict=True)
        message = self.client.chat.completions.create(
            model="gpt-4o",
            tools=tools,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Say hello to Alice Peterson"
                }
            ]
        )
        fn_output = self.collection.invoke_fn(message.choices[0].message.tool_calls[0])
        self.assertEqual({"say_hello": "Hello, Alice Peterson!"}, fn_output)

        message = self.client.chat.completions.create(
            model="gpt-4o-mini",
            tools=tools,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Say bye to Alice Peterson"
                }
            ]
        )
        fn_output = self.collection.invoke_fn(message.choices[0].message.tool_calls[0])
        self.assertEqual({"say_goodbye": "Goodbye, Alice Peterson!"}, fn_output)
