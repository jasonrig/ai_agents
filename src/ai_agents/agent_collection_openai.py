from copy import deepcopy
from typing import TypedDict, Literal, List

from openai.types.chat import ChatCompletionMessageToolCall

from src.ai_agents.agent import AgentCollection, input_schema, FunctionPayload


class OpenAITool(TypedDict):
    type: Literal["function"]
    function: dict


class AgentCollectionOpenAI(AgentCollection[OpenAITool, ChatCompletionMessageToolCall]):
    def tools(self, strict=True) -> List[OpenAITool]:
        def set_additional_properties_false(obj):
            if "properties" in obj:
                for key, value in obj["properties"].items():
                    set_additional_properties_false(value)
            if obj.get("type") == "object":
                obj["additionalProperties"] = False

        tools = list()
        for metadata, agent in self._agents.values():
            schema = input_schema(agent)
            del schema["description"]

            if strict:
                set_additional_properties_false(schema)

            tools.append({
                "type": "function",
                "function": {
                    "name": metadata.name,
                    "description": metadata.description,
                    "strict": strict,
                    "parameters": schema
                }
            })
        return deepcopy(tools)

    def extract_parameters(self, inp: ChatCompletionMessageToolCall) -> FunctionPayload:
        return FunctionPayload(inp.function.name, inp.function.arguments)
