from copy import deepcopy
from typing import TypedDict, Literal, List

from openai.types.chat import ChatCompletionMessageToolCall

from ai_agents.agent import AgentCollection, input_schema, FunctionInputPayload


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

    def extract_parameters(self, inp: ChatCompletionMessageToolCall) -> FunctionInputPayload:
        return FunctionInputPayload(inp.function.name, inp.function.arguments, {"tool_call_id": inp.id})
