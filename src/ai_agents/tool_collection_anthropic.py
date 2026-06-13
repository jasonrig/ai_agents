from typing import List

from anthropic.types import ToolParam, ToolUseBlock

from ai_agents.tool import ToolCollection, input_schema, FunctionInputPayload


class ToolCollectionAnthropic(ToolCollection[ToolParam, ToolUseBlock]):
    def tools(self) -> List[ToolParam]:
        tools = list()
        for metadata, callable_tool in self._tools_for_llm():
            schema = input_schema(callable_tool)
            del schema["description"]
            tool: ToolParam = {
                "name": metadata.name,
                "description": metadata.description,
                "input_schema": schema
            }
            tools.append(tool)
        return tools

    def extract_parameters(self, inp: ToolUseBlock) -> FunctionInputPayload:
        return FunctionInputPayload(inp.name, inp.input, {"tool_call_id": inp.id})
