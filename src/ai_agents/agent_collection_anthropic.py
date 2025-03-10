from typing import List

from anthropic.types import ToolParam, ToolUseBlock

from ai_agents.agent import AgentCollection, input_schema, FunctionInputPayload


class AgentCollectionAnthropic(AgentCollection[ToolParam, ToolUseBlock]):
    def tools(self) -> List[ToolParam]:
        tools = list()
        for metadata, agent in self._agents.values():
            schema = input_schema(agent)
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
