from typing import List

from google.genai.types import Tool, FunctionDeclaration, FunctionCall

from ai_agents.tool import ToolCollection, input_schema, FunctionInputPayload


class ToolCollectionGemini(ToolCollection[Tool, FunctionCall]):
    def tools(self) -> List[Tool]:
        tools = list()
        for metadata, callable_tool in self._tools.values():
            schema = input_schema(callable_tool)
            del schema["description"]
            tools.append(Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name=metadata.name,
                        description=metadata.description,
                        parameters=schema
                    )
                ]
            ))
        return tools

    def extract_parameters(self, inp: FunctionCall) -> FunctionInputPayload:
        return FunctionInputPayload(inp.name, inp.args)
