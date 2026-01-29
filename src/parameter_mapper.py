import inspect
from typing import Any

class ParameterMapper:
    PARAM_ALIASES = {
        "query": ["query", "search_query", "text", "prompt", "input"],
        "topic": ["topic", "subject", "theme", "prompt"],
        "goal": ["goal", "objective", "task", "prompt"],
        "plan": ["plan", "strategy", "approach"],
        "max_results": ["max_results", "limit", "count", "num_results"],
    }

    @staticmethod
    def map_parameters(tool_call_sig: inspect.Signature, provided: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        mapped = {}
        required_missing = []
        
        for param_name, param in tool_call_sig.parameters.items():
            if param_name == "self":
                continue
            
            if param_name in provided:
                mapped[param_name] = provided[param_name]
                continue
            
            if param_name in context:
                mapped[param_name] = context[param_name]
                continue
            
            for context_key, aliases in ParameterMapper.PARAM_ALIASES.items():
                if param_name in aliases and context_key in context:
                    mapped[param_name] = context[context_key]
                    break
            
            if param_name not in mapped:
                if param.default is not inspect.Parameter.empty:
                    mapped[param_name] = param.default
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    continue
                else:
                    required_missing.append(param_name)
        
        return mapped, required_missing
