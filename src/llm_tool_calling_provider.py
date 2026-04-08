#!/usr/bin/env python3
"""
Unified provider interface for LLM tool calling.

Implements OpenAI, Gemini, and DeepSeek adapters. Other providers can be
added by defining call_xxx_with_tools() and wiring into call_llm_with_tools().
"""

from typing import Any, List
import json
from types import SimpleNamespace


def call_openai_with_tools(messages: List[dict], tools: List[dict], model: str):
    """Call OpenAI API with tool calling enabled."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI package not installed. Run: pip install openai") from exc

    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.1,
    )


def _wrap_tool_calls(tool_calls: List[dict]):
    """Create a minimal OpenAI-like tool_calls structure."""
    wrapped = []
    for idx, tool_call in enumerate(tool_calls):
        wrapped.append(
            SimpleNamespace(
                id=f"tool_{idx}",
                function=SimpleNamespace(
                    name=tool_call["name"],
                    arguments=tool_call["arguments"],
                ),
            )
        )
    return wrapped


def _sanitize_gemini_schema(schema):
    """Remove schema fields not supported by Gemini function declarations."""
    if isinstance(schema, dict):
        sanitized = {}
        for key, value in schema.items():
            if key == "additionalProperties":
                continue
            sanitized[key] = _sanitize_gemini_schema(value)
        return sanitized
    if isinstance(schema, list):
        return [_sanitize_gemini_schema(item) for item in schema]
    return schema


def call_gemini_with_tools(messages: List[dict], tools: List[dict], model: str):
    """Call Gemini API with function calling enabled."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError("Google GenAI package not installed. Run: pip install google-genai") from exc

    import os
    import json

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    # Convert OpenAI-style tools to Gemini function declarations.
    func_decls = []
    for tool in tools:
        fn = tool.get("function", {})
        func_decls.append(
            types.FunctionDeclaration(
                name=fn.get("name"),
                description=fn.get("description", ""),
                parameters=_sanitize_gemini_schema(fn.get("parameters", {})),
            )
        )

    # Flatten messages to a single prompt for Gemini.
    def message_to_text(msg):
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "assistant")
            content = getattr(msg, "content", "")
        if content is None:
            content = ""
        return f"{role.upper()}: {content}"

    prompt = "\n\n".join([message_to_text(m) for m in messages])

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=func_decls)],
            temperature=0.1,
        ),
    )

    tool_calls = []
    text_parts = []

    if response.candidates:
        candidate = response.candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if parts is not None:
            for part in parts:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                if getattr(part, "function_call", None):
                    fn_call = part.function_call
                    tool_calls.append({
                        "name": fn_call.name,
                        "arguments": json.dumps(fn_call.args or {}),
                    })

    message = SimpleNamespace(
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=_wrap_tool_calls(tool_calls) if tool_calls else None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def call_deepseek_with_tools(messages: List[dict], tools: List[dict], model: str):
    """Call DeepSeek API (OpenAI-compatible) with tool calling enabled."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI package not installed. Run: pip install openai") from exc

    import os

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.1,
    )


def call_anthropic_with_tools(messages: List[dict], tools: List[dict], model: str):
    """Call Anthropic API with tool calling enabled."""
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("Anthropic package not installed. Run: pip install anthropic") from exc

    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Convert OpenAI-style tools to Anthropic tools.
    anthropic_tools = []
    for tool in tools:
        fn = tool.get("function", {})
        anthropic_tools.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {}),
            }
        )

    def _as_assistant_content(text_value, tool_calls_value):
        content_blocks = []
        if text_value:
            content_blocks.append({"type": "text", "text": text_value})
        for idx, tool_call in enumerate(tool_calls_value or []):
            tool_id = getattr(tool_call, "id", None) or tool_call.get("id") or f"tool_{idx}"
            fn = getattr(tool_call, "function", None) or tool_call.get("function") or {}
            name = getattr(fn, "name", None) or fn.get("name") or tool_call.get("name")
            raw_args = getattr(fn, "arguments", None) or fn.get("arguments") or tool_call.get("arguments") or "{}"
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                parsed_args = {}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": parsed_args,
                }
            )
        return content_blocks

    # Convert messages to Anthropic format.
    converted = []
    system_parts = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            if role == "system":
                if msg.get("content"):
                    system_parts.append(msg["content"])
                continue
            if role == "user":
                converted.append({"role": "user", "content": msg.get("content") or ""})
                continue
            if role == "assistant":
                assistant_content = _as_assistant_content(
                    msg.get("content") or "",
                    msg.get("tool_calls"),
                )
                converted.append({"role": "assistant", "content": assistant_content or ""})
                continue
            if role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", "tool_0"),
                                "content": msg.get("content") or "",
                            }
                        ],
                    }
                )
                continue
        else:
            # Handle SimpleNamespace-style assistant messages.
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None)
            if content is not None or tool_calls:
                assistant_content = _as_assistant_content(content or "", tool_calls)
                converted.append({"role": "assistant", "content": assistant_content or ""})

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=converted,
        tools=anthropic_tools,
        system="\n".join(system_parts) if system_parts else None,
        temperature=0.1,
    )

    tool_calls = []
    text_parts = []
    for part in response.content:
        if part.type == "text":
            text_parts.append(part.text)
        if part.type == "tool_use":
            tool_calls.append(
                {
                    "name": part.name,
                    "arguments": json.dumps(part.input or {}),
                    "id": part.id,
                }
            )

    wrapped = []
    for idx, tool_call in enumerate(tool_calls):
        wrapped.append(
            SimpleNamespace(
                id=tool_call.get("id", f"tool_{idx}"),
                function=SimpleNamespace(
                    name=tool_call["name"],
                    arguments=tool_call["arguments"],
                ),
            )
        )

    message = SimpleNamespace(
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=wrapped if wrapped else None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def call_qwen_with_tools(messages: List[dict], tools: List[dict], model: str):
    """Call Qwen API (OpenAI-compatible) with tool calling enabled."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI package not installed. Run: pip install openai") from exc

    import os

    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise ValueError("QWEN_API_KEY environment variable not set")

    base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.1,
    )


def call_llm_with_tools(provider: str, messages: List[dict], tools: List[dict], model: str) -> Any:
    """Dispatch tool-calling to the selected provider."""
    if provider == "openai":
        return call_openai_with_tools(messages, tools, model)
    if provider == "gemini":
        return call_gemini_with_tools(messages, tools, model)
    if provider == "deepseek":
        return call_deepseek_with_tools(messages, tools, model)
    if provider == "anthropic":
        return call_anthropic_with_tools(messages, tools, model)
    if provider == "qwen":
        return call_qwen_with_tools(messages, tools, model)
    raise ValueError(f"Unknown provider: {provider}")


def call_llm_text(provider: str, messages: List[dict], model: str) -> str:
    """Call the selected provider for a plain-text response (no tools)."""
    import os
    timeout_env = os.environ.get("LLM_REQUEST_TIMEOUT")
    timeout = float(timeout_env) if timeout_env else None
    if provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client_kwargs = {"api_key": api_key}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    if provider == "deepseek":
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        client_kwargs = {"api_key": api_key, "base_url": "https://api.deepseek.com"}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    if provider == "qwen":
        from openai import OpenAI
        api_key = os.environ.get("QWEN_API_KEY")
        if not api_key:
            raise ValueError("QWEN_API_KEY environment variable not set")
        base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        client_kwargs = {"api_key": api_key, "base_url": base_url}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    if provider == "anthropic":
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic") from exc
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
        system_parts = []
        converted = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system":
                if msg.get("content"):
                    system_parts.append(msg["content"])
                continue
            if role in {"user", "assistant"}:
                converted.append({"role": role, "content": msg.get("content") or ""})
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=converted,
            system="\n".join(system_parts) if system_parts else None,
            temperature=0.0,
        )
        return "\n".join([part.text for part in response.content if part.type == "text"])
    if provider == "gemini":
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("Google GenAI package not installed. Run: pip install google-genai") from exc
        import os
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        client = genai.Client(api_key=api_key)

        def message_to_text(msg):
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "assistant")
                content = getattr(msg, "content", "")
            if content is None:
                content = ""
            return f"{role.upper()}: {content}"

        prompt = "\n\n".join([message_to_text(m) for m in messages])
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )
        return response.text or ""
    raise ValueError(f"Unknown provider: {provider}")
