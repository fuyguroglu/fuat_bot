"""
The Agent Loop - the heart of Fuat_bot.

This implements the classic agent pattern:
1. Receive user message
2. Send to LLM with available tools
3. If LLM wants to use tools → execute them → send results back → goto 2
4. If LLM responds with text → return to user

This is the same pattern used by OpenClaw, Claude Code, and most AI agents.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import ollama
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from rich.console import Console

from .config import settings
from .tools import TOOL_SCHEMAS, execute_tool

console = Console()


# =============================================================================
# Tool Schema Conversion
# =============================================================================

def _convert_tools_to_gemini_format(tool_schemas: list[dict]) -> list[dict]:
    """Convert Anthropic-style tool schemas to Gemini function declarations."""
    gemini_tools = []
    for tool in tool_schemas:
        gemini_tool = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"]
        }
        gemini_tools.append(gemini_tool)
    return gemini_tools


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are Fuat's personal AI assistant. You run on their local machine and can help with various tasks using the tools available to you.

## Your Capabilities
- Read and write files in the workspace
- Execute bash commands
- Search indexed PDF documents semantically
- Help with coding, writing, research, and general tasks

## Guidelines
- Be direct and helpful
- When you need to do something, use your tools rather than just explaining
- If a task requires multiple steps, work through them systematically
- Ask for clarification if a request is ambiguous
- Be careful with destructive operations (deleting files, etc.)

## Document Search
You have access to indexed PDF documents. When users ask questions about:
- University regulations, policies, or procedures
- Course materials or syllabi
- Any content from indexed PDFs

Use the search_documents tool to find relevant information. Always:
1. Cite your sources with document name and page number
2. Use the exact citation format: "According to [Document Title] (page X)"
3. If unsure whether information exists, search first rather than guessing

## Workspace
Your workspace is a directory where you can create and manage files. All file paths are relative to this workspace.

Current date: {date}
"""

# More directive system prompt for Ollama models (they need stronger guidance)
SYSTEM_PROMPT_OLLAMA = """You are Fuat's personal AI assistant with access to tools.

IMPORTANT: You MUST call the available tools to complete tasks. DO NOT explain how to use tools - actually call them.

Available tools:
- list_directory: List files in a directory
- read_file: Read file contents
- write_file: Write to a file
- bash: Execute commands
- search_documents: Search indexed PDF documents
- index_document: Index a PDF for searching

When asked to list files, read files, or write files: IMMEDIATELY call the appropriate tool.

When asked questions about documents, regulations, or policies: IMMEDIATELY call search_documents.

Example:
User: "List files in workspace"
You: [CALL list_directory tool with path="."]

User: "What is the attendance policy?"
You: [CALL search_documents tool with query="attendance policy"]

DO NOT respond with explanations of how to call tools. Call them directly.

Always cite sources with document name and page number when using search_documents.

Current date: {date}
"""


# =============================================================================
# Session Management
# =============================================================================

class Session:
    """
    Manages conversation history and persistence.
    
    Sessions are stored as JSONL files (one JSON object per line),
    which makes them easy to append to and read.
    """
    
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.messages: list[dict[str, Any]] = []
        self.session_file = settings.sessions_dir / f"{self.session_id}.jsonl"
        
        # Ensure sessions directory exists
        settings.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing session if it exists
        if self.session_file.exists():
            self._load()
    
    def _load(self):
        """Load messages from session file."""
        with open(self.session_file, "r") as f:
            for line in f:
                if line.strip():
                    self.messages.append(json.loads(line))
    
    def _append(self, message: dict[str, Any]):
        """Append a message to the session file."""
        with open(self.session_file, "a") as f:
            f.write(json.dumps(message) + "\n")
    
    def add_user_message(self, content: str):
        """Add a user message to the session."""
        message = {"role": "user", "content": content}
        self.messages.append(message)
        self._append(message)
    
    def add_assistant_message(self, content: Any):
        """Add an assistant message to the session."""
        message = {"role": "assistant", "content": content}
        self.messages.append(message)
        self._append(message)
    
    def add_tool_result(self, tool_use_id: str, result: Any):
        """Add a tool result to the session."""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result) if isinstance(result, dict) else str(result),
                }
            ],
        }
        self.messages.append(message)
        self._append(message)
    
    def get_messages_for_api(self) -> list[dict[str, Any]]:
        """Get messages formatted for the Anthropic API."""
        return self.messages.copy()


# =============================================================================
# Agent
# =============================================================================

class Agent:
    """
    The main agent class that orchestrates the conversation loop.
    """

    def __init__(self, session: Session | None = None):
        self.provider = settings.llm_provider
        self.session = session or Session()
        self.model = settings.model_name

        # Initialize the appropriate client
        if self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=settings.get_api_key())
        elif self.provider == "gemini":
            self.client = genai.Client(api_key=settings.get_api_key())
            # Convert tool schemas once for reuse
            self.gemini_tools = _convert_tools_to_gemini_format(TOOL_SCHEMAS)
        elif self.provider == "ollama":
            # Ollama uses its native Python library
            # No client initialization needed - use ollama module directly
            self.client = None  # Not needed for Ollama
            # Override model with Ollama-specific model
            self.model = settings.ollama_model
        elif self.provider == "openai":
            raise NotImplementedError("OpenAI provider not yet implemented")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Initialize memory manager
        if settings.memory_enabled:
            from .memory import MemoryManager
            self.memory_manager = MemoryManager(settings.memory_dir)
        else:
            self.memory_manager = None
    
    def _get_system_prompt(self) -> str:
        """Build the system prompt with current context and memories."""
        # Use more directive prompt for Ollama models
        if self.provider == "ollama":
            base_prompt = SYSTEM_PROMPT_OLLAMA.format(
                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
        else:
            base_prompt = SYSTEM_PROMPT.format(
                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )

        # Inject memories if enabled
        if self.memory_manager and settings.memory_injection_enabled:
            try:
                memories = self.memory_manager.get_relevant_memories_for_injection()
                if memories:
                    base_prompt += f"\n\n{memories}"
            except Exception as e:
                # Don't fail the whole request if memory loading fails
                console.print(f"[yellow]Warning: Failed to load memories: {e}[/yellow]")

        return base_prompt

    def _convert_messages_to_gemini(self) -> list[genai_types.Content]:
        """Convert session messages to Gemini's Content format."""
        gemini_messages = []

        for msg in self.session.messages:
            if msg["role"] == "user":
                # Handle different content types
                if isinstance(msg["content"], str):
                    gemini_messages.append(
                        genai_types.Content(role="user", parts=[genai_types.Part(text=msg["content"])])
                    )
                elif isinstance(msg["content"], list):
                    # Tool results
                    parts = []
                    for item in msg["content"]:
                        if item.get("type") == "tool_result":
                            # Create FunctionResponse for tool results
                            parts.append(
                                genai_types.Part(
                                    function_response=genai_types.FunctionResponse(
                                        name=item.get("tool_use_id", "unknown"),
                                        response={"result": item.get("content", "")}
                                    )
                                )
                            )
                    if parts:
                        gemini_messages.append(genai_types.Content(role="function", parts=parts))
            elif msg["role"] == "assistant":
                if isinstance(msg["content"], str):
                    gemini_messages.append(
                        genai_types.Content(role="model", parts=[genai_types.Part(text=msg["content"])])
                    )
                elif isinstance(msg["content"], list):
                    # Tool use blocks - convert to FunctionCall
                    parts = []
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "tool_use":
                            parts.append(
                                genai_types.Part(
                                    function_call=genai_types.FunctionCall(
                                        name=item.get("name", ""),
                                        args=item.get("args", {})
                                    )
                                )
                            )
                    if parts:
                        gemini_messages.append(genai_types.Content(role="model", parts=parts))

        return gemini_messages

    def _call_llm(self) -> Any:
        """Make an API call to the LLM (provider-specific)."""
        if self.provider == "anthropic":
            return self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._get_system_prompt(),
                tools=TOOL_SCHEMAS,
                messages=self.session.get_messages_for_api(),
            )
        elif self.provider == "gemini":
            # Convert messages to Gemini format
            contents = self._convert_messages_to_gemini()

            # Create tool configuration
            tools = genai_types.Tool(function_declarations=self.gemini_tools)

            # Create config with tools and disable automatic function calling
            config = genai_types.GenerateContentConfig(
                tools=[tools],
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
                system_instruction=self._get_system_prompt(),
            )

            # Call the API
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return response
        elif self.provider == "ollama":
            # Ollama uses its native Python library
            # Convert tool schemas to Ollama format
            ollama_tools = []
            for tool in TOOL_SCHEMAS:
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                })

            # Build messages with system prompt as first message
            messages = [{"role": "system", "content": self._get_system_prompt()}]

            # Convert session messages to Ollama format
            for msg in self.session.get_messages_for_api():
                if msg["role"] == "user":
                    # Check if it's a tool result (Anthropic format)
                    if isinstance(msg["content"], list):
                        # Convert tool results to Ollama format
                        for item in msg["content"]:
                            if item.get("type") == "tool_result":
                                messages.append({
                                    "role": "tool",
                                    "content": item.get("content", "")
                                })
                    else:
                        # Regular user message
                        messages.append(msg)
                elif msg["role"] == "assistant":
                    # Check if this is an Ollama tool call message
                    if isinstance(msg["content"], dict) and "ollama_tool_calls" in msg["content"]:
                        # This was a tool call - Ollama needs it reconstructed
                        # For now, just send empty content (Ollama will see tool results next)
                        messages.append({"role": "assistant", "content": ""})
                    else:
                        # Regular assistant message
                        messages.append(msg)

            # Call Ollama with native API
            return ollama.chat(
                model=self.model,
                messages=messages,
                tools=ollama_tools if ollama_tools else None,
                options={
                    "num_predict": 4096,  # max_tokens equivalent
                }
            )
        else:
            raise ValueError(f"Provider {self.provider} not supported")
    
    def _process_tool_calls(self, response: Any) -> bool:
        """
        Process any tool calls in the response (provider-specific).

        Returns True if there were tool calls (meaning we need another LLM turn).
        """
        if self.provider == "anthropic":
            return self._process_anthropic_tool_calls(response)
        elif self.provider == "gemini":
            return self._process_gemini_tool_calls(response)
        elif self.provider == "ollama":
            return self._process_ollama_tool_calls(response)
        else:
            return False

    def _process_anthropic_tool_calls(self, response: anthropic.types.Message) -> bool:
        """Process tool calls from Anthropic response."""
        tool_calls = [block for block in response.content if block.type == "tool_use"]

        if not tool_calls:
            return False

        # Store the assistant's response (with tool calls)
        self.session.add_assistant_message(
            [block.model_dump() for block in response.content]
        )

        # Execute each tool and collect results
        for tool_call in tool_calls:
            console.print(f"[dim]🔧 Using tool: {tool_call.name}[/dim]")

            # Execute the tool
            result = execute_tool(tool_call.name, tool_call.input)

            # Show result preview
            if "error" in result:
                console.print(f"[red]   ❌ Error: {result['error']}[/red]")
            else:
                console.print(f"[green]   ✓ Success[/green]")

            # Add result to session
            self.session.add_tool_result(tool_call.id, result)

        return True

    def _process_gemini_tool_calls(self, response: Any) -> bool:
        """Process tool calls from Gemini response."""
        # Check if response has candidates
        if not hasattr(response, 'candidates') or not response.candidates:
            return False

        candidate = response.candidates[0]

        # Check for function calls in the parts
        if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
            return False

        # Look for function calls in the response parts
        function_calls = []
        for part in candidate.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_calls.append(part.function_call)

        if not function_calls:
            return False

        # Store assistant message with tool calls
        tool_call_data = []
        for fc in function_calls:
            tool_call_data.append({
                "type": "tool_use",
                "name": fc.name,
                "args": dict(fc.args) if hasattr(fc.args, '__iter__') else {}
            })
        self.session.add_assistant_message(tool_call_data)

        # Execute each function call
        for fc in function_calls:
            console.print(f"[dim]🔧 Using tool: {fc.name}[/dim]")

            # Convert args to dict
            args = dict(fc.args) if hasattr(fc.args, '__iter__') else {}

            # Execute the tool
            result = execute_tool(fc.name, args)

            # Show result preview
            if "error" in result:
                console.print(f"[red]   ❌ Error: {result['error']}[/red]")
            else:
                console.print(f"[green]   ✓ Success[/green]")

            # Add result to session (use function name as tool_use_id)
            self.session.add_tool_result(fc.name, result)

        return True

    def _process_ollama_tool_calls(self, response: Any) -> bool:
        """Process tool calls from Ollama native response."""
        # Check if response has message with tool_calls
        if not hasattr(response, 'message'):
            return False

        message = response.message

        # Check for tool calls
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            return False

        # For Ollama, store the raw response message which includes tool_calls
        # Ollama expects assistant messages with tool_calls to have empty string content
        # We'll store a special marker that we can convert back when sending to Ollama
        tool_call_data = {
            "ollama_tool_calls": [
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in message.tool_calls
            ]
        }
        # Store as empty content with tool call metadata
        self.session.add_assistant_message(tool_call_data)

        # Execute each tool call
        for tc in message.tool_calls:
            console.print(f"[dim]🔧 Using tool: {tc.function.name}[/dim]")

            # Get arguments (already a dict in Ollama native format)
            args = tc.function.arguments

            # Execute the tool
            result = execute_tool(tc.function.name, args)

            # Show result preview
            if "error" in result:
                console.print(f"[red]   ❌ Error: {result['error']}[/red]")
            else:
                console.print(f"[green]   ✓ Success[/green]")

            # Add result to session
            # For Ollama, we'll use the function name as the ID
            self.session.add_tool_result(tc.function.name, result)

        return True

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.
        
        This is the main entry point - it handles the full agent loop:
        user message → LLM → (tool calls → results →)* final response
        """
        # Add user message to session
        self.session.add_user_message(user_message)
        
        # Agent loop: keep going until we get a final text response
        max_iterations = 10  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Call the LLM
            response = self._call_llm()
            
            # Process any tool calls
            has_tool_calls = self._process_tool_calls(response)
            
            if has_tool_calls:
                # Tool calls were made, loop back for another LLM turn
                continue

            # No tool calls - extract text response and we're done
            if self.provider == "anthropic":
                text_blocks = [
                    block.text for block in response.content if block.type == "text"
                ]
                final_response = "\n".join(text_blocks)
            elif self.provider == "gemini":
                # For Gemini, extract text from response
                if hasattr(response, 'text'):
                    final_response = response.text
                else:
                    final_response = str(response)
            elif self.provider == "ollama":
                # For Ollama native response, extract message content
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    final_response = response.message.content or ""
                else:
                    final_response = "No response from Ollama"
            else:
                final_response = "Unsupported provider response"

            # Store the final response
            self.session.add_assistant_message(final_response)

            return final_response
        
        return "I seem to be stuck in a loop. Let me stop here and you can try rephrasing your request."


# =============================================================================
# Convenience function
# =============================================================================

def create_agent(session_id: str | None = None) -> Agent:
    """Create a new agent, optionally resuming an existing session."""
    session = Session(session_id)
    return Agent(session)
