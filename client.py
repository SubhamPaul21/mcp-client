import asyncio
import os
import sys
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from fastmcp import Client
from groq import Groq

load_dotenv()  # Load environment variables from .env
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stderr)


class MCPClient:
    def __init__(self):
        self.client: Client | None = None
        self.exit_stack = AsyncExitStack()
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    async def connect_and_chat(self, server_url: str):
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        self.client = Client(server_url)
        async with self.client:
            tools = await self.client.list_tools()
            print("\nConnected to server with tools:", [tool.name for tool in tools])
            await self.chat_loop()

    async def process_query(self, query: str) -> str:
        if not self.client:
            raise RuntimeError("Client not connected")

        messages = [{"role": "user", "content": query}]

        # Fetch tools from server and prepare for Groq
        tools = await self.client.list_tools()
        groq_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,  # Adapt if schema shape differs
                },
            }
            for tool in tools
        ]

        # Initial Groq API call
        response = await asyncio.to_thread(
            self.groq.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=groq_tools,
            max_tokens=1000,
        )

        tool_results = []
        final_text = []

        content_data = response.choices[0].message.content
        logging.info(f"content_data: {response.choices}")

        # Guard: content_data could be None or a string instead of list
        if content_data is None:
            content_data = []
        elif isinstance(content_data, str):
            content_data = [content_data]

        # Processing response contents (support strings and tool calls)
        for content in content_data:
            if isinstance(content, str):
                final_text.append(content)
            elif isinstance(content, dict) and content.get("type") == "tool_use":
                tool_name = content.get("name")
                tool_args = content.get("input")

                # Call MCP tool
                result = await self.client.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})

                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if "text" in content and content["text"]:
                    messages.append({"role": "assistant", "content": content["text"]})
                messages.append({"role": "user", "content": result.content})

                # Get updated response from Groq
                response = await asyncio.to_thread(
                    self.groq.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=groq_tools,
                    max_tokens=1000,
                )

                # Repeat normalization for new response
                new_content_data = response.choices[0].message.content
                if new_content_data is None:
                    new_content_data = []
                elif isinstance(new_content_data, str):
                    new_content_data = [new_content_data]

                final_text.extend(new_content_data)

        return "".join(final_text).strip()

    async def chat_loop(self):
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\nClaude response:", response)
            except Exception as e:
                print(f"\nError: {e}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <http://localhost:8000/mcp>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_and_chat(sys.argv[1])
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
