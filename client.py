import asyncio
import os
import sys
from contextlib import AsyncExitStack
from dotenv import load_dotenv
import json
from fastmcp import Client
from groq import Groq

load_dotenv()  # Load environment variables from .env
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

config = {
    "mcpServers": {
        "arxiv_server": {
            "url": "http://localhost:8000/mcp",
            "transport": "http",
        },
        "sentiment_analysis_server": {
            "url": "http://localhost:8001/mcp",
            "transport": "http",
        },
        "pull_request_agent_server": {
            "url": "http://localhost:8002/mcp",
            "transport": "http",
        },
    }
}


class MCPClient:
    def __init__(self):
        self.client: Client | None = None
        self.exit_stack = AsyncExitStack()
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    async def connect_and_chat(self):
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        self.client = Client(config)
        async with self.client:
            tools = await self.client.list_tools()
            print("\nConnected to server with tools:", [tool.name for tool in tools])
            await self.chat_loop()

    async def process_query(self, query: str) -> str:
        if not self.client:
            raise RuntimeError("Client not connected")

        system_prompt = """
            You are an intelligent assistant designed to help users by leveraging a variety of specialized tools and functions available to you. Your job is to understand the user's queries and determine when to call the appropriate tool to fetch accurate and relevant information. Always aim to provide clear, concise, and helpful responses by integrating tool outputs when necessary. If you don't have enough information, prompt the user for clarification. Your goal is to assist the user efficiently by combining your language understanding with the power of these external tools.

            You have access to the following tools:
            - search_papers: Search for academic papers on a specific topic.
            - extract_info: Extract detailed information about a specific paper.
            - get_topic_papers: Retrieve detailed information about papers on a specific topic.
            Use these tools to enhance your responses and provide the user with the best possible information. 
            You have access to the following prompt:
            - generate_search_prompt: Generate a prompt for the LLM to find and discuss academic papers on a specific topic.
            Use this prompt to create a focused search for papers related to the user's query.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

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

        choice = response.choices[0]
        message = choice.message
        logging.info(
            f"Groq completion: finish_reason={choice.finish_reason} | tool_calls={message.tool_calls}"
        )

        # Case 1: Normal chat/completion response
        if choice.finish_reason == "stop" and message.content:
            final_text.append(message.content)

        # Case 2: Tool call requested
        elif choice.finish_reason == "tool_calls" and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                # Arguments may be a JSON string; parse it if needed
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                # Call the MCP tool
                result = await self.client.call_tool(tool_name, tool_args)
                # tool_results.append({"call": tool_name, "result": result})

                logging.info(
                    f"[Calling tool {tool_name} with args {tool_args}. Output: {result.content}]"
                )

                # Inject a world-class assistant prompt
                assistant_prompt = f"""
                If you used the 'search_papers' tool, you should have a list of paper IDs in {result.content[0].text}.
                    The assistant has just retrieved information from a specialized tool in {result.content[0].text} which contains the paper ID retreived from the arxiv server. Use these paper IDs to fetch detailed information about the papers.
                    For each paper found, extract and organize the following information:
                    - Paper title
                    - Authors
                    - Publication date
                    - Brief summary of the key findings
                    - Main contributions or innovations
                    - Methodologies used
                    - Relevance to the topic '{tool_args.get("topic")}'
                    
                    3. Provide a comprehensive summary that includes:
                    - Overview of the current state of research in '{tool_args.get("topic")}'
                    - Common themes and trends across the papers
                    - Key research gaps or areas for future investigation
                    - Most impactful or influential papers in this area
                    
                    4. Organize your findings in a clear, structured format with headings and bullet points for easy readability.
                    
                    Please present both detailed information about each paper and a high-level synthesis of the research landscape in {tool_args.get("topic")}.
                    Avoid unnecessary repetition and keep the response focused and engaging.
                    """
                messages.append({"role": "assistant", "content": assistant_prompt})
                # messages.append({"role": "user", "content": result.content})

                # Get next LLM response with updated chat history
                response = await asyncio.to_thread(
                    self.groq.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=groq_tools,
                    max_tokens=1000,
                )
                next_choice = response.choices[0]
                next_content = next_choice.message.content
                if next_content:
                    final_text.append(next_content)

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
    client = MCPClient()
    try:
        await client.connect_and_chat()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
