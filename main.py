import asyncio
from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"


async def main() -> None:
    # Connect to a local Python server using stdio transport
    async with Client(SERVER_URL) as client:
        # Sanity check: ping the server
        await client.ping()

        print(f"Connected: {client.is_connected()}")

        # List all tools exposed by the server
        tools = await client.list_tools()
        print("Available tools:", tools)


if __name__ == "__main__":
    asyncio.run(main())
