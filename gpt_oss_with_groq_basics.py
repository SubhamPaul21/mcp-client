import asyncio
import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# Agno imports
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.mcp import MCPTools

# MCP imports
from mcp import StdioServerParameters

# Load environment variables
load_dotenv()


class AgnoCryptocurrencyResearchAgent:
    """
    Advanced cryptocurrency research agent using Agno framework with MCP Playwright integration
    """

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not found!")

        # Setup MCP Playwright server parameters
        self.mcp_server_params = StdioServerParameters(
            command="npx", args=["@playwright/mcp@latest"]
        )

        # Initialize agent components
        self.agent = None
        self.mcp_tools = None

    async def initialize_agent(self) -> Agent:
        """Initialize the Agno agent with Groq model and MCP tools"""

        # Initialize MCP tools with Playwright server
        self.mcp_tools = MCPTools(
            server_params=self.mcp_server_params,
            # Limit tools to prevent token overflow
            include_tools=[
                "browser_navigate",
                "browser_take_screenshot",
                "browser_click",
                "browser_evaluate",
                "browser_wait_for",
            ],
        )
        print("🔧 MCP Tools initialized with Playwright server: ", self.mcp_tools)
        # Create the agent with Groq GPT-OSS-120B model
        self.agent = Agent(
            name="Crypto Research Agent",
            model=Groq(id="openai/gpt-oss-120b"),
            tools=[self.mcp_tools],
            # Agent configuration for cryptocurrency research
            role="Expert Cryptocurrency Research Analyst",
            description="""
            You are an advanced cryptocurrency research agent with real-time web browsing capabilities.
            Your expertise includes technical analysis, market trends, price movements, and investment insights.
            """,
            instructions=[
                "Use playwright browser tools to gather real-time cryptocurrency data",
                "Navigate to reputable crypto websites like TradingView, CoinGecko, CoinMarketCap and relevant crypto websites",
                "Take screenshots to analyze charts and market data visually",
                "Extract specific price data, market trends, and technical indicators and show it in human readable format",
                "Provide comprehensive analysis with supporting evidence",
                "Conclude with actionable investment insights based on gathered data",
                "Don't show any code or technical details in the final response",
                "Use markdown formatting for better readability",
                "Keep responses concise but informative, focusing on key insights",
                "Avoid unnecessary jargon, explain technical terms clearly",
                "Maintain a professional and analytical tone throughout",
                "Ensure all responses are factually accurate and well-researched",
            ],
            # Agent behavior settings
            markdown=True,
            show_tool_calls=True,
            add_history_to_messages=True,
            # Performance optimizations
            # exponential_backoff=True, # Commented out if not supported in current version
            debug_mode=False,
        )

        return self.agent

    async def research_cryptocurrency(self, query: str) -> str:
        """
        Perform comprehensive cryptocurrency research using Agno's built-in multi-step tool calling
        """
        if not self.agent:
            await self.initialize_agent()

        print(f"🚀 Starting cryptocurrency research with Agno + Groq GPT-OSS-120B")
        print(f"📝 Query: {query}")
        print("=" * 80)

        try:
            # Use Agno's run method for async operation (not agenerate_response)
            response = await self.agent.arun(message=query, stream=False)

            return (
                response.content
                if response and hasattr(response, "content")
                else "No response content"
            )

        except Exception as e:
            error_msg = f"Research error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def research_cryptocurrency_sync(self, query: str) -> str:
        """
        Synchronous version using print_response method
        """
        if not self.agent:
            # Initialize synchronously for demo
            import asyncio

            asyncio.run(self.initialize_agent())

        print(f"🚀 Starting cryptocurrency research with Agno + Groq GPT-OSS-120B")
        print(f"📝 Query: {query}")
        print("=" * 80)

        try:
            # Use Agno's print_response method for synchronous operation
            response = self.agent.print_response(
                message=query, stream=True  # Enable streaming for real-time output
            )

            return "Research completed - check console output above"

        except Exception as e:
            error_msg = f"Research error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    async def analyze_multiple_cryptocurrencies(
        self, cryptocurrencies: List[str]
    ) -> Dict[str, str]:
        """
        Analyze multiple cryptocurrencies sequentially
        """
        results = {}

        for crypto in cryptocurrencies:
            query = f"Research {crypto} cryptocurrency: current price, market trends, technical analysis, and investment outlook"
            print(f"\n🔍 Analyzing {crypto}...")

            result = await self.research_cryptocurrency(query)
            results[crypto] = result

            # Small delay between requests to avoid rate limiting
            await asyncio.sleep(2)

        return results


async def main():
    """Main function to demonstrate the Agno cryptocurrency research agent"""

    # Initialize the agent
    agent = AgnoCryptocurrencyResearchAgent()

    # Sample research queries
    research_queries = [
        "Navigate to CoinGecko, analyze Bitcoin's current price and recent trends. Provide investment analysis based on technical indicators.",
        "Research Ethereum's price performance on Trading View. Compare with market conditions and provide trading recommendations.",
    ]

    print("🦾 Agno Cryptocurrency Research Agent")
    print("Powered by Groq GPT-OSS-120B + MCP Playwright")
    print("=" * 60)

    # Execute research queries
    for i, query in enumerate(research_queries, 1):
        print(f"\n📊 RESEARCH TASK {i}")
        print("-" * 40)

        try:
            # Try async method first
            result = await agent.research_cryptocurrency(query)

            print("\n✅ ANALYSIS COMPLETE")
            print("=" * 40)
            print(result)
            print("\n" + "=" * 80 + "\n")

        except Exception as e:
            print(f"❌ Async method failed: {e}")
            print("🔄 Trying synchronous method...")

            try:
                # Fallback to sync method
                result = agent.research_cryptocurrency_sync(query)
                print(f"\n✅ SYNC ANALYSIS COMPLETE: {result}")

            except Exception as e2:
                print(f"❌ Task {i} failed completely: {e2}")


def main_sync():
    """Synchronous main function using print_response"""

    # Initialize the agent
    agent = AgnoCryptocurrencyResearchAgent()

    # Single focused query for testing
    query = "Navigate to TradingView, analyze Bitcoin's current price and recent trends. Provide investment analysis based on technical indicators."

    print("🦾 Agno Cryptocurrency Research Agent (Sync)")
    print("Powered by Groq GPT-OSS-120B + MCP Playwright")
    print("=" * 60)

    # Use synchronous method
    result = agent.research_cryptocurrency_sync(query)
    print(f"\n✅ Final Result: {result}")


if __name__ == "__main__":
    # Try async first, fallback to sync if needed
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Async execution failed: {e}")
        print("🔄 Falling back to synchronous execution...")
        main_sync()
