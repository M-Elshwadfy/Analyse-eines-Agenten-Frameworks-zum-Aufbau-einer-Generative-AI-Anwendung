from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import asyncio

provider = OpenAIProvider(base_url="http://localhost:11434/v1")
model = OpenAIModel("llama3.1:8b", provider=provider)

agent = Agent(model=model)  


async def main():
    async with agent.run_stream('in 50 words, Where does "hello world" come from?') as result:  
        async for message in result.stream_text():  
            print(message)
            #> The first known
            #> The first known use of "hello,
            #> The first known use of "hello, world" was in
            #> The first known use of "hello, world" was in a 1974 textbook
            #> The first known use of "hello, world" was in a 1974 textbook about the C
            #> The first known use of "hello, world" was in a 1974 textbook about the C programming language.
asyncio.run(main())