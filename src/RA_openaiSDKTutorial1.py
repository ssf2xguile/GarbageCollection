import asyncio
import os
from dotenv import load_dotenv
from agents import Runner, Agent

load_dotenv()
os.environ['OPENAI_API_KEY']
agent = Agent(
    name='simple agent', # 必須
    instructions='入力された文章をそのまま返してください',
    model='gpt-4o-mini' # 現時点では未指定の場合`gpt-4o`
)

async def main():
    result = await Runner.run(agent, 'おはよう!!')
    print(result)

if __name__ == '__main__':
    asyncio.run(main())