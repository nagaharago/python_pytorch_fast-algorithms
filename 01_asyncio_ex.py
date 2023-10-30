import os
import asyncio
import time
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools

# TODO openaiのapiキー
os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(temperature=0.0)
tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
questions = [f"{i+1}の0.25乗は？" for i in range(10)]


# 逐次処理による実装
# def main():
#     s = time.perf_counter()
#     for q in questions:
#         agent.run(q)
#     elapsed = time.perf_counter() - s
#     print(f"逐次処理による処理時間： {elapsed:0.2f} 秒")  # 34.06 秒


# 非同期処理による実装
async def main():
    s = time.perf_counter()
    tasks = [agent.arun(q) for q in questions]
    await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - s
    print(f"並列処理に処理時間： {elapsed:0.2f} 秒")


asyncio.run(main())
