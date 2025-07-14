import asyncio
import os
from dotenv import load_dotenv
import subprocess
from agents import Runner, Agent, function_tool

load_dotenv()
os.environ['OPENAI_API_KEY']

@function_tool
def read_file(path: str) -> str:
    """指定されたパスのファイルを読み込みます。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@function_tool
def write_file(path: str, content: str) -> str:
    """指定されたパスにファイルを書き込みます。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"{path} に書き込みました。"

@function_tool
def run_fixed_code(path: str) -> str:
    """指定されたPythonファイルを実行して結果を返します。"""
    try:
        result = subprocess.run(["python", path], capture_output=True, text=True, check=True)
        return f"実行結果:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"実行中にエラー:\n{e.stderr}"

# エージェント定義
agent = Agent(
    name="BugFixer",
    instructions=(
        "与えられたPythonコードにバグが含まれている場合、それを修正し、"
        "`./src/RA_openaiSDKbuggy_program_fixed.py` として保存し、その後に実行してください。"
        "pythonファイルはpoetry環境として実行してください。"
    ),
    tools=[read_file, write_file, run_fixed_code],
)

async def run():
    message = (
        "ファイル `./src/RA_openaiSDKbuggy_program.py` を読み込み、バグを修正して "
        "`./src/RA_openaiSDKbuggy_program_fixed.py` に保存し、それを実行してください。"
    )
    response = await Runner.run(agent, message)
    print(response)

if __name__ == "__main__":
    asyncio.run(run())
