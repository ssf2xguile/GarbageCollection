import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, handoff, ModelSettings

load_dotenv()
os.environ['OPENAI_API_KEY']
# ========== ツール定義（LLM駆動、実装なし） ==========
@function_tool
def analyze_code(code: str) -> str:
    """Djangoコードの目的や挙動を自然言語で説明してください。"""
    pass

@function_tool
def detect_bugs(code: str, summary: str) -> str:
    """コードとその目的を元に、考えられるバグや問題点を列挙してください。"""
    pass

@function_tool
def fix_code(code: str, issues: str) -> str:
    """コードとそのバグ内容に基づいて、修正済の安全なDjangoコードを生成してください。"""
    pass


# ========== エージェント定義 ==========
model_cfg = ModelSettings(parallel_tool_calls=True)

# 3.最終エージェント
code_fixer = Agent(
    name="CodeFixer",
    instructions="あなたはバグ修正の専門家です。問題点を基にDjangoコードを修正してください。",
    tools=[fix_code],
    model_settings=model_cfg,
)

# 2.バグ検出エージェント
bug_detector = Agent(
    name="BugDetector",
    instructions="あなたはコードレビュアーです。コードとその説明をもとにバグを検出してください。",
    tools=[detect_bugs],
    handoffs=[handoff(code_fixer)],
    model_settings=model_cfg,
)

# 1.コードリーダーエージェント
code_reader = Agent(
    name="CodeReader",
    instructions="コードの目的を説明してください。",
    tools=[analyze_code],
    handoffs=[handoff(bug_detector)],
    model_settings=model_cfg,
)


# ========== Runner 実行 ==========
async def main():
    buggy_code = '''
        from django.http import JsonResponse
        import requests

        def proxy_view(request):
            url = request.GET.get("url")
            resp = requests.get(url)
            return JsonResponse({"status": resp.status_code, "data": resp.text})
    '''

    print("=== 🧠 CodeReader 出力 ===")
    result_1 = await Runner.run(code_reader, input=buggy_code)
    print(result_1.final_output)

    bug_input = f"""
        次のDjangoコードとその説明をもとに、問題点を洗い出してください。

        [コード]
        {buggy_code}

        [説明]
        {result_1.final_output}
    """
    print("\n=== 🐞 BugDetector 出力 ===")
    result_2 = await Runner.run(bug_detector, input=bug_input)
    print(result_2.final_output)

    fix_input = f"""
        次のDjangoコードとそのバグ内容をもとに、安全なコードに修正してください。

        [コード]
        {buggy_code}

        [バグ内容]
        {result_2.final_output}
    """
    print("\n=== 🛠 CodeFixer 出力（修正済コード） ===")
    result_3 = await Runner.run(code_fixer, input=fix_input)
    print(result_3.final_output)

if __name__ == '__main__':
    asyncio.run(main())