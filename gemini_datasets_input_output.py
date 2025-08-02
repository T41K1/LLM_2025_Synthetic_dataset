from google import genai
import os
import json
import re
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def generate_questions_only(count: int = 10) -> list:
    """
    第1段階：問題文のみを生成する
    """
    api_key = os.environ.get("gemini_api")
    if not api_key:
        raise ValueError("API key for Gemini is not set. Please set the 'gemini_api' environment variable.")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
数学の大学レベルの問題を{count}問作成してください。
問題文のみを生成し、解答は含めないでください。

以下のJSON形式で厳密に回答してください：

[
  {{
    "id": 1,
    "question": "問題文",
    "category": "分野名（微積分、線形代数、統計学、解析学、代数学など）",
    "difficulty": "難易度（easy/medium/hard）"
  }},
  {{
    "id": 2,
    "question": "問題文",
    "category": "分野名",
    "difficulty": "難易度"
  }}
]

条件：
- 各問題は日本語で記述
- 様々な数学分野から問題を作成
- 難易度はeasy/medium/hardで分類
- 問題文は具体的で明確に
- JSON形式以外の文字は一切含めない
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="あなたは数学の専門家です。大学レベルの数学問題を作成することが得意です。問題文のみを生成し、解答は含めないでください。",
            temperature=0.7,
            seed=42,
        )
    )
    
    try:
        # レスポンステキストからJSONコードブロックを抽出
        response_text = response.text
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
        
        questions = json.loads(json_str)
        return questions
        
    except json.JSONDecodeError as e:
        print(f"Warning: JSONの解析に失敗しました: {e}")
        print(f"生のレスポンス:\n{response.text}")
        return []


def generate_answer_and_reasoning(question_data: dict) -> dict:
    """
    第2段階：問題文を受け取って解答と思考過程を生成する
    """
    api_key = os.environ.get("gemini_api")
    if not api_key:
        raise ValueError("API key for Gemini is not set. Please set the 'gemini_api' environment variable.")
    
    client = genai.Client(api_key=api_key)
    
    question = question_data.get("question", "")
    category = question_data.get("category", "")
    difficulty = question_data.get("difficulty", "")
    
    prompt = f"""
以下の数学問題について、詳細な解答と思考過程を作成してください。

問題: {question}
分野: {category}
難易度: {difficulty}

以下のJSON形式で厳密に回答してください：

{{
  "question": "上記の問題文をここに記載",
  "category": "上記の分野をここに記載",
  "difficulty": "上記の難易度をここに記載",
  "thinking_process": "この問題を解くための思考過程や解法のアプローチを段階的に説明",
  "solution": "詳細な解答手順",
  "final_answer": "最終的な答え",
  "key_concepts": ["使用した主要な数学概念1", "概念2", "概念3"]
}}

条件：
- questionには上記で与えられた問題文をそのまま記載
- categoryには上記で与えられた分野をそのまま記載  
- difficultyには上記で与えられた難易度をそのまま記載
- thinking_processには問題を見たときの最初の分析から解法選択まで一連の思考過程を詳細に記述
- final_answerには簡潔で明確な最終答
- key_conceptsには問題解決に必要な数学概念を配列で
- すべて日本語で記述
- JSON形式以外の文字は一切含めない
- 数式のLaTeX記号やクォートは適切にエスケープすること
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="あなたは数学の専門家です。与えられた問題に対して、思考過程から最終解答まで詳細に説明することが得意です。",
            temperature=0.3,  # より一貫した解答のため低めに設定
            seed=42,
        )
    )
    
    try:
        # レスポンステキストからJSONコードブロックを抽出
        response_text = response.text
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response_text.strip()
        
        # よくある問題を修正
        # 1. Trailing commaを削除
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 2. 複数のJSON修正を試行
        for attempt in range(3):
            try:
                answer_data = json.loads(json_str)
                return answer_data
            except json.JSONDecodeError as e:
                if attempt == 0:
                    # 最初の試行: 制御文字を修正
                    json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                elif attempt == 1:
                    # 2回目の試行: バックスラッシュを修正
                    json_str = json_str.replace('\\', '\\\\')
                else:
                    # 最終試行: 失敗
                    raise e
        
    except json.JSONDecodeError as e:
        # JSON解析に失敗した場合は詳細なデバッグ情報を出力
        print(f"Warning: JSONの解析に失敗しました: {e}")
        print(f"JSON文字列の最初の200文字:")
        print(repr(json_str[:200]) if 'json_str' in locals() else "N/A")
        print(f"生のレスポンスの最初の500文字:")
        print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        
        # 元のデータ構造を保持してエラーを返す
        return {
            "question": question,
            "category": category,
            "difficulty": difficulty,
            "thinking_process": "解析エラー - JSON解析に失敗",
            "solution": f"解析エラー - {str(e)}",
            "final_answer": "解析エラー",
            "key_concepts": [],
            "debug_info": {
                "error": str(e),
                "raw_response_preview": response.text[:200]
            }
        }


def create_complete_dataset(question_count: int = 5) -> list:
    """
    2段階プロセスを統合してComplete datasetを作成
    """
    print(f"第1段階: {question_count}問の問題文を生成中...")
    questions = generate_questions_only(question_count)
    
    if not questions:
        print("問題文の生成に失敗しました")
        return []
    
    print(f"第1段階完了: {len(questions)}問の問題文を生成しました")
    print("\n第2段階: 各問題の解答と思考過程を生成中...")
    
    complete_dataset = []
    
    for i, question_data in enumerate(questions, 1):
        print(f"問題 {i}/{len(questions)} を処理中...")
        
        # IDを追加
        question_data["id"] = i
        
        # 解答と思考過程を生成
        answer_data = generate_answer_and_reasoning(question_data)
        
        # IDを追加して完全なデータセットに追加
        answer_data["id"] = i
        complete_dataset.append(answer_data)
    
    print(f"\n第2段階完了: {len(complete_dataset)}問の完全なデータセットを作成しました")
    return complete_dataset


def save_dataset(dataset: list, filename: str = "complete_math_dataset.json"):
    """
    データセットをJSONファイルに保存
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\nデータセットを {filename} に保存しました")


if __name__ == "__main__":
    # 5問の完全なデータセットを作成
    dataset = create_complete_dataset(question_count=5)
    
    if dataset:
        # 統計情報を表示
        print(f"\n=== データセット統計 ===")
        print(f"総問題数: {len(dataset)}")
        
        # カテゴリ別集計
        categories = {}
        difficulties = {}
        
        for problem in dataset:
            category = problem.get("category", "不明")
            difficulty = problem.get("difficulty", "不明")
            categories[category] = categories.get(category, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        print(f"\nカテゴリ別:")
        for category, count in categories.items():
            print(f"  {category}: {count}問")
        
        print(f"\n難易度別:")
        for difficulty, count in difficulties.items():
            print(f"  {difficulty}: {count}問")
        
        # データセットを保存
        save_dataset(dataset, "gemini_complete_dataset_001.json")
        
        # サンプルを表示
        print(f"\n=== サンプル問題 ===")
        if dataset:
            sample = dataset[0]
            print(f"問題: {sample.get('question', '')}")
            print(f"分野: {sample.get('category', '')}")
            print(f"思考過程: {sample.get('thinking_process', '')[:100]}...")
            print(f"最終答: {sample.get('final_answer', '')}")
    else:
        print("データセットの作成に失敗しました")