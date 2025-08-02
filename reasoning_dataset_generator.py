from google import genai
import os
import json
import re
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def generate_reasoning_dataset(count: int = 5) -> list:
    """
    reasoning能力向上のためのデータセットを生成
    <think>タグと構造化された思考プロセスを含む
    """
    api_key = os.environ.get("gemini_api")
    if not api_key:
        raise ValueError("API key for Gemini is not set. Please set the 'gemini_api' environment variable.")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
数学の大学レベルの問題を{count}問作成し、それぞれに対して詳細な推論プロセスを生成してください。

以下のJSON形式で厳密に回答してください：

[
  {{
    "id": 1,
    "question": "問題文",
    "category": "分野名",
    "difficulty": "難易度（easy/medium/hard）",
    "reasoning_response": "<think>\\n問題を理解する：\\n[問題の核心を把握]\\n\\n解法を検討する：\\n[複数のアプローチを比較]\\n\\n計算を実行する：\\n[段階的な計算過程]\\n\\n結果を検証する：\\n[答えの妥当性をチェック]\\n</think>\\n\\n[最終的な回答]",
    "key_reasoning_steps": [
      "問題理解",
      "解法選択", 
      "計算実行",
      "結果検証"
    ],
    "common_mistakes": [
      "よくある間違い1",
      "よくある間違い2"
    ]
  }}
]

要求事項：
- 各問題は異なる数学分野から選択
- reasoning_responseには<think>タグ内に詳細な思考プロセスを記述
- <think>タグ外には簡潔な最終回答のみ
- 思考プロセスは「理解→検討→実行→検証」の流れ
- common_mistakesには学習者が陥りやすい誤りを記載
- すべて日本語で記述
- JSON形式以外の文字は含めない
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="あなたは数学教育の専門家です。学習者の推論能力を向上させるため、思考プロセスを明示的に示すデータセットを作成してください。",
            temperature=0.6,
            seed=42,
        )
    )
    
    try:
        response_text = response.text
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
        
        dataset = json.loads(json_str)
        return dataset
        
    except json.JSONDecodeError as e:
        print(f"Warning: JSONの解析に失敗しました: {e}")
        print(f"生のレスポンス:\n{response.text}")
        return []


def generate_advanced_reasoning_patterns(count: int = 3) -> list:
    """
    より高度な推論パターンを含むデータセットを生成
    """
    api_key = os.environ.get("gemini_api")
    if not api_key:
        raise ValueError("API key for Gemini is not set.")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
複雑な数学問題を{count}問作成し、高度な推論パターンを含むデータセットを生成してください。

以下のJSON形式で厳密に回答してください：

[
  {{
    "id": 1,
    "question": "複雑な問題文",
    "category": "分野名",
    "difficulty": "hard",
    "reasoning_with_reflection": "<think>\\n## 初期分析\\n[問題の構造を分析]\\n\\n## 解法候補の検討\\n候補1: [方法1の説明]\\n候補2: [方法2の説明]\\n\\n## 選択理由\\n[なぜこの方法を選ぶか]\\n\\n## 実行過程\\n[詳細な計算]\\n\\n## 中間検証\\n[途中での妥当性チェック]\\n\\n## 結果確認\\n[最終答の検証]\\n\\n## 振り返り\\n[解法の妥当性と他の可能性]\\n</think>\\n\\n[簡潔な最終回答]",
    "alternative_approaches": [
      "代替解法1の概要",
      "代替解法2の概要"
    ],
    "metacognitive_notes": [
      "この問題で重要な洞察",
      "一般化できる原理"
    ],
    "error_analysis": {{
      "common_error": "よくある間違い",
      "why_wrong": "なぜ間違えるか",
      "how_to_avoid": "回避方法"
    }}
  }}
]

要求事項：
- 問題は証明問題、最適化問題、複雑な計算問題など多様に
- reasoning_with_reflectionには段階的思考＋振り返りを含む
- alternative_approachesには他の解法も提示
- metacognitive_notesには学習につながる洞察を記載
- error_analysisには典型的な誤りパターンを構造化
- すべて日本語で記述
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="あなたは高等数学教育の専門家です。深い思考力と推論能力を育成するためのデータセットを作成してください。",
            temperature=0.4,
            seed=42,
        )
    )
    
    try:
        response_text = response.text
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
        
        dataset = json.loads(json_str)
        return dataset
        
    except json.JSONDecodeError as e:
        print(f"Warning: JSONの解析に失敗しました: {e}")
        return []


def generate_step_by_step_reasoning(count: int = 3) -> list:
    """
    Step-by-step推論を重視したデータセット生成
    """
    api_key = os.environ.get("gemini_api")
    if not api_key:
        raise ValueError("API key for Gemini is not set.")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
複数ステップが必要な数学問題を{count}問作成し、明確なステップ分解を含むデータセットを生成してください。

以下のJSON形式で厳密に回答してください：

[
  {{
    "id": 1,
    "question": "複数ステップが必要な問題文",
    "category": "分野名", 
    "difficulty": "medium",
    "step_by_step_solution": [
      {{
        "step_number": 1,
        "step_title": "ステップのタイトル",
        "thinking": "<think>このステップで何をすべきか考える</think>",
        "action": "具体的な操作や計算",
        "result": "このステップの結果",
        "verification": "結果の確認方法"
      }},
      {{
        "step_number": 2,
        "step_title": "次のステップのタイトル",
        "thinking": "<think>前のステップを受けて何をすべきか</think>",
        "action": "具体的な操作や計算",
        "result": "このステップの結果", 
        "verification": "結果の確認方法"
      }}
    ],
    "final_answer": "最終的な答え",
    "step_dependencies": [
      "ステップ1→ステップ2の依存関係",
      "ステップ2→ステップ3の依存関係"
    ]
  }}
]

要求事項：
- 各ステップは独立して理解可能に
- thinkingには各ステップでの判断理由を記載
- verificationで各段階の正確性を確認
- step_dependenciesでステップ間の論理関係を明示
- 3-5ステップ程度の問題を作成
- すべて日本語で記述
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="あなたは論理的思考教育の専門家です。学習者が段階的推論を身につけられるデータセットを作成してください。",
            temperature=0.5,
            seed=42,
        )
    )
    
    try:
        response_text = response.text
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
        
        dataset = json.loads(json_str)
        return dataset
        
    except json.JSONDecodeError as e:
        print(f"Warning: JSONの解析に失敗しました: {e}")
        return []


def create_comprehensive_reasoning_dataset():
    """
    複数のreasoning手法を組み合わせた包括的データセット作成
    """
    print("=== Reasoning能力向上データセット生成開始 ===\n")
    
    # 1. 基本的なreasoning dataset
    print("1. 基本reasoning データセット生成中...")
    basic_reasoning = generate_reasoning_dataset(3)
    
    # 2. 高度なreasoning patterns
    print("2. 高度reasoning パターン生成中...")
    advanced_reasoning = generate_advanced_reasoning_patterns(2)
    
    # 3. Step-by-step reasoning
    print("3. Step-by-step reasoning生成中...")
    step_reasoning = generate_step_by_step_reasoning(2)
    
    # 統合データセット作成
    comprehensive_dataset = {
        "basic_reasoning": basic_reasoning,
        "advanced_reasoning": advanced_reasoning, 
        "step_by_step_reasoning": step_reasoning,
        "metadata": {
            "total_problems": len(basic_reasoning) + len(advanced_reasoning) + len(step_reasoning),
            "generation_date": "2025-07-31",
            "purpose": "LLM reasoning ability improvement"
        }
    }
    
    # 保存
    with open("comprehensive_reasoning_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(comprehensive_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 生成完了 ===")
    print(f"基本reasoning: {len(basic_reasoning)}問")
    print(f"高度reasoning: {len(advanced_reasoning)}問") 
    print(f"Step-by-step: {len(step_reasoning)}問")
    print(f"総計: {comprehensive_dataset['metadata']['total_problems']}問")
    print(f"ファイル保存: comprehensive_reasoning_dataset.json")
    
    return comprehensive_dataset


if __name__ == "__main__":
    dataset = create_comprehensive_reasoning_dataset()
    
    # サンプル表示
    if dataset["basic_reasoning"]:
        print(f"\n=== サンプル（基本reasoning） ===")
        sample = dataset["basic_reasoning"][0]
        print(f"問題: {sample.get('question', '')[:100]}...")
        print(f"推論応答の一部: {sample.get('reasoning_response', '')[:200]}...")