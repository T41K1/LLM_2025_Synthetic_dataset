from google import genai
import os
import json
from google.genai import types
import numpy as np
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("gemini_api")
# if api_key:
#     print("API key is set.")
# else:
#     print("API key is not set.")

def get_dataset():
    """
    geminiで簡易的にデータセットを作成する
    """
    api_key = os.environ.get("gemini_api")
    if not api_key:
        raise ValueError("API key for Gemini is not set. Please set the 'gemini_api' environment variable.")
    
    client = genai.Client(api_key=api_key)
    
    system = """
あなたは数学の専門家です。大学レベルの数学問題を作成することが得意です。
以下の条件に従って、数学の問題を1問生成してください。
以下のJSON形式で厳密に回答してください：

{
  "problems": [
    {   
      "id": 1,
      "question": "問題文",
      "answer": "解答",
      "category": "分野名（例：微積分、線形代数、統計学など）",
      "difficulty": "難易度（easy/medium/hard）"
    }
  ]
}

各問題は日本語で、解答も詳しく含めてください。JSON形式以外の文字は一切含めないでください。
    """
    
    response = client.models.generate_content(
        model = "gemini-2.5-flash", #model名
        contents = "system_instructionの指示を元に問題を作成してください。",#input内容
        config =types.GenerateContentConfig(
            system_instruction= system,  # システムインストラクション",
            temperature=0.5,  # 温度パラメータ
            seed=42,  # シード値
            
        )

    )
    
    try:
        # レスポンステキストからJSONコードブロックを抽出
        response_text = response.text
        
        # ```json で始まり ``` で終わるパターンを検索
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # コードブロックがない場合はそのまま使用
            json_str = response_text
        
        # JSON形式のレスポンスをパース
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        # JSON解析に失敗した場合は生のテキストを返す
        print(f"Warning: JSONの解析に失敗しました: {e}")
        print(f"生のレスポンス:\n{response.text}")
        return {"raw_response": response.text}


if __name__ == "__main__":
    dataset = get_dataset()
    print("生成されたデータセット:")
    print(json.dumps(dataset, ensure_ascii=False, indent=2))
    
    # データセットをJSONファイルに保存
    with open("gemini_dataset_0731_001.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"\nデータセットを gemini_dataset_0731_001.json に保存しました。")


