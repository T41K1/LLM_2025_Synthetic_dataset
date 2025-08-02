import json
import re

def debug_json_issues():
    """
    JSON解析エラーの典型的な原因をデバッグ
    """
    
    # 数学問題に含まれる問題文字の例
    problematic_strings = [
        'f(x) = x^2 + 2x + 1',  # 数式
        'A = \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}',  # LaTeX
        'x ∈ [0, 1]',  # 数学記号
        '"Hello World"',  # 内部の引用符
        'x\ny = 2',  # 改行文字
        'f(x,y,z) = x²+y²+z²',  # 上付き文字
        '$\\int_0^1 x dx = \\frac{1}{2}$',  # LaTeX数式
    ]
    
    print("=== JSON構文エラーの原因分析 ===\n")
    
    for i, text in enumerate(problematic_strings, 1):
        print(f"{i}. テスト文字列: {text}")
        
        # 直接JSONに埋め込んだ場合（問題のあるパターン）
        try:
            bad_json = f'{{"text": "{text}"}}'
            json.loads(bad_json)
            print("   ✅ 問題なし")
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON エラー: {e}")
            
        # 適切にエスケープした場合（修正版）
        try:
            escaped_text = json.dumps(text)  # 自動エスケープ
            good_json = f'{{"text": {escaped_text}}}'
            json.loads(good_json)
            print("   ✅ エスケープ後: 正常")
        except json.JSONDecodeError as e:
            print(f"   ❌ エスケープ後もエラー: {e}")
            
        print()


def analyze_gemini_response_patterns():
    """
    Geminiレスポンスの典型的なパターンを分析
    """
    
    # Geminiが返す可能性のあるレスポンス形式
    response_patterns = [
        # パターン1: 正常なJSONコードブロック
        '''```json
{
  "question": "2x + 3 = 7を解け",
  "answer": "x = 2"
}
```''',
        
        # パターン2: 前後にテキストがある
        '''数学問題を作成しました。

```json
{
  "question": "微分を求めよ",
  "answer": "導関数は..."
}
```

以上が回答です。''',
        
        # パターン3: JSONコードブロックなし
        '''{
  "question": "積分を計算せよ",
  "answer": "結果は..."
}''',
        
        # パターン4: 不正なJSON（trailing comma等）
        '''```json
{
  "question": "問題文",
  "answer": "解答",
}
```''',
        
        # パターン5: エスケープされていない特殊文字
        '''```json
{
  "question": "f(x) = x^2 + "hello" world",
  "answer": "x = \n2"
}
```''',
    ]
    
    print("=== Geminiレスポンスパターン分析 ===\n")
    
    for i, response in enumerate(response_patterns, 1):
        print(f"パターン {i}:")
        print("=" * 40)
        print(response[:100] + "..." if len(response) > 100 else response)
        print()
        
        # JSONコードブロック抽出を試行
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            print("抽出されたJSON:")
            print(json_str)
            
            try:
                parsed = json.loads(json_str)
                print("✅ JSON解析成功")
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析エラー: {e}")
        else:
            # コードブロックなしの場合
            try:
                parsed = json.loads(response.strip())
                print("✅ 直接JSON解析成功")
            except json.JSONDecodeError as e:
                print(f"❌ 直接JSON解析エラー: {e}")
        
        print("\n" + "="*50 + "\n")


def create_robust_json_parser():
    """
    より堅牢なJSONパーサーを作成
    """
    
    def parse_gemini_json(response_text: str) -> dict:
        """
        Geminiレスポンスから安全にJSONを抽出・解析
        """
        
        # Step 1: JSONコードブロックを探す
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # コードブロックがない場合は全体を試す
            json_str = response_text.strip()
        
        # Step 2: よくある問題を修正
        # Trailing commaを削除
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Step 3: JSON解析を試行
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析エラー: {e}")
            print(f"問題のあるJSON文字列の一部:")
            print(json_str[:200] + "..." if len(json_str) > 200 else json_str)
            
            # Step 4: より詳細なエラー分析
            try:
                # 文字レベルでの問題を特定
                lines = json_str.split('\n')
                for i, line in enumerate(lines, 1):
                    try:
                        # 各行を個別にチェック
                        if line.strip() and not line.strip().startswith('//'):
                            json.loads('{' + line + '}')
                    except json.JSONDecodeError:
                        print(f"問題のある行 {i}: {line}")
            except:
                pass
            
            return {"error": "JSON parsing failed", "raw_response": response_text}
    
    return parse_gemini_json


if __name__ == "__main__":
    print("JSON解析問題のデバッグツール実行\n")
    
    # 1. 基本的な問題の分析
    debug_json_issues()
    
    # 2. レスポンスパターンの分析
    analyze_gemini_response_patterns()
    
    # 3. 堅牢なパーサーのテスト
    parser = create_robust_json_parser()
    
    test_response = '''```json
{
  "question": "f(x) = x^2を微分せよ",
  "answer": "f'(x) = 2x",
}
```'''
    
    print("=== 堅牢パーサーのテスト ===")
    result = parser(test_response)
    print("結果:", result)