

import random
import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class MathProblem:
    question: str
    answer: str
    difficulty: str
    category: str


class MathDatasetGenerator:
    def __init__(self):
        self.problems = []
    
    def generate_arithmetic_problems(self, count: int = 100) -> List[MathProblem]:
        """基本的な算数問題を生成"""
        problems = []
        operations = ['+', '-', '*', '/']
        
        for _ in range(count):
            op = random.choice(operations)
            
            if op == '+':
                a, b = random.randint(1, 100), random.randint(1, 100)
                question = f"{a} + {b} = ?"
                answer = str(a + b)
            elif op == '-':
                a, b = random.randint(10, 100), random.randint(1, 50)
                if a < b:
                    a, b = b, a
                question = f"{a} - {b} = ?"
                answer = str(a - b)
            elif op == '*':
                a, b = random.randint(1, 20), random.randint(1, 20)
                question = f"{a} × {b} = ?"
                answer = str(a * b)
            else:  # division
                b = random.randint(2, 20)
                result = random.randint(1, 20)
                a = b * result
                question = f"{a} ÷ {b} = ?"
                answer = str(result)
            
            problems.append(MathProblem(
                question=question,
                answer=answer,
                difficulty="easy",
                category="arithmetic"
            ))
        
        return problems
    
    def generate_algebra_problems(self, count: int = 50) -> List[MathProblem]:
        """代数問題を生成"""
        problems = []
        
        for _ in range(count):
            # 一次方程式 ax + b = c
            a = random.randint(2, 10)
            x = random.randint(1, 20)
            b = random.randint(1, 30)
            c = a * x + b
            
            question = f"{a}x + {b} = {c}のとき、xの値は？"
            answer = str(x)
            
            problems.append(MathProblem(
                question=question,
                answer=answer,
                difficulty="medium",
                category="algebra"
            ))
        
        # 二次方程式の例も追加
        for _ in range(count // 2):
            # x^2 + bx + c = 0の形で解が整数になるもの
            x1, x2 = random.randint(1, 5), random.randint(1, 5)
            b = -(x1 + x2)
            c = x1 * x2
            
            question = f"x² + {b}x + {c} = 0の解は？"
            answer = f"x = {x1}, {x2}" if x1 != x2 else f"x = {x1}"
            
            problems.append(MathProblem(
                question=question,
                answer=answer,
                difficulty="hard",
                category="algebra"
            ))
        
        return problems
    
    def generate_geometry_problems(self, count: int = 30) -> List[MathProblem]:
        """幾何問題を生成"""
        problems = []
        
        for _ in range(count):
            problem_type = random.choice(['rectangle_area', 'circle_area', 'triangle_area'])
            
            if problem_type == 'rectangle_area':
                width = random.randint(3, 15)
                height = random.randint(3, 15)
                question = f"幅{width}cm、高さ{height}cmの長方形の面積は？"
                answer = f"{width * height}cm²"
            
            elif problem_type == 'circle_area':
                radius = random.randint(2, 10)
                area = 3.14 * radius * radius
                question = f"半径{radius}cmの円の面積は？（π=3.14として計算）"
                answer = f"{area}cm²"
            
            else:  # triangle_area
                base = random.randint(4, 20)
                height = random.randint(3, 15)
                area = base * height / 2
                question = f"底辺{base}cm、高さ{height}cmの三角形の面積は？"
                answer = f"{area}cm²"
            
            problems.append(MathProblem(
                question=question,
                answer=answer,
                difficulty="medium",
                category="geometry"
            ))
        
        return problems
    
    def generate_word_problems(self, count: int = 20) -> List[MathProblem]:
        """文章問題を生成"""
        problems = []
        
        templates = [
            {
                "template": "太郎くんは{item}を{initial}個持っています。友達から{given}個もらいました。太郎くんが持っている{item}は全部で何個ですか？",
                "items": ["りんご", "みかん", "鉛筆", "消しゴム", "シール"],
                "category": "word_problem_addition"
            },
            {
                "template": "花子さんは{money}円持っていました。{item}を{price}円で買いました。おつりはいくらですか？",
                "category": "word_problem_subtraction"
            }
        ]
        
        for _ in range(count):
            if random.random() < 0.5:
                # 足し算の文章問題
                item = random.choice(["りんご", "みかん", "鉛筆", "消しゴム", "シール"])
                initial = random.randint(5, 20)
                given = random.randint(3, 15)
                question = f"太郎くんは{item}を{initial}個持っています。友達から{given}個もらいました。太郎くんが持っている{item}は全部で何個ですか？"
                answer = f"{initial + given}個"
            else:
                # 引き算の文章問題
                money = random.randint(500, 2000)
                price = random.randint(100, money - 100)
                item = random.choice(["ノート", "ペン", "お菓子", "ジュース"])
                question = f"花子さんは{money}円持っていました。{item}を{price}円で買いました。おつりはいくらですか？"
                answer = f"{money - price}円"
            
            problems.append(MathProblem(
                question=question,
                answer=answer,
                difficulty="medium",
                category="word_problem"
            ))
        
        return problems
    
    def generate_llm_problems(self, count: int = 10, use_mock: bool = True) -> List[MathProblem]:
        """LLMを使って問題を生成（モック版とAPI版）"""
        problems = []
        
        if use_mock:
            # モックデータ（実際のLLM APIを使わない場合）
            mock_problems = [
                {
                    "question": "ある数の3倍に5を足すと23になります。この数は何ですか？",
                    "answer": "6",
                    "difficulty": "medium",
                    "category": "llm_generated"
                },
                {
                    "question": "50人のクラスで、男子は女子より8人多いです。男子と女子はそれぞれ何人ですか？",
                    "answer": "男子29人、女子21人",
                    "difficulty": "hard",
                    "category": "llm_generated"
                },
                {
                    "question": "時速60kmで走る車が3時間走ったとき、何km進みますか？",
                    "answer": "180km",
                    "difficulty": "easy",
                    "category": "llm_generated"
                }
            ]
            
            for i in range(min(count, len(mock_problems))):
                problem_data = mock_problems[i]
                problems.append(MathProblem(**problem_data))
        
        else:
            # 実際のLLM API使用例（OpenAI）
            try:
                import openai
                
                for _ in range(count):
                    prompt = """
                    小学生から中学生レベルの数学問題を1つ作成してください。
                    以下の形式で回答してください：
                    問題: [問題文]
                    答え: [答え]
                    """
                    
                    # Note: 実際に使用する場合はAPIキーの設定が必要
                    # response = openai.ChatCompletion.create(
                    #     model="gpt-3.5-turbo",
                    #     messages=[{"role": "user", "content": prompt}],
                    #     max_tokens=200
                    # )
                    # 
                    # content = response.choices[0].message.content
                    # # パースしてMathProblemオブジェクトを作成
                    
                    pass  # APIキーが設定されていない場合はスキップ
                    
            except ImportError:
                print("OpenAI ライブラリがインストールされていません。pip install openai")
        
        return problems
    
    def generate_dataset(self, 
                        arithmetic_count: int = 100,
                        algebra_count: int = 50, 
                        geometry_count: int = 30,
                        word_count: int = 20,
                        llm_count: int = 10) -> List[MathProblem]:
        """全種類の問題を生成してデータセットを作成"""
        
        print("算数問題を生成中...")
        arithmetic_problems = self.generate_arithmetic_problems(arithmetic_count)
        
        print("代数問題を生成中...")
        algebra_problems = self.generate_algebra_problems(algebra_count)
        
        print("幾何問題を生成中...")
        geometry_problems = self.generate_geometry_problems(geometry_count)
        
        print("文章問題を生成中...")
        word_problems = self.generate_word_problems(word_count)
        
        print("LLM生成問題を生成中...")
        llm_problems = self.generate_llm_problems(llm_count)
        
        all_problems = arithmetic_problems + algebra_problems + geometry_problems + word_problems + llm_problems
        random.shuffle(all_problems)
        
        self.problems = all_problems
        return all_problems
    
    def save_dataset(self, filename: str = "math_dataset.json"):
        """データセットをJSONファイルに保存"""
        dataset = []
        for problem in self.problems:
            dataset.append({
                "question": problem.question,
                "answer": problem.answer,
                "difficulty": problem.difficulty,
                "category": problem.category
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"データセットを {filename} に保存しました（{len(dataset)}問）")
    
    def get_statistics(self):
        """データセットの統計情報を表示"""
        if not self.problems:
            print("まずデータセットを生成してください")
            return
        
        categories = {}
        difficulties = {}
        
        for problem in self.problems:
            categories[problem.category] = categories.get(problem.category, 0) + 1
            difficulties[problem.difficulty] = difficulties.get(problem.difficulty, 0) + 1
        
        print(f"\n=== データセット統計 ===")
        print(f"総問題数: {len(self.problems)}")
        print(f"\nカテゴリ別:")
        for category, count in categories.items():
            print(f"  {category}: {count}問")
        print(f"\n難易度別:")
        for difficulty, count in difficulties.items():
            print(f"  {difficulty}: {count}問")


if __name__ == "__main__":
    # 数学データセット生成器を作成
    generator = MathDatasetGenerator()
    
    # データセットを生成
    problems = generator.generate_dataset(
        arithmetic_count=100,
        algebra_count=50,
        geometry_count=30,
        word_count=20,
        llm_count=5
    )
    
    # 統計情報を表示
    generator.get_statistics()
    
    # データセットを保存
    generator.save_dataset("math_dataset.json")
    
    # サンプル問題を表示
    print(f"\n=== サンプル問題 ===")
    for i, problem in enumerate(problems[:5]):
        print(f"{i+1}. 【{problem.category}】{problem.question}")
        print(f"   答え: {problem.answer} (難易度: {problem.difficulty})")
        print()
