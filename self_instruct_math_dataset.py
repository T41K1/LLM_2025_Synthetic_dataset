# -*- coding: utf-8 -*-
"""
Self-Instruct Method for Creating SFT High School Math Dataset
"""

import json
import random
import time
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


@dataclass
class InstructionResponsePair:
    """Data class for instruction-response pairs"""
    instruction: str
    input: str
    output: str
    category: str
    difficulty: str
    metadata: Dict[str, Any] = None


class SelfInstructMathDatasetGenerator:
    """Generate SFT dataset for high school math using Self-Instruct method"""
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
        """Initialize the generator"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.seed_prompts = self._load_seed_prompts()
        self.instruction_templates = self._load_instruction_templates()
        
    def _load_seed_prompts(self) -> List[Dict[str, str]]:
        """Define seed prompts for high school mathematics"""
        return [
            {
                "category": "algebra",
                "seed": "Solve the quadratic equation x² + 3x - 4 = 0.",
                "template": "quadratic equation solving problem"
            },
            {
                "category": "algebra", 
                "seed": "Solve the system of equations: x + y = 5, 2x - y = 1.",
                "template": "system of equations solving problem"
            },
            {
                "category": "geometry",
                "seed": "Find the area of a circle with radius 5 cm.",
                "template": "area and volume calculation problem"
            },
            {
                "category": "geometry",
                "seed": "In a right triangle with base 6 cm and height 8 cm, find the length of the hypotenuse.",
                "template": "triangle side length problem"
            },
            {
                "category": "trigonometry",
                "seed": "Find the value of sin 30°.",
                "template": "trigonometric function value problem"
            },
            {
                "category": "trigonometry",
                "seed": "Prove that sin²θ + cos²θ = 1.",
                "template": "trigonometric identity proof problem"
            },
            {
                "category": "probability",
                "seed": "Find the probability that the sum of two dice rolls equals 7.",
                "template": "probability calculation problem"
            },
            {
                "category": "statistics",
                "seed": "Find the mean of the test scores: {80, 75, 90, 85, 70}.",
                "template": "mean, variance, and standard deviation problem"
            },
            {
                "category": "calculus",
                "seed": "Find the derivative of f(x) = x³ - 2x² + x - 1.",
                "template": "function derivative problem"
            },
            {
                "category": "calculus",
                "seed": "Evaluate the integral ∫(2x + 3)dx.",
                "template": "indefinite and definite integral problem"
            }
        ]
    
    def _load_instruction_templates(self) -> List[str]:
        """Define templates for instruction generation"""
        return [
            "Solve the following problem:",
            "Answer the following math question:",
            "Perform the following calculation:",
            "Solve this problem step by step:",
            "Solve the following problem with detailed explanation:",
            "Solve the following problem and explain your reasoning:",
            "Solve and explain the following mathematical concept:",
            "Solve this problem and show the formulas or theorems used:"
        ]
    
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # Inference mode
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model.to("cuda")
            print("Using GPU")
        else:
            print("Using CPU")
    
    def generate_new_instruction(self, seed_examples: List[Dict[str, str]], category: str) -> str:
        """Generate new instruction from seed examples"""
        
        # Format seed examples as text
        examples_text = "\n".join([
            f"Example {i+1}: {ex['seed']}" 
            for i, ex in enumerate(seed_examples[:3])
        ])
        
        system_prompt = f"""You are a math problem creator. Based on the given examples, create one new high school level math problem in the {category} field.

Examples:
{examples_text}

Requirements:
1. Appropriate difficulty for high school students  
2. New problem similar to but different from the examples above  
3. Clear and understandable problem statement  
4. Calculable problem with a definite answer  
5. When you think, wrap your reasoning in `<think>...</think>` tags.  
6. After your `<think>`...`</think>`, output **only** the final problem statement (no explanations).

Your response must follow this exact format:

<think>
...your internal reasoning here...
</think>
<problem>
...just the problem statement here...
</problem>

Create one new {category} problem:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Create a new problem."}
        ]
        
        generated_text = self._generate_text(messages, max_tokens=512)
        
        # Extract problem from <problem> tags
        return self._extract_problem_from_tags(generated_text)
    
    def _extract_problem_from_tags(self, text: str) -> str:
        """Extract problem statement from <problem>...</problem> tags"""
        
        if not text:
            return ""
        
        # Look for <problem>...</problem> tags
        pattern = r'<problem>(.*?)</problem>'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            problem_text = match.group(1).strip()
            print(f"✅ Extracted problem: {problem_text[:100]}...")
            return problem_text
        else:
            print(f"⚠️  No <problem> tags found, returning raw text: {text[:100]}...")
            # Fallback: return the text after </think> if it exists
            if '</think>' in text:
                parts = text.split('</think>', 1)
                if len(parts) > 1:
                    return parts[1].strip()
            return text.strip()
    
    def generate_response(self, instruction: str, math_problem: str) -> str:
        """Generate response to instruction and problem"""
        
        full_instruction = f"{instruction}\n\n{math_problem}"
        
        system_prompt = """You are an excellent math teacher. Solve problems step by step in a way that high school students can understand.

Answer format:
<think>
1. Understanding the problem
2. Explanation of solution method
3. Calculation process
</think>

Final answer: [Your final answer here]

IMPORTANT: Always include your thinking process inside <think>...</think> tags, followed by the final answer."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_instruction}
        ]
        
        return self._generate_text(messages, max_tokens=1024)
    
    def _generate_text(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        """Common text generation processing"""
        try:
            #文字列からtokenにする
            inputs = self.tokenizer.apply_chat_template(
                messages, #inputとsystem promptを含むメッセージ
                add_generation_prompt=True, #整形後の文字列を生成プロンプトとして使用
                return_dict=True, #辞書形式で返す
                return_tensors="pt" #PyTorchのテンソル形式で返す
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            response = self.tokenizer.decode(
                outputs[0, inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True,
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def filter_and_validate(self, pair: InstructionResponsePair) -> bool:
        """Check quality of generated pairs"""
        
        # Minimum length check
        #あまりにも短いペアは除外
        if len(pair.instruction) < 10 or len(pair.output) < 20:
            return False
        
        #問題文に解答や思考過程が含まれていないかチェック
        # Check if input contains only problem statement (no solutions)
        problem_input = pair.input.lower()
        problematic_words = ['<think>', 'solution:', 'answer:', 'step 1:', 'first,', '<problem>']
        found_words = [word for word in problematic_words if word in problem_input]
        if found_words:
            print(f"⚠️  WARNING: Input contains problematic words: {found_words} (allowing for now)")
            # return False  # Temporarily disabled for debugging
        
        #出力に思考過程の<think>が存在するかどうか
        # Check if output contains thinking process
        if '<think>' not in pair.output or '</think>' not in pair.output:
            return False
        
        # Mathematical content check (simplified)
        math_keywords = [
            'calculate', 'find', 'solve', 'prove', 'answer', 'equation', 'function', 'derivative', 'integral',
            'triangle', 'probability', 'statistics', 'geometry', 'algebra', '×', '÷', '+', '-', '=',
            'sin', 'cos', 'tan', '∫', '∑', '√', '²', '³'
        ]
        
        #上記のようなキーワードが問題文と解答に含まれているかチェック
        has_math_content = any(keyword in pair.input + pair.output for keyword in math_keywords)
        
        if not has_math_content:
            return False
        
        #問題文の丸写しがないかチェック
        # Duplication check (simplified)
        if pair.instruction.lower() in pair.output.lower():
            return False
        
        return True
    
    def generate_diverse_pairs(self, num_pairs: int = 100) -> List[InstructionResponsePair]:
        """Generate diverse instruction-response pairs"""
        
        if self.model is None:
            self.load_model()
        
        #seedで与えられた問題を元に多様なペアを生成する。均等に各カテゴリから生成する。
        generated_pairs = []
        categories = list(set([seed['category'] for seed in self.seed_prompts]))
        #何回各カテゴリからペアを生成するか
        pairs_per_category = num_pairs // len(categories)
        
        for category in categories:
            print(f"\nGenerating {category} category problems...")
            category_seeds = [seed for seed in self.seed_prompts if seed['category'] == category]
            
            for _ in range(pairs_per_category):
                try:
                    # Select random instruction template
                    instruction_template = random.choice(self.instruction_templates)
                    
                    # Generate new problem
                    new_problem = self.generate_new_instruction(category_seeds, category)
                    
                    if not new_problem:
                        continue
                    
                    # Generate response
                    response = self.generate_response(instruction_template, new_problem)
                    
                    if not response:
                        continue
                    
                    # Create pair
                    pair = InstructionResponsePair(
                        instruction=instruction_template,
                        input=new_problem,
                        output=response,
                        category=category,
                        difficulty="high school level",
                        metadata={
                            "generated_at": time.time(),
                            "model": self.model_name
                        }
                    )
                    
                    # Quality check
                    if self.filter_and_validate(pair):
                        generated_pairs.append(pair)
                        print(f"  {category} {len(generated_pairs)}/{num_pairs} completed")
                    
                    # Wait a bit for GPU memory management
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Pair generation error: {e}")
                    continue
        
        return generated_pairs
    
    def save_dataset(self, pairs: List[InstructionResponsePair], filename: str):
        """Save dataset to JSON file"""
        
        # Convert to Alpaca format
        alpaca_format = []
        for pair in pairs:
            alpaca_format.append({
                "instruction": pair.instruction,
                "input": pair.input,
                "output": pair.output,
                "category": pair.category,
                "difficulty": pair.difficulty,
                "metadata": pair.metadata
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(alpaca_format, f, ensure_ascii=False, indent=2)
        
        print(f"\nDataset saved: {filename}")
        print(f"Total pairs: {len(pairs)}")
        
        # Category statistics
        category_counts = {}
        for pair in pairs:
            category_counts[pair.category] = category_counts.get(pair.category, 0) + 1
        
        print("\nCategory breakdown:")
        for category, count in category_counts.items():
            print(f"  {category}: {count} problems")


def main():
    """Main execution function"""
    
    print("Starting Self-Instruct high school math SFT dataset generation...")
    
    # Initialize generator
    generator = SelfInstructMathDatasetGenerator()
    
    # Generate dataset
    pairs = generator.generate_diverse_pairs(num_pairs=5)  # Start with smaller number
    
    # Save
    if pairs:
        generator.save_dataset(pairs, "self_instruct_math_dataset.json")
    else:
        print("No valid pairs were generated.")


if __name__ == "__main__":
    main()