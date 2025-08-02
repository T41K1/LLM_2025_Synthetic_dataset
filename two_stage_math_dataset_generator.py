# -*- coding: utf-8 -*-
"""
Two-Stage Math Dataset Generator
First generates problems, then adds solutions separately
"""

import json
import random
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


@dataclass
class MathProblem:
    """Data class for math problems"""
    problem_id: str
    instruction: str
    input: str
    category: str
    difficulty: str
    generated_at: float
    model_used: str
    status: str = "problem_only"  # "problem_only", "solved", "failed"


@dataclass
class CompletePair:
    """Data class for complete instruction-response pairs"""
    problem_id: str
    instruction: str
    input: str
    output: str
    category: str
    difficulty: str
    problem_generated_at: float
    solution_generated_at: float
    problem_model: str
    solution_model: str


class TwoStageMathDatasetGenerator:
    """Two-stage math dataset generator with configurable models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration"""
        self.config = config or self._default_config()
        self.tokenizer = None
        self.model = None
        self.current_model_name = None
        self.seed_prompts = self._load_seed_prompts()
        self.instruction_templates = self._load_instruction_templates()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "models": {
                "deepseek": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                "qwen": "Qwen/Qwen2.5-7B-Instruct",
                "llama": "meta-llama/Llama-3.1-8B-Instruct"
            },
            "default_model": "deepseek",
            "generation_params": {
                "problem": {
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.95
                },
                "solution": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.95
                }
            },
            "batch_size": 10,
            "sleep_interval": 0.1
        }
    
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
    
    def load_model(self, model_key: str = None):
        """Load model and tokenizer"""
        if model_key is None:
            model_key = self.config["default_model"]
        
        model_name = self.config["models"].get(model_key)
        if not model_name:
            raise ValueError(f"Model key '{model_key}' not found in config")
        
        if self.current_model_name == model_name:
            print(f"Model {model_name} already loaded")
            return
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Inference mode
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model.to("cuda")
            print("Using GPU")
        else:
            print("Using CPU")
        
        self.current_model_name = model_name
        print(f"Model loaded: {model_name}")
    
    def switch_model(self, model_key: str):
        """Switch to a different model"""
        print(f"Switching to model: {model_key}")
        self.load_model(model_key)
    
    def _generate_text(self, messages: List[Dict[str, str]], generation_type: str = "problem") -> str:
        """Common text generation processing"""
        if self.model is None:
            self.load_model()
        
        try:
            params = self.config["generation_params"][generation_type]
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=params["max_new_tokens"],
                do_sample=True,
                temperature=params["temperature"],
                top_p=params["top_p"],
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
    
    def generate_single_problem(self, category: str, problem_id: str) -> Optional[MathProblem]:
        """Generate a single math problem"""
        category_seeds = [seed for seed in self.seed_prompts if seed['category'] == category]
        
        # Format seed examples as text
        examples_text = "\n".join([
            f"Example {i+1}: {ex['seed']}" 
            for i, ex in enumerate(category_seeds[:3])
        ])
        
        system_prompt = f"""You are a math problem creator. Based on the given examples, create 1 new high school level math problem in the {category} field.

Examples:
{examples_text}

Requirements:
1. Appropriate difficulty for high school students
2. New problem similar to but different from the examples above
3. Clear and understandable problem statement
4. Calculable problem with a definite answer
5. IMPORTANT: Only output the problem statement itself, no solutions, explanations, or thinking process

Create 1 new {category} problem (problem statement only):"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Create a new problem."}
        ]
        
        problem_text = self._generate_text(messages, "problem")
        
        if not problem_text or len(problem_text.strip()) < 10:
            return None
        
        # Select random instruction template
        instruction_template = random.choice(self.instruction_templates)
        
        return MathProblem(
            problem_id=problem_id,
            instruction=instruction_template,
            input=problem_text.strip(),
            category=category,
            difficulty="high school level",
            generated_at=time.time(),
            model_used=self.current_model_name,
            status="problem_only"
        )
    
    def generate_problem_bank(self, num_problems_per_category: int = 10) -> List[MathProblem]:
        """Generate bank of problems (Stage 1)"""
        print("=== STAGE 1: Generating Problem Bank ===")
        
        problem_bank = []
        categories = list(set([seed['category'] for seed in self.seed_prompts]))
        
        for category in categories:
            print(f"\nGenerating {num_problems_per_category} {category} problems...")
            
            for i in range(num_problems_per_category):
                try:
                    problem_id = f"{category}_{i+1:03d}_{int(time.time())}"
                    problem = self.generate_single_problem(category, problem_id)
                    
                    if problem:
                        problem_bank.append(problem)
                        print(f"  Generated: {problem.problem_id}")
                    
                    # Sleep for GPU memory management
                    time.sleep(self.config["sleep_interval"])
                    
                except Exception as e:
                    print(f"Error generating problem: {e}")
                    continue
        
        print(f"\nStage 1 Complete: Generated {len(problem_bank)} problems")
        return problem_bank
    
    def solve_single_problem(self, problem: MathProblem) -> Optional[str]:
        """Generate solution for a single problem"""
        full_instruction = f"{problem.instruction}\n\n{problem.input}"
        
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
        
        return self._generate_text(messages, "solution")
    
    def generate_solutions(self, problem_bank: List[MathProblem], 
                          model_key: str = None) -> List[CompletePair]:
        """Generate solutions for all problems (Stage 2)"""
        print("=== STAGE 2: Generating Solutions ===")
        
        if model_key:
            self.switch_model(model_key)
        
        completed_pairs = []
        
        for i, problem in enumerate(problem_bank):
            try:
                print(f"\nSolving problem {i+1}/{len(problem_bank)}: {problem.problem_id}")
                
                solution = self.solve_single_problem(problem)
                
                if solution and self._validate_solution(solution):
                    pair = CompletePair(
                        problem_id=problem.problem_id,
                        instruction=problem.instruction,
                        input=problem.input,
                        output=solution,
                        category=problem.category,
                        difficulty=problem.difficulty,
                        problem_generated_at=problem.generated_at,
                        solution_generated_at=time.time(),
                        problem_model=problem.model_used,
                        solution_model=self.current_model_name
                    )
                    
                    completed_pairs.append(pair)
                    print(f"  ✓ Solution generated for {problem.problem_id}")
                else:
                    print(f"  ✗ Failed to generate valid solution for {problem.problem_id}")
                
                # Sleep for GPU memory management
                time.sleep(self.config["sleep_interval"])
                
            except Exception as e:
                print(f"Error solving problem {problem.problem_id}: {e}")
                continue
        
        print(f"\nStage 2 Complete: Generated {len(completed_pairs)} complete pairs")
        return completed_pairs
    
    def _validate_solution(self, solution: str) -> bool:
        """Validate solution quality"""
        if not solution or len(solution.strip()) < 20:
            return False
        
        # Check for thinking process tags
        if '<think>' not in solution or '</think>' not in solution:
            return False
        
        # Check for mathematical content
        math_keywords = [
            'calculate', 'find', 'solve', 'answer', 'equation', 'function',
            '+', '-', '×', '÷', '=', 'sin', 'cos', 'tan', '∫', '√'
        ]
        
        has_math_content = any(keyword in solution.lower() for keyword in math_keywords)
        return has_math_content
    
    def save_problem_bank(self, problem_bank: List[MathProblem], filename: str):
        """Save problem bank to JSON file"""
        problem_data = [asdict(problem) for problem in problem_bank]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(problem_data, f, ensure_ascii=False, indent=2)
        
        print(f"Problem bank saved: {filename} ({len(problem_bank)} problems)")
    
    def load_problem_bank(self, filename: str) -> List[MathProblem]:
        """Load problem bank from JSON file"""
        with open(filename, 'r', encoding='utf-8') as f:
            problem_data = json.load(f)
        
        problem_bank = [MathProblem(**data) for data in problem_data]
        print(f"Problem bank loaded: {filename} ({len(problem_bank)} problems)")
        return problem_bank
    
    def save_complete_dataset(self, completed_pairs: List[CompletePair], filename: str):
        """Save complete dataset in Alpaca format"""
        alpaca_format = []
        
        for pair in completed_pairs:
            alpaca_format.append({
                "instruction": pair.instruction,
                "input": pair.input,
                "output": pair.output,
                "metadata": {
                    "problem_id": pair.problem_id,
                    "category": pair.category,
                    "difficulty": pair.difficulty,
                    "problem_generated_at": pair.problem_generated_at,
                    "solution_generated_at": pair.solution_generated_at,
                    "problem_model": pair.problem_model,
                    "solution_model": pair.solution_model
                }
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(alpaca_format, f, ensure_ascii=False, indent=2)
        
        print(f"Complete dataset saved: {filename}")
        self._print_dataset_stats(completed_pairs)
    
    def _print_dataset_stats(self, completed_pairs: List[CompletePair]):
        """Print dataset statistics"""
        category_counts = {}
        model_combinations = {}
        
        for pair in completed_pairs:
            # Category stats
            category_counts[pair.category] = category_counts.get(pair.category, 0) + 1
            
            # Model combination stats
            combo = f"{pair.problem_model} → {pair.solution_model}"
            model_combinations[combo] = model_combinations.get(combo, 0) + 1
        
        print(f"\nDataset Statistics:")
        print(f"Total pairs: {len(completed_pairs)}")
        print(f"\nCategory breakdown:")
        for category, count in category_counts.items():
            print(f"  {category}: {count} problems")
        
        print(f"\nModel combinations:")
        for combo, count in model_combinations.items():
            print(f"  {combo}: {count} pairs")


def main():
    """Main execution function with example usage"""
    
    # Custom configuration (optional)
    config = {
        "models": {
            "deepseek": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "qwen": "Qwen/Qwen2.5-7B-Instruct"  # Add other models as needed
        },
        "default_model": "deepseek",
        "generation_params": {
            "problem": {"max_new_tokens": 150, "temperature": 0.8, "top_p": 0.9},
            "solution": {"max_new_tokens": 400, "temperature": 0.7, "top_p": 0.95}
        }
    }
    
    print("Starting Two-Stage Math Dataset Generation...")
    
    # Initialize generator
    generator = TwoStageMathDatasetGenerator(config)
    
    # Stage 1: Generate problems
    problem_bank = generator.generate_problem_bank(num_problems_per_category=5)
    generator.save_problem_bank(problem_bank, "math_problem_bank.json")
    
    # Stage 2: Generate solutions (same model)
    completed_pairs = generator.generate_solutions(problem_bank)
    generator.save_complete_dataset(completed_pairs, "math_dataset_single_model.json")
    
    # Optional: Stage 2 with different model
    # completed_pairs_mixed = generator.generate_solutions(problem_bank, model_key="qwen")
    # generator.save_complete_dataset(completed_pairs_mixed, "math_dataset_mixed_models.json")


if __name__ == "__main__":
    main()