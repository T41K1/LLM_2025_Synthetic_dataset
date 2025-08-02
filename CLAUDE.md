# CLAUDE.md

以下には、日本語で答えてください.
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **math problem dataset generator** for creating educational datasets. The main purpose is to programmatically generate various types of mathematical problems suitable for language model training or educational applications.

## Architecture

- **Core Class**: `MathDatasetGenerator` in `datasets_make.py` handles all problem generation
- **Data Structure**: `MathProblem` dataclass represents individual problems with question, answer, difficulty, and category
- **Problem Categories**: 
  - Arithmetic (basic operations)
  - Algebra (linear/quadratic equations) 
  - Geometry (area calculations)
  - Word problems (Japanese language scenarios)
  - LLM-generated (framework for AI integration)

## Development Commands

```bash
# Run the dataset generator
python datasets_make.py

# Setup virtual environment (if needed)
uv venv
source .venv/bin/activate
```

## Key Implementation Notes

- **Language**: Problem text is in Japanese, code/comments in English
- **Dependencies**: Pure standard library (no external packages)
- **Output**: UTF-8 encoded JSON format
- **Package Manager**: UV (not pip)
- **Python Version**: Requires 3.13+

## LLM Integration

The `generate_llm_problems()` method includes a framework for OpenAI API integration but currently uses mock data. To enable real LLM generation:
1. Install openai package: `uv add openai`
2. Set API key in environment
3. Uncomment the API call code in `datasets_make.py:221-227`

## Dataset Configuration

Problem counts are configurable in the `generate_dataset()` method call:
- `arithmetic_count`: Basic math operations (default: 100)
- `algebra_count`: Equations (default: 50) 
- `geometry_count`: Area problems (default: 30)
- `word_count`: Story problems (default: 20)
- `llm_count`: AI-generated problems (default: 10)