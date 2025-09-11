# Model Evaluation Scripts

This repository contains two evaluation scripts for testing language models on multiple-choice questions with repeated trials to measure consistency and accuracy.

## Overview

Both scripts are designed to evaluate models on JSON-formatted datasets containing multiple-choice questions (A, B, C, D format). They perform multiple trials per question to assess model consistency and calculate both overall and per-trial accuracy metrics.

## Files

- `api-eval.py`: Evaluation script for API-based language models
- `local-eval.py`: Evaluation script for locally hosted models using GPU inference

## Data Format

Both scripts expect a JSON file with the following structure:

```json
[
  {
    "id": "question_1",
    "query": "What is the capital of France?\nA) London\nB) Berlin\nC) Paris\nD) Madrid",
    "answer": "C"
  },
  {
    "id": "question_2",
    "query": "Which of the following is a programming language?\nA) HTML\nB) Python\nC) CSS\nD) JSON",
    "answer": "B"
  }
]
```

## API Evaluation Script (`api-eval.py`)

### Prerequisites

```bash
pip install openai
```

### Configuration

Before running, update the following in the script:

1. **API Configuration** :

```python
   client = OpenAI(
       api_key="your-api-key-here",  # Replace with your actual API key
       base_url="your-api-base-url-here",  # Replace with your API base URL
   )
```

1. **File Paths and Models** :

```python
   json_file_path = 'path/to/your/test.json'  # Path to your test data
   models = ['model-name-1', 'model-name-2', 'model-name-3']  # List of models to evaluate
```

### Usage

```bash
python api-eval.py
```

### Features

- **Checkpoint System** : Automatically saves progress and can resume from interruptions
- **Multiple Model Support** : Evaluate multiple models in sequence
- **Retry Mechanism** : Handles API failures with automatic retries
- **Detailed Logging** : Comprehensive logging of evaluation progress and results

### Output

- Creates `evaluation_results_{model_name}.json` for each model
- Contains detailed trial results, overall accuracy, and per-trial accuracy metrics

## Local GPU Evaluation Script (`local-eval.py`)

### Prerequisites

```bash
pip install torch transformers
```

### Configuration

Before running, update the following in the script:

1. **Model Configuration** :

```python
   base_model = 'path/to/your/model'  # Replace with your model path
   device_map = 'cuda:0'  # Change to your preferred device
```

1. **File Paths** :

```python
   json_file_path = 'path/to/your/test.json'  # Replace with your JSON file path
   model_name = "your_model_name"  # Replace with your model name for output file
```

### Usage

```bash
python local-eval.py
```

### Features

- **GPU Acceleration** : Utilizes CUDA for fast local inference
- **Temperature Sampling** : Uses temperature=1.0 for varied responses across trials
- **Memory Efficient** : Uses bfloat16 precision to reduce memory usage
- **Flexible Device Selection** : Easy configuration for different GPU devices

### Output

- Creates `{model_name}_evaluation_results.json`
- Contains detailed trial results, overall accuracy, and per-trial accuracy metrics

## Output Format

Both scripts generate JSON files with the following structure:

```json
{
  "results": [
    {
      "id": "question_1",
      "query": "What is the capital of France?...",
      "trials": [
        {
          "trial": 1,
          "raw_output": "The answer is C) Paris",
          "predicted_answer": "C",
          "gold_answer": "C",
          "correct": 1
        }
      ]
    }
  ],
  "overall_accuracy": 0.85,
  "trial_accuracies": [
    {
      "trial": 1,
      "accuracy": 0.87
    }
  ]
}
```

## Key Features

### Regex Filtering

Both scripts use a regex filter to extract answers from model outputs:

- Looks for patterns matching A, B, C, or D
- Handles various output formats
- Returns "[invalid]" for unparseable responses

### Multiple Trials

- Default: 10 trials per question
- Measures consistency across multiple runs
- Provides both overall and per-trial accuracy metrics

### Error Handling

- **API Script** : Retry mechanism for failed API calls
- **Local Script** : Proper GPU memory management
- Both: Graceful handling of malformed responses

## Customization

### Changing Answer Format

To evaluate models with different answer formats, modify the `RegexFilter` pattern:

```python
# For yes/no questions
filter = RegexFilter(regex_pattern=r"(Yes|No)", fallback="[invalid]")

# For numbered options
filter = RegexFilter(regex_pattern=r"([1-4])", fallback="[invalid]")
```

### Adjusting Generation Parameters

Modify generation parameters in the respective scripts:

**API Script** :

```python
completion = client.chat.completions.create(
    model=request_data['model'],
    messages=request_data['messages'],
    temperature=1.0,  # Adjust temperature
    max_tokens=64,    # Adjust max tokens
    # ... other parameters
)
```

**Local Script** :

```python
pred = model.generate(
    **inputs,
    max_new_tokens=64,      # Adjust max tokens
    temperature=1.0,        # Adjust temperature
    repetition_penalty=1.0, # Adjust repetition penalty
    # ... other parameters
)
```

## Privacy and Security

These scripts have been designed with privacy in mind:

- No hardcoded API keys or personal information
- Configurable file paths and model names
- Clean, reusable code suitable for open-source distribution

## Troubleshooting

### Common Issues

1. **API Script** :

- Check API key and base URL configuration
- Verify network connectivity
- Monitor rate limits

1. **Local Script** :

- Ensure sufficient GPU memory
- Check CUDA availability: `torch.cuda.is_available()`
- Verify model path and format compatibility

1. **Both Scripts** :

- Validate JSON data format
- Check file permissions
- Ensure required dependencies are installed
