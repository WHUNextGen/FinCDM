import json
import re
import os
import time
from typing import List, Dict, Any
import logging
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API configuration
client = OpenAI(
    api_key="your-api-key-here",  # Replace with your actual API key
    base_url="your-api-base-url-here",  # Replace with your API base URL
)

# Regex filter class


class RegexFilter:
    def __init__(self, regex_pattern: str = r"(A|B|C|D)", group_select=0, fallback: str = "[invalid]"):
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resp):
        match = self.regex.findall(resp)
        if match:
            match = match[self.group_select]
            if isinstance(match, tuple):
                match = [m for m in match if m]
                match = match[0] if match else self.fallback
            return match.strip()
        return self.fallback

# Load JSON file


def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"JSON file {file_path} does not exist")
        raise
    except json.JSONDecodeError:
        logger.error(f"JSON file {file_path} format error")
        raise

# Load checkpoint file


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Checkpoint file {checkpoint_path} format error, restarting from beginning")
    return {'results': [], 'overall_accuracy': 0, 'trial_accuracies': []}

# Save checkpoint


def save_checkpoint(checkpoint_path: str, results: List[Dict], overall_accuracy: float, trial_accuracies: List[Dict]):
    checkpoint_data = {
        'results': results,
        'overall_accuracy': overall_accuracy,
        'trial_accuracies': trial_accuracies
    }
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

# API call function with retry mechanism


def call_api(request_data: Dict[str, Any], retry: int = 1) -> str:
    for attempt in range(retry):
        try:
            completion = client.chat.completions.create(
                model=request_data['model'],
                messages=request_data['messages'],
                stream=False,
                temperature=1.0,
                max_tokens=64,
                presence_penalty=0,
                frequency_penalty=0,
                top_p=1,
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_msg = f"[API Error: {str(e)}]"
            if attempt < retry - 1:
                logger.warning(
                    f"Retry {attempt + 1}/{retry}, waiting 1 second...")
                time.sleep(1)
            else:
                return error_msg

# Process single trial


def run_trial(query: str, trial_idx: int, filter: RegexFilter, gold_answer: str, model: str) -> Dict[str, Any]:
    request_data = {
        'messages': [
            {'role': 'user', 'content': query}
        ],
        'model': model,
        "stream": False,
    }

    raw_output = call_api(request_data)
    raw_output = raw_output.replace(query, "")
    predicted_answer = filter.apply(raw_output)

    return {
        'trial': trial_idx + 1,
        'raw_output': raw_output,
        'predicted_answer': predicted_answer,
        'gold_answer': gold_answer,
        'correct': 1 if predicted_answer == gold_answer else 0
    }

# Evaluation function


def evaluate(json_file_path: str, num_trials: int, model: str):
    data = load_json(json_file_path)
    filter = RegexFilter()
    results = []

    # Load checkpoint
    checkpoint_path = f'evaluation_results_{model}.json'
    checkpoint = load_checkpoint(checkpoint_path)
    completed_ids = {item['id'] for item in checkpoint['results']}
    results = checkpoint['results']

    nums = 0
    for item in data:
        if item['id'] in completed_ids:
            logger.info(
                f"Question ID {item['id']} already evaluated, skipping")
            continue

        nums += 1
        logger.info(
            f'Evaluating model {model}, question {nums}/{len(data)}, ID: {item["id"]}')
        query = item['query']
        gold_answer = item['answer']
        item_results = []

        # Run multiple trials sequentially
        for trial in range(num_trials):
            result = run_trial(query, trial, filter, gold_answer, model)
            item_results.append(result)

        # Save results for this question
        results.append({
            'id': item['id'],
            'query': query,
            'trials': item_results
        })

        # Calculate current overall accuracy
        total_trials = sum(len(item['trials']) for item in results)
        correct_count = sum(
            1 for item in results for trial in item['trials'] if trial['correct'])
        overall_accuracy = correct_count / total_trials if total_trials else 0

        # Calculate accuracy for each trial round
        trial_accuracies = []
        for trial_idx in range(num_trials):
            trial_correct = sum(
                1 for item in results if item['trials'][trial_idx]['correct'])
            trial_total = len(results)
            trial_accuracy = trial_correct / trial_total if trial_total else 0
            trial_accuracies.append({
                'trial': trial_idx + 1,
                'accuracy': trial_accuracy
            })

        # Save checkpoint
        save_checkpoint(checkpoint_path, results,
                        overall_accuracy, trial_accuracies)

    return results, overall_accuracy

# Evaluate multiple models consecutively


def evaluate_models(json_file_path: str, num_trials: int, models: List[str]):
    for model in models:
        logger.info(f"Starting evaluation for model: {model}")
        results, accuracy = evaluate(json_file_path, num_trials, model)
        logger.info(
            f"Model {model} evaluation completed, overall accuracy: {accuracy:.2%}")
        for item in results:
            logger.info(f"\nQuestion ID: {item['id']}")
            for trial in item['trials']:
                logger.info(f"  Trial {trial['trial']} - Predicted: {trial['predicted_answer']}, "
                            f"Gold answer: {trial['gold_answer']}, Correct: {trial['correct']}, "
                            f"Raw output: {trial['raw_output']}")


# Execute evaluation
if __name__ == "__main__":
    json_file_path = 'path/to/your/test.json'  # Replace with your JSON file path
    num_trials = 10
    # Replace with your model list
    models = ['model-name-1', 'model-name-2', 'model-name-3']

    evaluate_models(json_file_path, num_trials, models)
