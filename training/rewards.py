"""Reward functions for GRPO training.

Adapted from: https://github.com/knoveleng/open-rs/blob/main/src/open_r1/rewards.py

These reward functions evaluate model completions across different dimensions:
- Accuracy (correctness of answers)
- Format (proper use of tags and structure)
- Reasoning (step-by-step thinking)
- Length (efficiency and conciseness)
- Code (execution and formatting)
"""

import asyncio
import json
import math
import re
from typing import Dict, List, Callable


# Optional dependencies - gracefully handle if not installed
try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: latex2sympy2_extended and math_verify not available. Math accuracy rewards will be disabled.")

try:
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox
    load_dotenv()
    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False
    print("Warning: E2B code interpreter not available. Code execution rewards will be disabled.")


def is_math_verify_available() -> bool:
    """Check if math verification libraries are available."""
    return MATH_VERIFY_AVAILABLE


def is_e2b_available() -> bool:
    """Check if E2B code interpreter is available."""
    return E2B_AVAILABLE


# ============================================================================
# Format Conversion Helpers
# ============================================================================

def _normalize_completions(completions, **kwargs):
    """Normalize completions to List[List[Dict[str, str]]] format.

    GRPOTrainer can call reward functions with different formats:
    1. completions as List[str] - most common
    2. samples in kwargs as List[Dict]
    3. Already in List[List[Dict]] format (from other sources)

    Args:
        completions: Various formats of completions
        **kwargs: May contain 'samples' or other completion formats

    Returns:
        Normalized format: List[List[Dict[str, str]]]
    """
    # Handle samples in kwargs (like simple_reward_function does)
    if "samples" in kwargs:
        samples = kwargs["samples"]
        texts = [sample.get("response", sample.get("completion", "")) for sample in samples]
        return [[{"content": text}] for text in texts]

    # Handle None or empty completions
    if completions is None:
        completions = kwargs.get("responses", kwargs.get("completions", []))

    # If already in correct format List[List[Dict]]
    if completions and isinstance(completions[0], list):
        if completions[0] and isinstance(completions[0][0], dict):
            return completions

    # If List[str] format (most common from GRPOTrainer)
    if completions and isinstance(completions[0], str):
        return [[{"content": text}] for text in completions]

    # If List[Dict] format
    if completions and isinstance(completions[0], dict):
        if "content" in completions[0]:
            return [[comp] for comp in completions]
        # Extract text from dict
        texts = [comp.get("response", comp.get("completion", str(comp))) for comp in completions]
        return [[{"content": text}] for text in texts]

    # Fallback: try to convert to string
    return [[{"content": str(comp)}] for comp in completions]


# ============================================================================
# Accuracy Rewards
# ============================================================================

def accuracy_reward(completions=None, solution: List[str] = None, **kwargs) -> List[float]:
    """Reward function that checks if the completion matches the ground truth solution.

    Uses LaTeX parsing and verification for mathematical correctness.

    Args:
        completions: Model completions (List[str] or other formats)
        solution: List of ground truth solutions
        **kwargs: Additional arguments

    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)
    """
    if not MATH_VERIFY_AVAILABLE:
        print("Math verification not available, returning neutral rewards")
        return [0.5] * len(completions or kwargs.get("samples", []))

    # Normalize to expected format
    completions = _normalize_completions(completions, **kwargs)
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)

        rewards.append(reward)

    return rewards


# ============================================================================
# Format Rewards
# ============================================================================

def format_reward(completions=None, **kwargs) -> List[float]:
    """Reward function that checks if reasoning is enclosed in <think></think> tags.

    The reasoning process should be within <think> and </think> tags,
    while the final answer should be within <answer> and </answer> tags.

    Args:
        completions: Model completions (List[str] or other formats)
        **kwargs: Additional arguments

    Returns:
        List of rewards (1.0 if properly formatted, 0.0 otherwise)
    """
    def count_tags(text: str) -> float:
        count = 0.0
        # We only count </think> tag, because <think> tag is available in system prompt
        if text.count("\n</think>\n") == 1:
            count += 1.0
        return count

    # Normalize to expected format
    completions = _normalize_completions(completions, **kwargs)
    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def tag_count_reward(completions=None, **kwargs) -> List[float]:
    """Reward function that checks for the desired number of think and answer tags.

    Gives partial rewards for having the correct tags:
    - 0.5 for opening <think> tag
    - 0.5 for closing </think> tag

    Args:
        completions: Model completions (List[str] or other formats)
        **kwargs: Additional arguments

    Returns:
        List of rewards (0.0 to 1.0 based on tag presence)
    """
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.5
        if text.count("\n</think>\n") == 1:
            count += 0.5
        return count

    # Normalize to expected format
    completions = _normalize_completions(completions, **kwargs)
    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


# ============================================================================
# Reasoning Rewards
# ============================================================================

def reasoning_steps_reward(completions=None, **kwargs) -> List[float]:
    r"""Reward function that checks for clear step-by-step reasoning.

    Looks for reasoning indicators like:
    - Numbered steps: "Step 1:", "Step 2:", etc.
    - Numbered lists: "1.", "2.", etc.
    - Bullet points: "- " or "* "
    - Transition words: "First,", "Second,", "Next,", "Finally,"

    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words

    Args:
        completions: Model completions (List[str] or other formats)
        **kwargs: Additional arguments

    Returns:
        List of rewards (0.0 to 1.0, encourages 3+ reasoning steps)
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"

    # Normalize to expected format
    completions = _normalize_completions(completions, **kwargs)
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


# ============================================================================
# Length-based Rewards
# ============================================================================

def len_reward(completions=None, solution: List[str] = None, **kwargs) -> List[float]:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Based on the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: Model completions (List[str] or other formats)
        solution: List of ground truth solutions
        **kwargs: Additional arguments

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    if not MATH_VERIFY_AVAILABLE:
        print("Math verification not available, returning neutral rewards")
        return [0.0] * len(completions or kwargs.get("samples", []))

    # Normalize to expected format
    completions = _normalize_completions(completions, **kwargs)
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
) -> Callable:
    """Generate a reward function that scales based on completion length using a cosine schedule.

    Shorter correct solutions are rewarded more than longer ones.
    Longer incorrect solutions are penalized less than shorter ones.

    Args:
        min_value_wrong: Minimum reward for wrong answers
        max_value_wrong: Maximum reward for wrong answers
        min_value_correct: Minimum reward for correct answers
        max_value_correct: Maximum reward for correct answers
        max_len: Maximum length for scaling

    Returns:
        A reward function with the specified cosine scaling parameters
    """
    def cosine_scaled_reward(completions=None, solution: List[str] = None, **kwargs) -> List[float]:
        if not MATH_VERIFY_AVAILABLE:
            print("Math verification not available, returning neutral rewards")
            return [0.0] * len(completions or kwargs.get("samples", []))

        # Normalize to expected format
        completions = _normalize_completions(completions, **kwargs)
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


# ============================================================================
# Repetition Penalty Rewards
# ============================================================================

def get_repetition_penalty_reward(ngram_size: int, max_penalty: float) -> Callable:
    """Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.

    Reference implementation from: https://github.com/eddycmu/demystify-long-cot

    Args:
        ngram_size: Size of the n-grams
        max_penalty: Maximum (negative) penalty for wrong answers

    Returns:
        A reward function that penalizes repetitive text
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions=None, **kwargs) -> List[float]:
        """Reward function that penalizes repetitions.

        Args:
            completions: Model completions (List[str] or other formats)
            **kwargs: Additional arguments

        Returns:
            List of penalties (0.0 for no repetition, max_penalty for high repetition)
        """
        # Normalize to expected format
        completions = _normalize_completions(completions, **kwargs)
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)

        return rewards

    return repetition_penalty_reward


# ============================================================================
# Code Evaluation Rewards
# ============================================================================

def extract_code(completion: str) -> str:
    """Extract Python code from markdown code blocks.

    Args:
        completion: Model completion text

    Returns:
        Extracted code string (empty if no code found)
    """
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(completions=None, **kwargs) -> List[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: Model completions (List[str] or other formats)
        **kwargs: Must contain 'verification_info' with test cases

    Returns:
        List of rewards based on test case success rate
    """
    if not E2B_AVAILABLE:
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # Normalize to expected format
    completions = _normalize_completions(completions, **kwargs)

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    evaluation_script_template = """
import subprocess
import json

def evaluate_code(code, test_cases):
    passed = 0
    total = len(test_cases)
    exec_timeout = 5

    for case in test_cases:
        process = subprocess.run(
            ["python3", "-c", code],
            input=case["input"],
            text=True,
            capture_output=True,
            timeout=exec_timeout
        )

        if process.returncode != 0:  # Error in execution
            continue

        output = process.stdout.strip()

        # TODO: implement a proper validator to compare against ground truth.
        # For now we just check for exact string match on each line of stdout.
        all_correct = True
        for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
            all_correct = all_correct and line1.strip() == line2.strip()

        if all_correct:
            passed += 1

    success_rate = (passed / total)
    return success_rate

code_snippet = {code}
test_cases = json.loads({test_cases})

evaluate_code(code_snippet, test_cases)
"""

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(
            code=json.dumps(code),
            test_cases=json.dumps(json.dumps(info["test_cases"]))
        )
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if not all(v["language"] == language for v in verification_info):
        raise ValueError("All verification_info must have the same language", verification_info)

    try:
        rewards = run_async_from_sync(scripts, language)
    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python") -> Callable:
    """Format reward function specifically for code responses.

    Checks that responses follow the format:
    <think>
    ...reasoning...
    </think>
    <answer>
    ```python
    ...code...
    ```
    </answer>

    Args:
        language: Programming language supported by E2B
                 https://e2b.dev/docs/code-interpreting/supported-languages

    Returns:
        A reward function that checks code formatting
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions=None, **kwargs) -> List[float]:
        # Normalize to expected format
        completions = _normalize_completions(completions, **kwargs)
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


# ============================================================================
# Async Code Execution Helpers
# ============================================================================

def run_async_from_sync(scripts: List[str], language: str) -> List[float]:
    """Function wrapping the `run_async` function to execute from sync context.

    Args:
        scripts: List of code scripts to execute
        language: Programming language for execution

    Returns:
        List of execution rewards
    """
    try:
        rewards = asyncio.run(run_async(scripts, language))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: List[str], language: str) -> List[float]:
    """Asynchronously execute multiple scripts in E2B sandbox.

    Args:
        scripts: List of code scripts to execute
        language: Programming language for execution

    Returns:
        List of execution rewards
    """
    if not E2B_AVAILABLE:
        return [0.0] * len(scripts)

    # Create the sandbox by hand, currently there's no context manager for this version
    sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(sbx, script, language) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    # Kill the sandbox after all the tasks are complete
    await sbx.kill()

    return rewards


async def run_script(sbx, script: str, language: str) -> float:
    """Execute a single script in the E2B sandbox.

    Args:
        sbx: E2B sandbox instance
        script: Code script to execute
        language: Programming language

    Returns:
        Execution reward (float from script output or 0.0 on error)
    """
    execution = await sbx.run_code(script, language=language)
    try:
        return float(execution.text)
    except (TypeError, ValueError):
        return 0.0


# ============================================================================
# Convenience function for getting all reward functions
# ============================================================================

def get_all_available_rewards() -> Dict[str, Callable]:
    """Get a dictionary of all available reward functions.

    Returns:
        Dictionary mapping reward names to reward functions
    """
    rewards = {
        "format": format_reward,
        "tag_count": tag_count_reward,
        "reasoning_steps": reasoning_steps_reward,
    }

    # Add math-based rewards if available
    if MATH_VERIFY_AVAILABLE:
        rewards.update({
            "accuracy": accuracy_reward,
            "length": len_reward,
            "cosine_scaled": get_cosine_scaled_reward(),
        })

    # Add code-based rewards if available
    if E2B_AVAILABLE:
        rewards.update({
            "code": code_reward,
            "code_format": get_code_format_reward(),
        })

    # Add parameterized reward generators
    rewards.update({
        "repetition_penalty": get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.5),
    })

    return rewards


def print_available_rewards():
    """Print information about available reward functions."""
    print("Available Reward Functions:")
    print("=" * 60)

    print("\n📝 Format & Structure Rewards:")
    print("  - format_reward: Checks for proper <think></think> tags")
    print("  - tag_count_reward: Partial credit for tag presence")

    print("\n🧠 Reasoning Rewards:")
    print("  - reasoning_steps_reward: Evaluates step-by-step thinking")

    print("\n🔄 Quality Rewards:")
    print("  - get_repetition_penalty_reward: Penalizes repetitive text")

    if MATH_VERIFY_AVAILABLE:
        print("\n✅ Accuracy Rewards (requires math_verify):")
        print("  - accuracy_reward: Checks mathematical correctness")
        print("  - len_reward: Length-based efficiency (Kimi 1.5)")
        print("  - get_cosine_scaled_reward: Cosine-scaled length rewards")
    else:
        print("\n⚠️  Math accuracy rewards disabled (install latex2sympy2_extended and math_verify)")

    if E2B_AVAILABLE:
        print("\n💻 Code Rewards (requires E2B):")
        print("  - code_reward: Executes and validates code")
        print("  - get_code_format_reward: Checks code formatting")
    else:
        print("\n⚠️  Code execution rewards disabled (install e2b-code-interpreter)")

    print("\n" + "=" * 60)