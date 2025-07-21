
import os, re
from dotenv import load_dotenv
import cmbagent
import os
import re
import numpy as np
import pandas as pd
import copy
import glob
import json
import cmbagent
from human_eval.data import write_jsonl, read_problems, HUMAN_EVAL, stream_jsonl
from human_eval.evaluation import evaluate_functional_correctness, estimate_pass_at_k
from human_eval.execution import check_correctness

from dotenv import load_dotenv

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools
import tqdm


def load_key():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

def my_agent(task, api_key, outdir):
    return cmbagent.one_shot(
        task=task,
        agent="engineer",
        engineer_model="gpt-4o-mini",
        api_keys={"OPENAI": api_key},
        work_dir= outdir,
    )

def extract_code(result, function_signature):
    for message in result.get("chat_history", []):
        if message.get("name") == "engineer" and "```python" in message.get("content", ""):
            match = re.search(r"```python(.*?)```", message["content"], re.DOTALL)
            if match:
                code = match.group(1).strip()
                if not code.startswith("def "):
                    code = function_signature.rstrip() + "\n" + code
                return code
    print("CMBagent failed to create an output")
    return None

def benchmark_problem(i, problem, n_completions):

    comps = []
    api_key = load_key()
    completions = []

    outdir = os.path.abspath(f"cmbagent_output/problem_{i}")
    os.makedirs(os.path.join(outdir, "cost"), exist_ok=True)

    for j in range(n_completions):
        res = my_agent(problem["prompt"], api_key, outdir)
        completions.append(extract_code(res, problem["prompt"]))
    return i, completions



def evaluate_functional_correctness( # The same modified function as specified within my notebooks
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
    num_problems: int = 5 # I added this in to simply limit number of problems evaluated for cmbagent
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """


    problems = read_problems(problem_file)

    problems_items = list(problems.items())[:num_problems]  # I added this in to simply limit number of problems evaluated for cmbagent
    problems = dict(problems_items) # I added this in to simply limit number of problems evaluated for cmbagent


    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k
