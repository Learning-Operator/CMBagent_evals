import os
import json
import cmbagent
from typing import Literal
import pandas as pd
from pandas import DataFrame
from typing import Optional
from datasets import DatasetDict, Dataset
from datasets import load_dataset
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed 
import re
import time
import traceback

import math

def store_results(results: dict, filename: str = "results.json"):
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Stored results to {path}")

def run_eval_agent(api_key: str, correct: str, test: str, max_retries:int=3):

    client = OpenAI(api_key=api_key)
    
    for attempt in range(max_retries):
            new_assistant = client.beta.assistants.create(
                name="Physics_Evaluator",
                instructions="""
                
                You are a physics problem evaluation expert. Your task is to determine if a solution matches the correct answer for a theoretical physics problem.

                EVALUATION CRITERIA:
                4. Compare the mathematical expressions carefully
                5. Consider equivalent forms (e.g., different variable names, algebraic manipulations)
                6. Focus on the core physics and mathematical content, not formatting
                7. If expressions are mathematically equivalent, consider them correct
                8. Ignore minor formatting differences like spacing or LaTeX syntax variations.
                9 if the resultant answer is "null", you are to score the answer as wrong.

                IMPORTANT: You must respond with EXACTLY ONE NUMBER:
                - Output "1" if the solutions are mathematically equivalent/correct
                - Output "0" if the solutions are different/incorrect

                Do not provide any explanation, just the number 1 or 0.""",

                model="gpt-4o-mini", 
                temperature=0.3, 
                top_p=1,
            )
            
            thread = client.beta.threads.create(messages=[])
            
            prompt = f"""
            PHYSICS PROBLEM EVALUATION

            CORRECT ANSWER (Ground Truth):
            {correct}

            Model's output:
            {test}

            INSTRUCTIONS:
            - Look for <ANSWER> tags, \\boxed{{}} expressions, or equation environments
            - Compare the found answer with the correct answer for mathematical equivalence
            - Consider different but equivalent forms as correct
            - Focus on the mathematical content, not formatting
            - Ignore LaTeX formatting differences

            RESPOND WITH EXACTLY: 1 (if equivalent) or 0 (if different)
            """

            client.beta.threads.messages.create(
                thread_id=thread.id,
                content=prompt,
                role='user',
            )
            
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=new_assistant.id,
                instructions="Evaluate if the physics solutions are mathematically equivalent by searching through all 3 messages",
            )
            
            timeout = 60 
            start_time = time.time()
            
            while run.status not in ["completed", "failed", "cancelled", "expired"]:
                if time.time() - start_time > timeout:
                    print(f"Evaluation timeout on attempt {attempt + 1}")
                    break
                    
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

            if run.status == "completed":
                messages = client.beta.threads.messages.list(
                    thread_id=thread.id,
                    order="desc",
                    limit=1
                )
                
                content = messages.data[0].content[0].text.value
                
                client.beta.threads.delete(thread.id)
                client.beta.assistants.delete(new_assistant.id)
                
                return content
            else:
                print(f"Run failed with status: {run.status}")
                try:
                    client.beta.threads.delete(thread.id)
                    client.beta.assistants.delete(new_assistant.id)
                except:
                    pass
    
    return "0"  

def run_mas(i: int, 
            model: str, 
            prompt: str, 
            mode: Literal['one_shot', 'chat', 'planning_and_control', 'planning_and_control_context_carryover'],
            api_key: str = 'your-api-key',
            work_dir: str = './outputs'
            ):

    full_work_dir = os.path.join(work_dir, f"problem_{i}")
    os.makedirs(full_work_dir, exist_ok=True)
    os.makedirs(os.path.join(full_work_dir, 'time'), exist_ok=True)
    
    api_keys = {'OPENAI': api_key}
    
    if mode == 'one_shot':
        result = cmbagent.one_shot(
            task=prompt,
            engineer_model=model,
            api_keys=api_keys,
            work_dir=full_work_dir
        )

    elif mode == 'chat':
        result = cmbagent.chat(
            task=prompt,
            engineer_model=model,
            api_keys=api_keys,
            work_dir=full_work_dir
        )

    elif mode == 'planning_and_control':
        result = cmbagent.planning_and_control(
            task=prompt,
            api_keys=api_keys,
            planner_model=model,          
            plan_reviewer_model=model,    
            engineer_model=model,        
            researcher_model=model,
            work_dir=full_work_dir
        )

    elif mode == 'planning_and_control_context_carryover':
        result = cmbagent.planning_and_control_context_carryover(
            task=prompt,
            api_keys=api_keys,
            planner_model=model,          
            plan_reviewer_model=model,    
            engineer_model=model,        
            researcher_model=model,
            work_dir=full_work_dir
        )

    else:
        raise ValueError(f'Invalid mode: {mode}')
        
    print(f"MAS completed for problem {i}")
    return result


def run_all_benchmarks(
    model: str = 'gpt-4o-mini',
    n_samples: int = 30,
    n_workers: int = 3,
    n_completions: int = 15,
    mode: Literal['one_shot', 'chat', 'planning_and_control', 'planning_and_control_context_carryover'] = 'planning_and_control',
    problems: Dataset = None,
    api_key: str = 'your-api-key',
    results_filename: str = "results.json"
):

    if problems is None:
        problems = load_dataset("ZhiqiGao/TPBench")['public']
    
    problems_df = pd.DataFrame(problems)
    
    print(f"Starting benchmark with {n_samples} problems")
    print(f"Dataset has {len(problems_df)} total problems")
    
    results = {}
    completed_count = 0

    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        outputs = {}
        for i in range(n_samples):
            future = exe.submit(
                benchmark_problem, i, model, mode, api_key, problems_df, n_completions
            )
            outputs[future] = i

        for fut in as_completed(outputs):
            i = outputs[fut]
            problem_id, completions = fut.result()
            results[str(problem_id)] = completions  
            completed_count += 1
            print(f"Completed problem {problem_id} ({completed_count}/{n_samples})")


    print(f"Benchmark completed. {len(results)} problems processed.")
    store_results(results, filename=results_filename)
    
    stats = evaluate_results(n_completions = n_completions ,results_file=results_filename, problems_df=problems_df, api_key=api_key)
    return stats

def benchmark_problem(
    i: int,
    model: str,
    mode: Literal['one_shot', 'chat', 'planning_and_control', 'planning_and_control_context_carryover'],
    api_key: str,
    problems: DataFrame,
    n_completions: int = 1,
    work_dir: str = './outputs'
):
    completions = []
    problem = problems.iloc[i]
    
    prompt = f"""
        You are solving a theoretical physics problem from the TPBench benchmark.

        ALL ANSWERS MUST BE PLACED IN A `results.md` FILE, OTHERWISE YOUR OUTPUT WILL BE CONSIDERED INCORRECT.

        ### Required Procedure:
            1. **Solve the problem step by step**, showing all intermediate work and derivations.
            2. **Place the final boxed answer** within <ANSWER> tags, using proper LaTeX formatting, like this:  
               <ANSWER>\\boxed{{your\\_final\\_answer}}</ANSWER>
            3. The final answer **must be a complete mathematical expression**, including all required values and/or symbols.
            4. **Do NOT** include any explanation, derivation, or commentary inside the <ANSWER> tags — only the final LaTeX result.
            5. **You MUST save your entire output exclusively in a file named `results.md`.**
            6. Even if you do not reach a solution, you MUST still create a valid `results.md` file.
            7. If an error occurs, or you cannot solve the problem, your `results.md` file must explicitly say so inside the <ANSWER> tags, for example:  
               <ANSWER>\\text{{Solution not reached or unable to solve problem}}</ANSWER>
            8. Under **ALL** circumstances, you must create and save a **valid** `results.md` file — no exceptions.
            9. The `results.md` file must contain all the steps and the final answer formatted exactly as required.
        
        ### How to save your result:
            You MUST save your output by writing plain text (Markdown) to a file named `results.md`.  
            Do **NOT** generate any other files (e.g., no `.py` or `.txt` files).  
            Do **NOT** execute the `results.md` file as code — it is a Markdown file and must be treated as such.

        Below is a minimal example of how to write the result file (you must do something equivalent):

        ```python
        with open("results.md", "w") as f:
            f.write("# TPBench Solution\\n\\n")
            f.write("## Step-by-step Solution\\n")
            f.write("... your detailed derivations here ...\\n\\n")
            f.write("## Final Answer\\n")
            f.write("<ANSWER>\\\\boxed{{your\\\\_final\\\\_answer}}</ANSWER>\\n")

        Output Requirements:
        If the problem asks for an expression, return the full expression in LaTeX.

        If it asks for a numeric value, compute and provide that value.

        Follow the strict answer format given:

        {problem['code_answer_requirements']}

        Problem to Solve:
        {problem['problem']}

        Reference Implementation:
        {problem['reference_implementation']}

        REMEMBER:

        Your entire output MUST be enclosed only in the results.md file.
        Missing this file or placing output anywhere else will cause your submission to be marked incorrect.
        Always create the file, even if you fail to solve the problem.
        If you cannot solve, clearly say so inside the <ANSWER> tags in the file.
        Save your output exclusively in a file titled results.md. This file is mandatory.

        ## Final Checklist:
        - [ ] Created `results.md` file
        - [ ] Included step-by-step solution
        - [ ] Final answer in <ANSWER> tags with \\boxed{{}} format
        - [ ] Followed the specified answer requirements
        - [ ] File saved with UTF-8 encoding
        
        REMEMBER: The `results.md` file is the ONLY output that matters. Without it, your solution will be marked as incorrect regardless of correctness.
        
        Execute your solution now and create the required file.
        """


    for j in range(n_completions):
        print(f"Starting problem {i}, completion {j+1}/{n_completions}")
        
        result = run_mas(
            i=i,
            model=model,
            prompt=prompt,
            mode=mode,
            api_key=api_key,
            work_dir=os.path.join(work_dir, f"problem_{i}_completion_{j}")
        )
        
        if result is not None:

            extracted_answer = extract_answer(result, problem_id=i, completion_id=j, work_dir=work_dir)
            completions.append(extracted_answer)
            print(f"Completed problem {i}, completion {j+1}/{n_completions}")
            
            if extracted_answer:
                print(f"Extracted answer: {extracted_answer[:200]}...") 
            else:
                print("No answer extracted")
        else:
            print(f"MAS returned None for problem {i}, completion {j+1}")
            completions.append(None)
                
    return i, completions


def evaluate_results(n_completions: int, results_file: str = "results.json", problems_df: DataFrame = None, api_key: str = 'your-api-key'):

    results_path = os.path.join("results", results_file)
    
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found!")
        return None
    
    with open(results_path, "r") as f:
        results = json.load(f)

    if problems_df is None:
        problems = load_dataset("ZhiqiGao/TPBench")["public"]
        problems_df = pd.DataFrame(problems)

    total_completions = 0
    correct_completions = 0
    total_problems = len(results)
    problems_with_correct_answer = 0

    evaluation_results = {}
    
    print(f"Starting evaluation of {total_problems} problems")

    with ThreadPoolExecutor(max_workers=2) as exe:
        reviews = {}
        
        for idx_str, completions in results.items():
            idx = int(idx_str)

            problem = problems_df.iloc[idx]
            correct_answer = problem['answer']
            
            for j, completion in enumerate(completions):
                future = exe.submit(run_eval_agent, api_key, correct_answer, completion)
                reviews[future] = (idx, j, completion)
        
        print(f"Submitted {len(reviews)} evaluations")
        
        completed_evals = 0

        for review in as_completed(reviews):
            idx, j, completion = reviews[review]
            eval_result = review.result()

            score_match = re.search(r'\b([01])\b', eval_result.strip())

            score = int(score_match.group(1))
            
            if idx not in evaluation_results:
                evaluation_results[idx] = []
            evaluation_results[idx].append(score)
            
            total_completions += 1
            if score == 1:
                correct_completions += 1
            
            completed_evals += 1
            print(f"Problem {idx}, completion {j}: {'✓' if score == 1 else '✗'} ({completed_evals}/{len(reviews)})")
                

    pass_k_totals = {k: 0.0 for k in [1, 5, 10]}
    num_problems = 0

    for idx, scores in evaluation_results.items():
        c = sum(scores)
        if c > 0:
            problems_with_correct_answer += 1

        pass_k_dict = estimate_pass_k(numb_correct=c,numb_completions= n_completions,k_vals=[1, 5, 10])

        for k in pass_k_totals:
            pass_k_totals[k] += pass_k_dict.get(k, 0.0)
        num_problems += 1

    avg_pass_k = {k: (pass_k_totals[k] / num_problems if num_problems > 0 else 0.0) for k in pass_k_totals}

    accuracy = correct_completions / total_completions 

    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"Average Pass@k values: {avg_pass_k}")
    print(f"Total problems evaluated: {total_problems}")
    print(f"Total completions: {total_completions}")
    print(f"Model accuracy is: {accuracy}")
    print(f"Correct completions: {correct_completions}")
    print(f"Problems with at least one correct answer: {problems_with_correct_answer}")
    print("="*50)

    detailed_results = {
        "evaluation_results": evaluation_results,
        "total_completions": total_completions,
        "correct_completions": correct_completions,
        "accuracy": accuracy,
        "total_problems": total_problems,
        "problems_with_correct_answer": problems_with_correct_answer
    }
    
    store_results(detailed_results, filename="evaluation_results.json")
    return detailed_results

def extract_answer(result: dict, problem_id: int, completion_id: int = 0, work_dir: str = './outputs') -> Optional[str]:

    possible_filenames = [
        "results.md",
        "result.md",
        "Results.md",
        "Result.md",
        "RESULTS.md",
        "RESULT.md",
        "TPBench_Solution.md",
        "tpBench_Solution.md",
        "Tpbench_Solution.md",
        "tpbench_solution.md",
        "TPbench_Solution.md",
        "TPBENCH_SOLUTION.md",
        "TPBench_solution.md",
        "tpBench_solution.md",
        "TPBenchSolution.md",
        "tpBenchSolution.md",
        "TpbenchSolution.md",
        "tpbenchsolution.md",
        "TPbenchSolution.md",
        "TPBENCHSOLUTION.md",
        "TPBenchsolution.md",
        "tpBenchsolution.md",
        "TP_Bench_Solution.md",
        "tp_bench_solution.md",
        "Tp_Bench_Solution.md",
        "TP_bench_solution.md",
        ]
    
    base_dir = os.path.join(
        work_dir, 
        f"problem_{problem_id}_completion_{completion_id}", 
        f"problem_{problem_id}", 
        "control"
    )

    for filename in possible_filenames:
        result_file_path = os.path.join(base_dir, filename)
        
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"Successfully read {filename} for problem {problem_id}, completion {completion_id}")
            print(f"File path: {result_file_path}")
            print(f"Content length: {len(content)} characters")
            
            return content

    print(f"No result files found for problem {problem_id}, completion {completion_id}")
    print(f"Searched in directory: {base_dir}")
    print(f"Tried filenames: {', '.join(possible_filenames)}")
    
    return "answer not found"


def estimate_pass_k(numb_correct: int, numb_completions: int, k_vals: list[int]) -> dict:

    result = {}
    for k in k_vals:
        if not (0 <= numb_correct <= numb_completions) or not (1 <= k <= numb_completions):
            result[k] = 0.0
            continue

        denom = math.comb(numb_completions, k)
        numer = math.comb(numb_completions - numb_correct, k)
        result[k] = 1.0 - (numer / denom)

    return result
