# CMBAgent Evaluation

Benchmarking of the **CMBAgent** multi-agent system across multiple evaluation datasets.

## Overview

This repository contains benchmarking scripts and results for evaluating CMBAgent's performance on various coding and reasoning tasks. The multi-agent system uses specialized agents to tackle complex problems through structured reasoning.

---

## Benchmarks Included

### Standard Benchmarks
- **HumanEval** – 164 hand-crafted programming problems
- **TPBench** – Theoretical physics problem-solving benchmark
- **[Other benchmarks]** – Yet to be used

---

## Current Results

---

## HumanEval Results

**Model:** CMBAgent (one-shot mode, engineer agent with GPT-4o-mini)  
**Configuration:** 30 problems × 15 completions (450 total evaluations)

| Metric   | Score   | Description                         |
|----------|---------|-------------------------------------|
| Pass@1   | 0.9933  | Probability of correct output in 1 completion |
| Pass@2   | 0.9990  | Probability of correct output in 2 completions |
| Pass@5   | 1.0000  | Probability of correct output in 5 completions |
| Pass@10  | 1.0000  | Probability of correct output in 10 completions |

---

## TPBench Public Dataset Results

### CMBAgent Evaluation Report on TPBench Public Dataset

#### Overview

Each problem was attempted with 5 completions (50 total per run).

---

### Performance Summary

| Metric                          | Run 1 (Initial Prompt) | Run 2 (Final Prompt) |
|---------------------------------|-------------------------|-----------------------|
| **Total Problems**              | 10                      | 10                    |
| **Total Completions**           | 50                      | 50                    |
| **Correct Completions**         | 28                      | 23                    |
| **Accuracy**                    | 0.56                    | 0.46                  |
| **Pass@1**                      | 0.56                    | 0.46                  |
| **Pass@5**                      | 0.70                    | 0.80                  |
| **Problems with ≥1 Correct**    | 7 / 10                  | 8 / 10                |

---

### Run 1: TPBench Public Dataset Results (Initial Prompt)

**Model:** CMBAgent with GPT-4o-mini

#### Performance Metrics
- **Accuracy**: 56% (28/50)
- **Pass@1**: 0.56
- **Pass@5**: 0.70

#### Results by Problem

| Problem ID | Correct / Total | Success Rate |
|------------|------------------|---------------|
| 0          | 0 / 5            | 0%            |
| 1          | 4 / 5            | 80%           |
| 2          | 4 / 5            | 80%           |
| 3          | 4 / 5            | 80%           |
| 4          | 0 / 5            | 0%            |
| 5          | 3 / 5            | 60%           |
| 6          | 5 / 5            | 100%          |
| 7          | 3 / 5            | 60%           |
| 8          | 0 / 5            | 0%            |
| 9          | 5 / 5            | 100%          |

#### Performance Metrics
- **Accuracy**: 46% (23/50)
- **Pass@1**: 0.46
- **Pass@5**: 0.80

#### Results by Problem

| Problem ID | Correct / Total | Success Rate |
|------------|------------------|---------------|
| 0          | 1 / 5            | 20%           |
| 1          | 5 / 5            | 100%          |
| 2          | 5 / 5            | 100%          |
| 3          | 2 / 5            | 40%           |
| 4          | 1 / 5            | 20%           |
| 5          | 2 / 5            | 40%           |
| 6          | 3 / 5            | 60%           |
| 7          | 0 / 5            | 0%            |
| 8          | 0 / 5            | 0%            |
| 9          | 4 / 5            | 80%           |

---
