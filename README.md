# CMBAgent Evaluation

Benchmarking of the **cmbagent** multi-agent system across multiple evaluation datasets.

## Overview

This repository contains benchmarking scripts and results for evaluating cmbagent's performance on various coding and reasoning tasks. The multi-agent system uses specialized agents to tackle complex problems through structured reasoning.

## Benchmarks Included

### Standard Benchmarks
- **HumanEval** - 164 hand-crafted programming problems
- **TPBench** - Theoretical physics problem solving benchmark
- **[Other benchmarks]** - Yet to be used

---

## Current Results

## HumanEval Results

**Model:** cmbagent one-shot mode, engineer agent with GPT-4o-mini  
**Configuration:** 30 problems × 15 completions (450 total evaluations)

| Metric | Score | Percentage |
|--------|-------|------------|
| Pass@1 | 0.9933 | **99.33%** |
| Pass@2 | 0.9990 | **99.90%** |
| Pass@5 | 1.0000 | **100.00%** |
| Pass@10 | 1.0000 | **100.00%** |

---

## TPBench Results

**Model:** cmbagent with GPT-4o-mini  
**Dataset:** TPBench (Public) - Theoretical physics problems  
**Configuration:** 10 problems × 5 completions (50 total evaluations)

### Performance Metrics
| Metric | Score | Description |
|--------|-------|-------------|
| Pass@1 | 0.56 | Probability of correct output in 1 completion |
| Pass@5 | 0.70 | Probability of correct output in 5 completions |
| Overall Accuracy | 56% | 28/50 correct completions |
| Problem Coverage | 70% | 7/10 problems solved |

### Results by Problem
| Problem | Correct/Total | Success Rate |
|---------|---------------|--------------|
| 0 | 0/5 | 0% |
| 1 | 4/5 | 80% |
| 2 | 4/5 | 80% |
| 3 | 4/5 | 80% |
| 4 | 0/5 | 0% |
| 5 | 3/5 | 60% |
| 6 | 5/5 | 100% |
| 7 | 3/5 | 60% |
| 8 | 0/5 | 0% |
| 9 | 5/5 | 100% |

---
