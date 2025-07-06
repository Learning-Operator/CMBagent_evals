# CMBAgent Eval

Benchmarking of the **cmbagent** multi-agent system across multiple evaluation datasets.

## Overview

This repository contains benchmarking scripts and results for evaluating cmbagent's performance on various coding and reasoning tasks. The multi-agent system uses specialized agents to tackle complex problems through structured reasoning.

## Benchmarks Included

### Standard Benchmarks
- **HumanEval** - 164 hand-crafted programming problems
- **[Other benchmarks]** - yet to be used

### Custom Benchmarks
- **[Your custom benchmarks]** - Presently being created

## Current Results

### HumanEval Results
**Model:** cmbagent engineer agent with GPT-4o-mini  
**Configuration:** 30 problems × 15 completions (450 total evaluations)

| Metric | Score | Percentage |
|--------|-------|------------|
| pass@1 | 0.9933 | **99.33%** |
| pass@2 | 0.9990 | **99.90%** |
| pass@5 | 1.0000 | **100.00%** |
| pass@10 | 1.0000 | **100.00%** |
