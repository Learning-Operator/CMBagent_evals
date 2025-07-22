# CMBAgent Evaluation Report on TPBench public dataset

## Overview

Each problem was attempted with 5 completions (50 total per run). The goal was to measure the model’s reasoning accuracy and coverage under different prompt conditions.



## Performance Summary

| Metric                          |          Run 1          |         Run 2         |
|---------------------------------|-------------------------|-----------------------|
| **Total Problems**              | 10                      | 10                    |
| **Total Completions**           | 50                      | 50                    |
| **Correct Completions**         | 28                      | 23                    |
| **Accuracy**                    | 0.56                    | 0.46                  |
| **Pass@1**                      | 0.56                    | 0.46                  |
| **Pass@5**                      | 0.70                    | 0.80                  |
| **Problems with ≥1 Correct**    | 7 / 10                  | 8 / 10                |



## Run 1: TPBench Public Dataset Results (Initial Prompt)

### Performance Metrics
- **Accuracy**: 56% (28/50)
- **Pass@1**: 0.56
- **Pass@5**: 0.70

### Results by Problem

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



## Run 2: Evaluation with Finalized Prompt

### Performance Metrics
- **Accuracy**: 46% (23/50)
- **Pass@1**: 0.46
- **Pass@5**: 0.80

### Results by Problem

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
