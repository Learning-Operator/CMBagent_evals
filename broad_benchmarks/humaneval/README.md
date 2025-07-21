# Benchmark Results

**Model being benchmarked:** cmbagent engineer agent with GPT-4o-mini

**Test Configuration:**
- num_questions: 30
- completions_per: 15  
- Total_completions: 450

**Note:** Results may not be the most accurate due to only evaluating cmbagent through 30 questions, or 18% of the HumanEval dataset. More comprehensive testing may be required for definitive results.

## Pass@k Metrics

| Metric | Score | Percentage |
|--------|-------|------------|
| **pass@1** | 0.9933 | **99.33%** |
| **pass@2** | 0.9990 | **99.90%** |
| **pass@3** | 0.9999 | **99.99%** |
| **pass@5** | 1.0000 | **100.00%** |
| **pass@10** | 1.0000 | **100.00%** |
| **pass@13** | 1.0000 | **100.00%** |
| **pass@15** | 1.0000 | **100.00%** |

## Summary

These are strong results! The model achieves near-perfect performance:

- **99.33%** success rate with just 1 attempt
- **99.90%** success rate with 2 attempts  
- **Perfect 100%** success rate with 5 or more attempts
