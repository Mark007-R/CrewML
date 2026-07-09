| Dataset | Metric | Dummy (floor) | default RF | Solo agent | AutoML (FLAML) |
|---|---|---|---|---|---|
| credit-g | ROC AUC | 0.5000 | 0.7783 | 0.7521 *(mock)* | 0.7352 |
| diabetes | ROC AUC | 0.5000 | 0.8118 | 0.7987 *(mock)* | 0.8039 |
| vehicle | macro-F1 | 0.1028 | 0.7260 | 0.7763 *(mock)* | 0.7785 |
| cpu_small | R² | -0.0029 | 0.9726 | 0.9747 *(mock)* | 0.9759 |
| kin8nm | R² | -0.0002 | 0.6948 | 0.8120 *(mock)* | 0.8421 |

*(mock)* — solo agent ran without an LLM key; that column is MOCK and not a headline number (EVAL_PROTOCOL.md §5).
