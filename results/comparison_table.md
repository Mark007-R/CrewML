| Dataset | Metric | Dummy (floor) | default RF | Solo agent | AutoML (FLAML) | **Crew** | Crew − Solo | Crew − AutoML | Crew − default RF |
|---|---|---|---|---|---|---|---|---|---|
| credit-g | ROC AUC | 0.5000 | 0.7783 | 0.6517 | 0.7352 | 0.7913 | +0.1396 | +0.0561 | +0.0130 |
| diabetes | ROC AUC | 0.5000 | 0.8118 | 0.8147 | 0.8039 | 0.8150 | +0.0003 | +0.0111 | +0.0032 |
| vehicle | macro-F1 | 0.1028 | 0.7260 | — | 0.7785 | 0.8326 | — | +0.0541 | +0.1065 |
| cpu_small | R² | -0.0029 | 0.9726 | 0.7129 | 0.9759 | 0.9750 | +0.2621 | -0.0009 | +0.0023 |
| kin8nm | R² | -0.0002 | 0.6948 | — | 0.8421 | 0.8182 | — | -0.0239 | +0.1234 |

All scores are on the LOCKED held-out split, higher is better. The crew's column is a final score taken after the run finished; the crew never saw this split while modeling (EVAL_PROTOCOL.md §3).

* **Crew − Solo**: crew wins 3/3 datasets.
* **Crew − AutoML**: crew wins 3/5 datasets.
* **Crew − default RF**: crew wins 5/5 datasets.
