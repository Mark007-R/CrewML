"""CrewML — an autonomous multi-agent ML engineering crew (LangGraph).

The crew takes a raw tabular dataset + a task (classification/regression) and
autonomously produces a trained, evaluated model + report. Its differentiator is
a genuine multi-agent structure with a Critic loop, plus an honest evaluation
against a locked held-out set the crew never sees during modeling.
"""

__version__ = "0.1.0"
