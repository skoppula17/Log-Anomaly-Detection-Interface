# Sources / References

## Dataset
- LogHub (LogPAI): large collection of system log datasets including HDFS and preprocessed artifacts.
  - https://github.com/logpai/loghub
  - Zenodo record: https://zenodo.org/records/8196385

## Method inspiration (not code)
- DeepLog (next-event prediction + top-k rule for anomaly detection):
  - Du et al., "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning" (CCS 2017)
  - PDF: https://acmccs.github.io/papers/p1285-duA.pdf

## Log parsing background (future work)
- Drain3 (streaming log template miner, if we move from Event_traces -> raw parsing later):
  - https://github.com/logpai/Drain3
  - PyPI: https://pypi.org/project/drain3/0.6/

## Course specs
- ECE 570 project tracks handout (Track 2 requirements, deliverables)
- Project checkpoint 1 instructions (8-slide structure, code snippet limits, result requirement)

## Software Libraries Used
- Streamlit (used for standard, simple data dashboards to fulfill Track 2 product prototyping):
  - https://streamlit.io/

## Extra Notes
- This repo implements a baseline from scratch (PyTorch).
- No external repository code was copied.
