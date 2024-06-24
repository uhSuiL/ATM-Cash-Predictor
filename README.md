<h1 align="center" style="font-family: 'Times New Roman',serif">
    Strategic DLinear-T：Decomposition Linear with Time2Vec in Strategy Nets for ATM Cash Balance Prediction
</h1>
<hr />

# Introduction

This is the `Curriculum design` for my `Machine Learning` Course in `DUFE`

- Prepare: `pip install -r requirements.txt`
- Visualization of results: [results_vis.ipynb](./notebook/results_vis.ipynb)
- Train The Model (for 20 times)
  - Strategic DLinear: `python DLinear-_Strategic.py`
  - Strategic DLinear-E: `python DLinear-E_Strategic.py`
  - Strategic DLinear-T: `python DLinear-T_Strategic.py`
  - DLinear-T: `python DLinear_T2V.py`
  - GRU related models are also provided

# Structure

- `data/`: storage for data
  - related utils in `data/__init__.py`
- `model/`
  - all final implements: `model/__init__.py`
  - all related layers: `model/layer.py`
  - related utils: `model/utils.py` (including model loading&saving, visualize...)
- `notebook/`: some examples for training or visualization 

# Epilogue
Though the project achieves the outcome, I'm not satisfied with it because of the defect of the training data
and my lack of ability and cooperation.
I feel full of apologies for my team members and shameful for myself.

Yet another record for my failure.