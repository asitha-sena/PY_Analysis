# PY_Analysis
This module has finite-difference and finite-element implementations of the p-y method to analyze laterally loaded pile foundations. It was originally written in Python 2.7 but this branch has been re-written for Python 3.7+.

The objective of writing this software was to carry out p-y analyses of laterally loaded pile foundations for my PhD dissertation. There are several commercially available software to carry this type of analyses but, they are restricted by the p-y models (non-linear soil springs) that are included in their databases. As a researcher, I needed a way to implement and evaluate the performance of some of the newer p-y models, that have been published in the literature, under various conditions and this software provides an easy framework to achieve this goal.

The current version of this module has several limitations which are detailed within the individual functions. The static analysis component has been fully implemented (in the elastic range of the pile) and verified for a variety of piles, soil types, and p-y models. A dynamic analysis module which can calculate the natural frequency of wind turbine structures that are founded on monopiles (based on solving the associated eigen value problem) is currently being developed and will be added soon.

A Jupyter notebook with several examples are provided as a tutorial to get started.
