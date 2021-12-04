# FromSurface2Depth

## Overview

This is the complete source code for the master thesis by Roland Stenger, handed in at 22. Mai 2021. It includes the pipeline from simulating training data, preprocessing and training Spatiotemporal LSTM's within several experiments. The project was performed at the Max-Planck-Institute for Dynamics and Self-Organization in the group of Prof. Dr. U. Parlitz and Prof. Dr. S. Luther. The project deals with the reconstruction of excitable media under the surface of a 3-dimensional cube, where the Barkley model is used to generate the data. The neural networks are trained with the surface dynamics of the same cube.

The pipelines are build with kedro, a framework to build robust and scalable data pipelines by providing uniform project templates, data abstraction, configuration and modular data science code. 

## conf

## 


## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
kedro install
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```