# FromSurface2Depth

## Overview

This is the complete source code for the master thesis by Roland Stenger, handed in at 22. Mai 2021. It includes the pipeline from simulating training data, preprocessing and training Spatiotemporal LSTM's within several experiments. The project was performed at the Max-Planck-Institute for Dynamics and Self-Organization in the group of Prof. Dr. U. Parlitz and Prof. Dr. S. Luther. The project deals with the reconstruction of excitable media under the surface of a 3-dimensional cube, where the Barkley model is used to generate the data. The neural networks are trained with the surface dynamics of the same cube.

The pipelines are build with kedro, a framework to build robust and scalable data pipelines by providing uniform project templates, data abstraction, configuration and modular data science code. 

## conf
 This folder contains all specifications for data files and specific simulation- and training-hyperparameters. To use the specifications, kedro uses the flag --env "folder_name"; for example: kedro run --env exp4

## data

In this folder, all the data which will be generated during the pipeline run, is stored. The folder location are defined by the  environment (conf-folder). 

## docs


## documents

## logs

## notebook 

## scripts

## src




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