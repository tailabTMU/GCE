# GCE: Confidence Calibration Error for Improved Trustworthiness of Graph Neural Networks

This code is the implementation of Graph Calibration Error (GCE) loss proposed in "GCE: Confidence Calibration Error for Improved Trustworthiness of Graph Neural Networks" submitted to CAIAC2024. The proposed method has been evaluated on two tasks:
- Graph Classification
    - Proteins Dataset
    - Enzymes Dataset
- Node Classification
    - PubMed Dataset
    - Cora Dataset
    - DBLP Dataset

Paper: Hirad Daneshvar and Reza Samavi. "GCE: Confidence Calibration Error for Improved Trustworthiness of Graph Neural Networks." Accepted in The 37<sup>th</sup> Canadian Conference on Artificial Intelligence (Canadian AI 2024).

## Setup
The code has been dockerized (using GPU). The requirements are included in the requirements.txt file. If you choose to use docker, you don't need to install the packages as it will automatically install them all. To use the docker, make sure you create a copy of _.env.example_ file and name it _.env_ and complete it according to your system. To use the dockerized version, you will need a Ubuntu based system.

If you choose to run the code using CPU, you don't need to use docker as the requirements for CPU support is included in a file called _requirements_cpu.txt_.

## Running Experiments
After creating the _.env_ file, you first need to build the image using ```docker compose build```. Then you need to run ```docker compose up -d``` to start the project. To run the experiments, you need to run the following:
- Node Classification on the Homogeneous Datasets:
    - ```docker compose exec torch bash -c "python3.9 node_classification_uncalibrated.py --dataset=Cora"```: Runs the uncalibrated experiments on the Cora dataset.
    - ```docker compose exec torch bash -c "python3.9 node_classification_uncalibrated.py --dataset=PubMed"```: Runs the uncalibrated experiments on the PubMed dataset.
    - ```docker compose exec torch bash -c "python3.9 node_classification_gcl.py --dataset=Cora"```: Runs the GCL experiments on the Cora dataset.
    - ```docker compose exec torch bash -c "python3.9 node_classification_gcl.py --dataset=PubMed"```: Runs the GCL experiments on the PubMed dataset.
    - ```docker compose exec torch bash -c "python3.9 node_classification_brier_contrib.py --dataset=Cora"```: Runs the GCE experiments on the Cora dataset.
    - ```docker compose exec torch bash -c "python3.9 node_classification_brier_contrib.py --dataset=PubMed"```: Runs the GCE experiments on the PubMed dataset.
    - ```docker compose exec torch bash -c "python3.9 node_classification_brier_contrib_dynamic.py --dataset=Cora"```: Runs the GCE<sub>dynamic</sub> experiments on the Cora dataset.
    - ```docker compose exec torch bash -c "python3.9 node_classification_brier_contrib_dynamic.py --dataset=PubMed"```: Runs the GCE<sub>dynamic</sub> experiments on the PubMed dataset.

- Node Classification on the Heterogeneous Dataset:
    - ```docker compose exec torch bash -c "python3.9 heterogeneous_node_classification_uncalibrated.py"```: Runs the uncalibrated experiments.
    - ```docker compose exec torch bash -c "python3.9 heterogeneous_node_classification_gcl.py"```: Runs the GCL experiments.
    - ```docker compose exec torch bash -c "python3.9 heterogeneous_node_classification_brier_contrib.py"```: Runs the GCE experiments.
    - ```docker compose exec torch bash -c "python3.9 heterogeneous_node_classification_brier_contrib_dynamic.py"```: Runs the GCE<sub>dynamic</sub> experiments.

- Graph Classification Tasks:
    - ```docker compose exec torch bash -c "python3.9 graph_classification_uncalibrated.py --dataset=enzymes"```: Runs the uncalibrated experiments on the Enzymes dataset.
    - ```docker compose exec torch bash -c "python3.9 graph_classification_uncalibrated.py --dataset=proteins"```: Runs the uncalibrated experiments on the Proteins dataset.
    - ```docker compose exec torch bash -c "python3.9 graph_classification_gcl.py --dataset=enzymes"```: Runs the GCL experiments on the Enzymes dataset.
    - ```docker compose exec torch bash -c "python3.9 graph_classification_gcl.py --dataset=proteins"```: Runs the GCL experiments on the Proteins dataset.
    - ```docker compose exec torch bash -c "python3.9 graph_classification_brier_contrib.py --dataset=enzymes"```: Runs the GCE experiments on the Enzymes dataset.
    - ```docker compose exec torch bash -c "python3.9 graph_classification_brier_contrib.py --dataset=proteins"```: Runs the GCE experiments on the Proteins dataset.
    - ```docker compose exec torch bash -c "python3.9 graph_classification_brier_contrib_dynamic.py --dataset=enzymes"```: Runs the GCE<sub>dynamic</sub> experiments on the Enzymes dataset.
    - ```docker compose exec torch bash -c "python3.9 graph_classification_brier_contrib_dynamic.py --dataset=proteins"```: Runs the GCE<sub>dynamic</sub> experiments on the Proteins dataset.

    > Note: You don't need to download the datasets. The datasets will be automatically downloaded.

## Generating Results
After running the experiments, the results will be saved in:
- A folder called _saved_info_graph_classification_ in the root directory of the project for graph classification
- A folder called _saved_info_node_classification_ in the root directory of the project for node classification on the homogeneous datasets
- A folder called _saved_info_heterogeneous_node_classification_ in the root directory of the project for graph classification on the heterogeneous dataset

You need to run the following commands to create results files:
- ```docker compose exec torch bash -c "python3.9 node_classification_calibration_results.py"```: Creates a results file for each dataset in the _saved_info_node_classification_ folder.
- ```docker compose exec torch bash -c "python3.9 heterogeneous_node_classification_calibration_results.py"```: Creates a results file in the _saved_info_heterogeneous_node_classification_folder.
- ```docker compose exec torch bash -c "python3.9 graph_classification_calibration_results.py"```: Creates a results file for each dataset in the _saved_info_graph_classification_ folder.

## Cite
If you find the content useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{Daneshvar2024GCE,
	author = {Daneshvar, Hirad and Samavi, Reza},
	journal = {Proceedings of the Canadian Conference on Artificial Intelligence},
	year = {2024},
	month = {may 27},
	note = {https://caiac.pubpub.org/pub/w0amk640},
	publisher = {Canadian Artificial Intelligence Association (CAIAC)},
	title = {GCE: Confidence {Calibration} {Error} for {ImprovedTrustworthiness} of {Graph} {Neural} {Networks}},
}
```
