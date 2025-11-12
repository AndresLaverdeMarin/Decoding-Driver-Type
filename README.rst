Decoding Driver Type
====================

Video Abstract
==============

.. image:: video/Video_abstract.gif
   :alt: Video Abstract - Decoding Driver Type
   :align: center
   :width: 100%

|

For higher quality, `click here to download the full video (MP4) <https://github.com/AndresLaverdeMarin/Decoding-Driver-Type/raw/main/video/Video_abstract.mp4>`_

.. _start-info:

Information Table
=================

:versions:      |gh-version| |rel-date| |python-ver|
:sources:       https://github.com/AndresLaverdeMarin/Decoding-Driver-Type |codestyle|
:keywords:      Traffic engineering, Automated Vehicles, Driver identification, Driver behavior, Deep learning
:short name:    Decoding Driver Type: Unveiling Autonomous Vehicles using Trajectory Data
:Copyright and License:     © Copyright (c) 2021 European Union.

              Licensed under the EUPL, Version 1.2 or – as soon they will be approved by the European Commission – subsequent versions of the EUPL (the "Licence");
              You may not use this work except in compliance with the Licence.
              You may obtain a copy of the Licence at: |proj-lic|

              Unless required by applicable law or agreed to in writing, software distributed under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS
              OF ANY KIND, either express or implied. See the Licence for the specific language governing permissions and limitations under the Licence.
:Datasets: AstaZero (see `References`_ section)

    BJTU dataset (see `References`_ section)


.. _end-info:

.. contents:: Table of Contents
  :backlinks: top

.. _start-introduction:
Introduction
============
This project contains all the code used and described in Human or Machine: A novel deep learning framework for autonomous driver identification based on vehicle trajectories.

.. _end-introduction:

.. _start-install:
Installation
============
Prerequisites
-------------
**Python-3.10+** is required.
It requires **numpy/scipy** and **pandas** libraries with native backends.

.. Tip::
    On *Windows*, it is preferable to use the `Anaconda <https://www.anaconda.com/products/individual>`__ distribution.
    To avoid possible incompatibilities with other projects

To run the deep learning models using GPU `cuDNN 8.1 <https://developer.nvidia.com/cudnn>`__ nvidia API is required.

.. Tip::
    If you use a different tensorflow-gpu version check `TensorFlow 2  <https://www.tensorflow.org/install/source_windows#gpu>`__
    to install correct versions of the compilers and drivers.

Download
--------
Download the sources,

- either with *git*, by giving this command to the terminal::

      git clone https://github.com/AndresLaverdeMarin/ACC-Identification.git --depth=1
.. _end-install:

.. _start-structure:

Project files and folders
=========================
The files and folders of the project are listed below::

    Decoding-Driver-Type
    │
    │   README.rst                  # Project documentation and usage guide
    │   requirements.txt            # Python dependencies and package versions
    │   main.py                     # Main execution script for model training and evaluation
    │   .gitignore                  # Git ignore patterns
    │
    ├───data                        # Processed datasets and analysis results
    │   │   AstaZero_data_processed.csv              # AstaZero dataset with proposed features
    │   │   AstaZero_data_processed_Li_et_al.csv     # AstaZero dataset with Li et al. features
    │   │   Bjtu_data_processed.csv                  # BJTU dataset with proposed features
    │   │   Bjtu_data_processed_Li_et_al.csv         # BJTU dataset with Li et al. features
    │   │
    │   └───ACC_data_analysis_results                # Statistical analysis outputs
    │           MMD_test_Scenario_A.png              # Maximum Mean Discrepancy test visualization
    │           MMD_test_Scenario_B.png              # Maximum Mean Discrepancy test visualization
    │           permutation_test_logs_scenario_A.pkl # Permutation test results for concept drift
    │           permutation_test_logs_scenario_B.pkl # Permutation test results for concept drift
    │
    ├───src                         # Source code for models and data processing
    │   │   predict_biLSTM.py                # Prediction script using biLSTM model
    │   │   predict_Li_et_al_LSTM.py         # Prediction script using Li et al. LSTM model
    │   │   predict_logistic_regression.py   # Prediction script using logistic regression
    │   │   predict_SVM_linear.py            # Prediction script using linear SVM
    │   │
    │   ├───data                    # Data preprocessing and feature extraction scripts
    │   │       Li_et_al_processed_data.py   # Data processing following Li et al. methodology
    │   │       processed_data.py            # Data processing for proposed feature set
    │   │
    │   ├───final_models            # Pre-trained models and fitted scalers
    │   │       biLSTM_5s.h5                       # Trained bidirectional LSTM model (5-second window)
    │   │       biLSTM.hdf5                        # Alternative biLSTM checkpoint
    │   │       Li_et_al_LSTM_5s.h5                # Trained Li et al. LSTM model (5-second window)
    │   │       logistic_regression.pkl            # Trained logistic regression classifier
    │   │       SVM_linear.pkl                     # Trained linear SVM classifier
    │   │       scaler_biLSTM_5s.pkl               # Feature scaler for biLSTM model
    │   │       scaler_Li_et_al_LSTM_5s.pkl        # Feature scaler for Li et al. LSTM
    │   │       scaler_logistic_regression.pkl     # Feature scaler for logistic regression
    │   │       scaler_SVM_linear.pkl              # Feature scaler for linear SVM
    │   │
    │   ├───models                  # Model architecture definitions
    │   │       biLSTM.py            # Bidirectional LSTM architecture implementation
    │   │       Li_et_al_LSTM.py     # Li et al. LSTM architecture implementation
    │   │       logistic_regression.py # Logistic regression with bagging ensemble
    │   │       SVM_linear.py        # Linear SVM with bagging ensemble
    │   │
    │   └───statistical_analysis    # Statistical testing and concept drift analysis
    │           statistical_test.ipynb         # ANOVA and Tukey HSD tests for model comparison
    │           concept_drift_analysis.ipynb   # Concept drift detection using MMD
    │
    ├───plots                       # Visualization and plotting utilities
    │       plots.ipynb             # Jupyter notebook for generating publication-ready figures
    │
    └───video                       # Animation and visualization for presentations
        │   scene_1.py              # Manim script for Hilbert-Huang Transform visualization
        │   Video_abstract.mp4      # Video abstract of the research
        │
        └───data                    # Data files for video generation
                data2plotACC.csv    # Preprocessed ACC data for visualization
                data2plotHV.csv     # Preprocessed human vehicle data for visualization

.. _end-structure:

.. _start-usage:

Usage Example
=============
This section demonstrates how to use the biLSTM model for driver type prediction based on the ``src/predict_biLSTM.py`` file.

Basic Usage
-----------
To predict driver types using the pre-trained biLSTM model:

.. code-block:: python

    from pathlib import Path
    from os.path import abspath, join, dirname
    from tensorflow import keras
    from data.processed_data import read

    # Get the project path
    prj_dir = Path(abspath(join(dirname(__file__), "..")))

    # Load and preprocess data
    X, y = read(
        prj_dir.joinpath("data").joinpath("AstaZero_data_processed.csv"),
        "./final_models/scaler_biLSTM_5s.pkl",
        5,
    )

    # Load the pre-trained biLSTM model
    model = keras.models.load_model(
        prj_dir.joinpath("src").joinpath("final_models").joinpath("biLSTM_5s.h5")
    )

    # Evaluate the model
    model.evaluate(X, y, verbose=1)

    # Make predictions
    predictions = model.predict(X)

The ``read`` function loads trajectory data and applies the saved scaler transformation. The model uses a 5-second window (indicated by the ``5`` parameter) for sequence-based predictions to distinguish between human and autonomous drivers.

.. _end-usage:

.. _start-cite:

How to Cite
===========

If you use this code or methodology in your research, please cite our paper::

    Article Title: Human or Machine: A novel deep learning framework for autonomous
                    driver identification based on vehicle trajectories

    Journal: IEEE Transactions on Intelligent Transportation Systems (TITS)

    DOI: 10.1109/TITS.2025.3628491

    Manuscript Number: T-ITS-24-09-3652

BibTeX entry::

    @article{laverde2025human,
        title = {Human or Machine: A novel deep learning framework for autonomous driver identification based on vehicle trajectories},
        journal = {IEEE Transactions on Intelligent Transportation Systems},
        year = {2025},
        doi = {10.1109/TITS.2025.3628491},
        note = {Manuscript Number: T-ITS-24-09-3652}
    }

.. _end-cite:

.. _start-references:

References
==========

This project uses datasets from the following publications:

AstaZero Dataset (OpenACC)
--------------------------

**Citation Information**::

    Article Title: OpenACC. An open database of car-following experiments to study
                    the properties of commercial ACC systems

    Authors: Makridis, Michail and Mattas, Konstantinos and Anesiadou, Aikaterini
             and Ciuffo, Biagio

    Journal: Transportation Research Part C: Emerging Technologies

    Volume: 125

    DOI: 10.1016/j.trc.2021.103047

    Year: 2021

    URL: https://www.sciencedirect.com/science/article/pii/S0968090X21000772

**BibTeX entry**::

    @article{makridis_openacc_2021,
        title = {{OpenACC}. {An} open database of car-following experiments to study the properties of commercial {ACC} systems},
        volume = {125},
        issn = {0968-090X},
        url = {https://www.sciencedirect.com/science/article/pii/S0968090X21000772},
        doi = {10.1016/j.trc.2021.103047},
        journal = {Transportation Research Part C: Emerging Technologies},
        author = {Makridis, Michail and Mattas, Konstantinos and Anesiadou, Aikaterini and Ciuffo, Biagio},
        month = {April},
        year = {2021},
        pages = {103047}
    }

BJTU Dataset
------------

**Citation Information**::

    Article Title: On some experimental features of car-following behavior and how
                    to model them

    Authors: Jiang, Rui and Hu, Mao-Bin and Zhang, H. M. and Gao, Zi-You and Jia, Bin
             and Wu, Qing-Song

    Journal: Transportation Research Part B: Methodological

    Volume: 80

    DOI: 10.1016/j.trb.2015.08.003

    Year: 2015

    URL: http://www.sciencedirect.com/science/article/pii/S0191261515001782

**BibTeX entry**::

    @article{jiang_experimental_2015,
        title = {On some experimental features of car-following behavior and how to model them},
        volume = {80},
        issn = {0191-2615},
        url = {http://www.sciencedirect.com/science/article/pii/S0191261515001782},
        doi = {10.1016/j.trb.2015.08.003},
        journal = {Transportation Research Part B: Methodological},
        author = {Jiang, Rui and Hu, Mao-Bin and Zhang, H. M. and Gao, Zi-You and Jia, Bin and Wu, Qing-Song},
        month = {October},
        year = {2015},
        pages = {338--354}
    }

.. _end-references:

.. _start-sub:

.. |python-ver| image::  https://img.shields.io/badge/Python-3.10-informational
    :alt: Supported Python versions of latest release in PyPi

.. |gh-version| image::  https://img.shields.io/badge/GitHub%20release-1.0.0-orange
    :target: https://github.com/JRCSTU/gearshift/releases
    :alt: Latest version in GitHub

.. |rel-date| image:: https://img.shields.io/badge/rel--date-12--11--2025-orange
    :target: https://github.com/JRCSTU/gearshift/releases
    :alt: release date

.. |br| image:: https://img.shields.io/badge/docs-working%20on%20that-red
    :alt: GitHub page documentation

.. |doc| image:: https://img.shields.io/badge/docs-passing-success
    :alt: GitHub page documentation

.. |proj-lic| image:: https://img.shields.io/badge/license-European%20Union%20Public%20Licence%201.2-lightgrey
    :target:  https://joinup.ec.europa.eu/software/page/eupl
    :alt: EUPL 1.2

.. |codestyle| image:: https://img.shields.io/badge/code%20style-black-black.svg
    :target: https://github.com/ambv/black
    :alt: Code Style

.. |pypi-ins| image:: https://img.shields.io/badge/pypi-v1.1.3-informational
    :target: https://pypi.org/project/wltp-gearshift/
    :alt: pip installation

.. |binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/git/https%3A%2F%2Fcode.europa.eu%2Fjrc-ldv%2Fjrshift.git/main?labpath=Notebooks%2FGUI_binder_interface.ipynb
    :alt: JupyterLab for Gerashift Calculation Tool (stable)

.. |CO2| replace:: CO\ :sub:`2`
.. _end-sub:
