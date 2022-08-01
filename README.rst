ACC-Identification
==================
.. _start-info:

:versions:      |gh-version| |rel-date| |python-ver|
:sources:       https://github.com/AndresLaverdeMarin/ACC-Identification |codestyle|
:keywords:      ACC, Human driver, Deep Learning, Machine Learning
:short name:    ACC-Identification
:Copyright and License:     © Copyright (c) 2021 European Union.

              Licensed under the EUPL, Version 1.2 or – as soon they will be approved by the European Commission – subsequent versions of the EUPL (the "Licence");
              You may not use this work except in compliance with the Licence.
              You may obtain a copy of the Licence at: |proj-lic|

              Unless required by applicable law or agreed to in writing, software distributed under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS
              OF ANY KIND, either express or implied. See the Licence for the specific language governing permissions and limitations under the Licence.
:Datasets: AstaZero (link: https://doi.org/10.1016/j.trc.2021.103047)

    BJTU dataset (link: https://doi.org/10.1016/j.trb.2015.08.003)


.. _end-info:

.. contents:: Table of Contents
  :backlinks: top

.. _start-introduction:
Introduction
============
This project contains all the code used and described in (to be updated with the link).

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

    ACC-identification
    │   .gitignore
    │   README.rst
    │
    ├───data <- Third party data
    │       AstaZero_data_processed.csv
    │       AstaZero_data_processed_Li_et_al.csv
    │       Bjtu_data_processed.csv
    │       Bjtu_data_processed_Li_et_al.csv
    │
    └───src
        │   predict_biLSTM.py
        │   predict_Li_et_al_LSTM.py
        │   predict_logistic_regression.py
        │   predict_SVM_linear.py
        │
        ├───data <- Contains the scripts to generate data.
        │       Li_et_al_processed_data.py
        │       processed_data.py
        │
        ├───final_models
        │       biLSTM_5s.h5
        │       Li_et_al_LSTM_5s.h5
        │       logistic_regression.pkl
        │       scaler_biLSTM_5s.pkl
        │       scaler_Li_et_al_LSTM_5s.pkl
        │       scaler_logistic_regression.pkl
        │       scaler_SVM_linear.pkl
        │       SVM_linear.pkl
        │
        └───models <- Models architecture.
                biLSTM.py
                Li_et_al_LSTM.py
                logistic_regression.py
                SVM_linear.py

.. _end-structure:

.. _start-sub:

.. |python-ver| image::  https://img.shields.io/badge/Python-3.10-informational
    :alt: Supported Python versions of latest release in PyPi

.. |gh-version| image::  https://img.shields.io/badge/GitHub%20release-1.0.0-orange
    :target: https://github.com/JRCSTU/gearshift/releases
    :alt: Latest version in GitHub

.. |rel-date| image:: https://img.shields.io/badge/rel--date-31--07--2021-orange
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