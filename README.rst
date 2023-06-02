Decoding Driver Type
====================
.. _start-info:

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

.. _start-references:
References
==========
Dataset references::

    @article{makridis_openacc_2021,
        title = {{OpenACC}. {An} open database of car-following experiments to study the properties of commercial {ACC} systems},
        volume = {125},
        issn = {0968-090X},
        url = {https://www.sciencedirect.com/science/article/pii/S0968090X21000772},
        doi = {10.1016/j.trc.2021.103047},
        abstract = {Adaptive Cruise Control (ACC) systems are becoming increasingly available as a standard equipment in modern commercial vehicles. Their penetration rate in the fleet is constantly increasing, as well as their use, especially under freeway conditions. At the same time, limited information is openly available on how these systems actually operate and their differences depending on the vehicle manufacturer or model. This represents an important gap because as the number of ACC vehicles on the road increases, traffic dynamics on freeways may change accordingly, and new collective phenomena, which are only marginally known at present, could emerge. Yet, as ACC systems are introduced as comfort options and their operation is entirely under the responsibility of the driver, vehicle manufacturers do not have explicit requirements to fulfill nor they have to provide any evidence about their performances. As a result, any safety implication connected to their interactions with other road users escapes any monitoring and opportunity of improvement. This work presents a set of experimental car-following campaigns, providing an overview of the behavior of commercial ACC systems under different driving conditions. Furthermore, the suggestion of a unified data structure across the different tests facilitates comparison between the different campaigns, vehicles, systems and specifications. The complete data is published as an open-access database (OpenACC), available to the research community. As more test campaigns will be carried out, OpenACC will evolve accordingly. The activity is performed in the framework of the openData policy of the European Commission Joint Research Centre with the objective to engage the whole scientific community towards a better understanding of the properties of ACC vehicles in view of anticipating their possible impacts on traffic flow and prevent possible problems connected to their widespread introduction. In this light, OpenACC, over time, also aims at becoming a reference point to study if and how the parameters of such systems need to be regulated, how homogeneously they behave, how new ACC car-following models should be designed for traffic microsimulation purposes and what are the key differences between ACC systems and human drivers.},
        language = {en},
        urldate = {2021-07-31},
        journal = {Transportation Research Part C: Emerging Technologies},
        author = {Makridis, Michail and Mattas, Konstantinos and Anesiadou, Aikaterini and Ciuffo, Biagio},
        month = apr,
        year = {2021},
        keywords = {Adaptive cruise control, Car-following, Driver behavior, Empirical observations, Microsimulation, Open data, Traffic flow, Vehicle dynamics},
        pages = {103047},
        file = {ScienceDirect Full Text PDF:C\:\\Users\\mmakridis\\Zotero\\storage\\7XS3GEXB\\Makridis et al. - 2021 - OpenACC. An open database of car-following experim.pdf:application/pdf;ScienceDirect Snapshot:C\:\\Users\\mmakridis\\Zotero\\storage\\9GEF4ZBQ\\S0968090X21000772.html:text/html},
    }

    @article{jiang_experimental_2015,
	title = {On some experimental features of car-following behavior and how to model them},
	volume = {80},
	issn = {0191-2615},
	url = {http://www.sciencedirect.com/science/article/pii/S0191261515001782},
	doi = {10.1016/j.trb.2015.08.003},
	abstract = {We have carried out car-following experiments with a 25-car-platoon on an open road section to study the relation between a car’s speed and its spacing under various traffic conditions, in the hope to resolve a controversy surrounding this fundamental relation of vehicular traffic. In this paper we extend our previous analysis of these experiments, and report new experimental findings. In particular, we reveal that the platoon length (hence the average spacing within a platoon) might be significantly different even if the average velocity of the platoon is essentially the same. The findings further demonstrate that the traffic states span a 2D region in the speed-spacing (or density) plane. The common practice of using a single speed-spacing curve to model vehicular traffic ignores the variability and imprecision of human driving and is therefore inadequate. We have proposed a car-following model based on a mechanism that in certain ranges of speed and spacing, drivers are insensitive to the changes in spacing when the velocity differences between cars are small. It was shown that the model can reproduce the experimental results well.},
	urldate = {2019-05-13},
	journal = {Transportation Research Part B: Methodological},
	author = {Jiang, Rui and Hu, Mao-Bin and Zhang, H. M. and Gao, Zi-You and Jia, Bin and Wu, Qing-Song},
	month = oct,
	year = {2015},
	keywords = {Car-following, Experiment, Model, Traffic flow},
	pages = {338--354},
	file = {ScienceDirect Full Text PDF:C\:\\Users\\mmakridis\\Zotero\\storage\\A7V3EDGW\\Jiang et al. - 2015 - On some experimental features of car-following beh.pdf:application/pdf;ScienceDirect Snapshot:C\:\\Users\\mmakridis\\Zotero\\storage\\3C9TD42B\\S0191261515001782.html:text/html},
    }
.. _end-references:

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
