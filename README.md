# woher

 ~ Conceptual LSTM art ~
 
 The goal of this project is to generate city names based on a location (lat/lon).
 The model is trained on [geonames.org data](http://download.geonames.org/export/dump/readme.txt) called cities500
 which contain city names and locations for cities with a population > 500.
 
 WIP. If everything works out this will become a useless but funny reverse geocoder.

## Installation
How to set up stuff

#### Development Environment
Install Anaconda then run:

```conda create -n woher python=3 tensorflow keras pandas matplotlib jupyter```

Activate the environemnt:

```conda activate woher```

#### Run Jupyter Notebook for Model Training
With the conda environment setup and activated run:

```jupyter notebook```
