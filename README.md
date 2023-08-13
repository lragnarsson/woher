# woher
 A useless reverse geocoder powered by AI

 The goal of this project is to generate city names based on a location (lat/lon).
 The model is trained on [geonames.org data](http://download.geonames.org/export/dump/readme.txt) called cities500
 which contain city names and locations for cities with a population > 500.

## Gen 1 - LSTM
Pretty good at generating realistic city names, but coordinates not taken into account properly.

#### Development Environment
Install Anaconda then run:

```conda create -n woher python=3 tensorflow keras pandas matplotlib jupyter```

Activate the environemnt:

```conda activate woher```

#### Run Jupyter Notebook for Model Training
With the conda environment setup and activated run:

```jupyter notebook```

## Gen 2 - Fine Tuned Llama 2
Notebooks need to run with a decent GPU and sizable RAM. E.g. Colab with extra RAM.