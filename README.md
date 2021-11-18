single-cell_classification
==============================

## 1.	Background and Task 

When there are tumour cells, immune cells infiltrate into the tumour to kill tumour cells. In order to investigate drug performances to cure cancers, it is necessary to develop a tool to classify immune cells inside tumours before and after the patients get their treatments. This tool can compare which cells still exist inside the tumours and which ones have disappeared due to the treatment, Figure 1. 
To develop a tool to classify cells inside the tumours by using the single-cell reference Atlas to create a cell type classifier. This tool can then automatically assign cell labels to new single-cell studies. 

 
<img width="766" alt="CellClassification" src="https://user-images.githubusercontent.com/89971977/142504350-07cd9bf1-f2e5-4f0f-8e77-94c68c91a4d2.PNG">
 
<p align="center">
**Figure 1** Cell Classification
</p>

All icons used in this report are from [flaticon](https://www.flaticon.com/) 

## 2.	Features Descriptions 
•	3,000 genes 

•	43 cell types (108k cells). 43 cell types are the maximum number of cells. 

## 3.	Methodology 

**Part 1: Data Conversion** <br>   
RDS data (Adata format) is transformed to h5ad data format so that it can be transformed to csv file by Python.

**Part 2: Data Exploration** <br>
Structures of dataset were investigated using 

**Part 3: Data Reduction** <br>
Original dataset is reduced to implementable dataset size, Figure 2. 
Original dataset is reduced to 30% in a random fashion, is denoted as “Test Dataset”. It can be observed that the dataset is unbalanced. This means the amount of cell types are unbalanced in the original dataset. This dataset is used in the unsupervised learning model to generate clusters. This same dataset is used in the Supervised learning model as a Test Dataset (unseen dataset). 

<img width="599" alt="DataFlow" src="https://user-images.githubusercontent.com/89971977/142503742-c2ead5cf-989b-44c3-8ff1-726340fdc765.PNG">
 
**Figure 2** Data Reduction

Two datasets were generated, and they both are denoted as a “Train Dataset”, and these will be used as a training dataset for the supervised models. Ten percent of the original dataset is randomly selected, and this is unbalanced data. This “Train Dataset” is split into 80% and 20%, train and test dataset to train the supervised model, respectively.

In a similar fashion, another “Train Dataset” is un-randomly selected from the original dataset and it is 10% of the original amount of data. The un-random dataset was selected based on the first order of data in the given file. 

**Part 4: Unsupervised Machine Learning Model** <br> 
Hierarchical unsupervised machine learning model is developed to classify cells based on cell types (“cell_type” in the dataset). The dataset dimension is reduced by using the Principal Component Analysis (PCA) to perform linear dimensional reduction then the Uniform Manifold Approximation and Projection (UMAP) is used to do a 2D plot for a scatter visualization. This method is widely performed in gene expression problems (Seurat - Guided Clustering Tutorial (satijalab.org)). Then the “Kmeans” clustering, together with “Predicted Classes” from the supervised model, is used to label cell types to this plot. 
The unsupervised model file name: “HierarchicalUnsupervised30Subset_20211028.ipynb” 

**Part 5: Supervised Machine Learning Model** <br> 
Two sets of cell numbers were explored, 43 and 12 cells, respectively. These are denoted as “12 Class” and “43 Classes” throughout the project. Each number of cells were explored using balanced and unbalanced dataset as described in Figure 2. 
Additional supervised model was developed based on given dataset that its dimension has been reduced using a PCA method. 


**Part 6: Model Interpretation** <br> 
Gene expressions for each cell were generated and is known as “Feature Importance”. This feature importance was assigned to 12 Classes (12 cell types). The 12 Classes were predicted based on unbalanced dataset from a supervised learning model. 

## 4.	Results <br>
For 12 classes, 96% prediction accuracy of cells were predicted and the prediction under a Macro F1 method shows 74%. 

## 5.	Outlook <br>
The model can be simplified by reduction of features. Only the most significant genes, based on the “Feature Importance”, output described earlier can be pre-selected and remain in the model. Those genes with less significant features can be excluded from the model. 



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
