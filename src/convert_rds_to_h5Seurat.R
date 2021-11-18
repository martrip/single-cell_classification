# environment: idorsia_python
# conversion manual: https://mojaveazure.github.io/seurat-disk/articles/convert-anndata.html

# clean workspace
rm(list = ls())

# set working directory
setwd("~/Propulsion/final_project/idorsia")

# install package seurat
install.packages("seurat")

# install package seuratdata
install.packages("devtools")
library(devtools)
devtools::install_github('satijalab/seurat-data')

# install package seuratdisk
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
remotes::install_github("mojaveazure/seurat-disk")

# import libraries
library(Seurat)
library(SeuratData)
library(SeuratDisk)

# load final .rds data
result_final <- readRDS('./data/raw/onco-atlas_v1-final.rds')
result_final

# Converting the final Seurat object to an AnnData file
SaveH5Seurat(result_final, filename = "./data/raw/onco-atlas_v1-final.h5Seurat")
Convert("./data/raw/onco-atlas_v1-final.h5Seurat", dest = "h5ad")

