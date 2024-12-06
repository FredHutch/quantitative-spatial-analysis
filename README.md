# Quantitative Spatial Analysis
Analyze collections of spatial data objects

## Background

Spatial data in this case is any dataset in which measurements on a set of observations
have been collected, where each observation also has a location in physical space.

A good example of this is spatial transcriptomics, where each cell (observation) has
been measured for the expression of a collection of genes (features), and where the location
of each cell is given in x/y coordinates.

## Purpose

The goal of this project is to allow users to:

- Select and annotate regions from one or more spatial datasets (e.g. cores from a TMA)
- Build an aggregate collection which consists of multiple regions from one or more source datasets
- Save that collection as a cohesive data package
- Perform clustering on cells (e.g. k-means, leiden, louvain) and annotate each cluster (i.e. cell type)
- Perform neighborhood analysis across the aggregate collection
- Compute summary metrics on the cell type vs. neighborhood vs. region levels
- Visualize the entire spatial data collection in an interactive way

