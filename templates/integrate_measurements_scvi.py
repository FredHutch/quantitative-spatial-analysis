#!/usr/bin/env python3

"""
Use scvi-tools to integrate measurements across all regions
"""

import scvi
import torch
import anndata as ad
import logging
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/integrate_measurements.txt")
    ]
)
logger = logging.getLogger(__name__)

scvi.settings.seed = 0
logger.info("Last run with scvi-tools version:", scvi.__version__)

torch.set_float32_matmul_precision("high")


def integrate_measurements(adata: ad.AnnData) -> ad.AnnData:
    """
    Use scvi-tools to integrate measurements across all regions
    """
    logger.info("Setting up AnnData object for SCVI")
    scvi.model.SCVI.setup_anndata(adata, batch_key="region")
    logger.info("Creating the SCVI model")
    model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    logger.info("Traiing the SCVI model")
    model.train()
    logger.info("Getting normalized expression from the SCVI model")
    adata.X = model.get_normalized_expression()
    return adata


def main():
    logger.info("Reading spatial data")
    adata = ad.read_h5ad("spatialdata.h5ad")
    logger.info(f"Spatial data has {adata.shape[0]:,} observations and {adata.shape[1]:,} variables")

    logger.info("Integrating measurements")
    integrated_adata = integrate_measurements(adata)
    logger.info(f"Integrated data has {integrated_adata.shape[0]:,} observations and {integrated_adata.shape[1]:,} variables")

    logger.info("Writing integrated data")
    integrated_adata.write("integrated_spatialdata.h5ad")
    logger.info("Done")


if __name__ == "__main__":
    main()
