#!/usr/bin/env python3

from pathlib import Path
import anndata as ad


spatial = ad.concat(
    [
        ad.read_h5ad(fp)
        for fp in Path(".").glob("*.h5ad")
    ],
    uns_merge="same"
)

spatial.write("spatial.h5ad")
