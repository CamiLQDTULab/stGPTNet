#!/usr/bin/env python
# coding: utf-8

import sys
import os
import novae
import scanpy as sc
import pandas as pd

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


OUTPUT_DIR = "/workspace/stGPTNet_benchmark/Novae_maynard/data-cc"
MODEL_NAME = "prism-oncology/novae-scConcept-multi-species"
RADIUS = 150


def find_best_resolution(model, adata, label_key="Region"):
    true_n = adata.obs[label_key].nunique()

    for res in [0.1 * i for i in range(1, 21)]:
        col = model.assign_domains(adata, resolution=res)
        n = adata.obs[col].nunique()

        print(f"res={res:.1f} | domains={n} | true={true_n}")

        if n >= true_n:
            return res

    return 1.0


def evaluate(adata, domain_col):
    df = adata.obs[["Region", domain_col]].dropna()

    ari = adjusted_rand_score(
        df["Region"],
        df[domain_col]
    )

    nmi = normalized_mutual_info_score(
        df["Region"],
        df[domain_col]
    )

    return ari, nmi, len(df)


def run_experiment(train_id, test_id, fine_tune=False):

    mode = "Fine-tune" if fine_tune else "Zero-shot"

    print("\n" + "=" * 70)
    print(f"{mode}: {train_id} -> {test_id}")
    print("=" * 70)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    train_path = f"{OUTPUT_DIR}/{train_id}_scConcept.h5ad"
    test_path = f"{OUTPUT_DIR}/{test_id}_scConcept.h5ad"

    adata_train = sc.read_h5ad(train_path)
    adata_test = sc.read_h5ad(test_path)

    print(f"Train: {adata_train.shape}")
    print(f"Test : {adata_test.shape}")

    # --------------------------------------------------
    # Spatial graph
    # --------------------------------------------------
    novae.spatial_neighbors(
        adata_train,
        radius=RADIUS
    )

    novae.spatial_neighbors(
        adata_test,
        radius=RADIUS
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = novae.Novae.from_pretrained(MODEL_NAME)

    # --------------------------------------------------
    # Zero-shot / Fine-tune
    # --------------------------------------------------
    if fine_tune:
        print("\nFine-tuning Novae...")

        model.fine_tune(
            adata_train,
            accelerator="gpu",
            num_workers=4,
            max_epochs =30,
        )

        model.compute_representations(
            adata_train,
            accelerator="gpu",
            num_workers=4
        )

        model.compute_representations(
            adata_test,
            accelerator="gpu",
            num_workers=4
        )

    else:
        print("\nComputing zero-shot representations...")

        model.compute_representations(
            adata_train,
            zero_shot=True,
            accelerator="gpu",
            num_workers=4
        )

        model.compute_representations(
            adata_test,
            zero_shot=True,
            accelerator="gpu",
            num_workers=4
        )

    # --------------------------------------------------
    # Find resolution from TRAIN
    # --------------------------------------------------
    print("\nFinding best resolution...")

    best_resolution = find_best_resolution(
        model,
        adata_train,
        label_key="Region"
    )

    print(f"\nSelected resolution: {best_resolution:.1f}")

    # --------------------------------------------------
    # Assign domains
    # --------------------------------------------------
    train_domain_col = model.assign_domains(
        adata_train,
        resolution=best_resolution
    )

    test_domain_col = model.assign_domains(
        adata_test,
        resolution=best_resolution
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    ari, nmi, n_cells = evaluate(
        adata_test,
        test_domain_col
    )

    print("\nResults")
    print("-" * 40)
    print(f"Resolution : {best_resolution:.1f}")
    print(f"Domain col : {test_domain_col}")
    print(f"Valid cells: {n_cells}")
    print(f"ARI        : {ari:.4f}")
    print(f"NMI        : {nmi:.4f}")

    return {
        "train": train_id,
        "test": test_id,
        "method": mode,
        "resolution": best_resolution,
        "ARI": ari,
        "NMI": nmi,
        "n_cells": n_cells
    }


def main():

    if len(sys.argv) != 3:
        print("Usage:")
        print("python test.py TRAIN_ID TEST_ID")
        print("Example:")
        print("python test.py 151507 151508")
        sys.exit(1)

    train_id = sys.argv[1]
    test_id = sys.argv[2]

    results = []

    # Zero-shot
    results.append(
        run_experiment(
            train_id,
            test_id,
            fine_tune=False
        )
    )

    # Fine-tune
    results.append(
        run_experiment(
            train_id,
            test_id,
            fine_tune=True
        )
    )

    # Save results
    df = pd.DataFrame(results)

    output_file = f"novae_{train_id}_{test_id}_results.csv"
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("Final results")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()