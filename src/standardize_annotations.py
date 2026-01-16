"""Translate InChIKey to JCP2022 ID using JUMP metadata database."""

import duckdb
import pandas as pd


def load_inchikey_to_jumpid_mapping(db_path: str = "/work/datasets/jump_core/annotations/jump_metadata.duckdb") -> pd.DataFrame:
    """
    Load the InChIKey to JCP2022 ID mapping from the compound table.

    Args:
        db_path: Path to the JUMP metadata DuckDB database

    Returns:
        DataFrame with columns: Metadata_InChIKey, Metadata_JCP2022
    """
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("""
        SELECT
            Metadata_InChIKey,
            Metadata_JCP2022
        FROM compound
        WHERE Metadata_InChIKey IS NOT NULL
    """).df()
    con.close()
    return df.dropna()


def translate_inchikey_to_jumpid(
    inchikeys: list[str],
    db_path: str = "/work/datasets/jump_core/annotations/jump_metadata.duckdb",
    match_connectivity_only: bool = True
) -> pd.DataFrame:
    """
    Translate InChIKeys to JCP2022 IDs.

    Args:
        inchikeys: List of InChIKeys to translate
        db_path: Path to the JUMP metadata DuckDB database
        match_connectivity_only: If True (default), match only on the connectivity layer
                                (first part before hyphen). If False, match on full InChIKey.

    Returns:
        DataFrame with columns: Metadata_InChIKey, InChIKey_Connectivity, Metadata_JCP2022
        If match_connectivity_only=True, deduplicates based on connectivity layer before mapping
    """
    mapping = load_inchikey_to_jumpid_mapping(db_path)
    input_df = pd.DataFrame({"Metadata_InChIKey": inchikeys})

    # Always extract connectivity layer
    input_df["InChIKey_Connectivity"] = input_df["Metadata_InChIKey"].str.split("-").str[0]

    if match_connectivity_only:
        # Remove duplicates based on connectivity layer before merging
        input_df = input_df.drop_duplicates(subset="InChIKey_Connectivity")

        # Extract connectivity layer from mapping
        mapping["InChIKey_Connectivity"] = mapping["Metadata_InChIKey"].str.split("-").str[0]

        # Keep all unique JCP2022 IDs per connectivity (one connectivity may map to multiple JCP IDs)
        mapping = mapping[["InChIKey_Connectivity", "Metadata_JCP2022"]].drop_duplicates()

        # Merge on connectivity layer - will create multiple rows if multiple JCP2022 IDs exist
        result = input_df.merge(
            mapping,
            on="InChIKey_Connectivity",
            how="left"
        )
        result = result[["Metadata_InChIKey", "InChIKey_Connectivity", "Metadata_JCP2022"]]
    else:
        # Merge on full InChIKey
        result = input_df.merge(mapping, on="Metadata_InChIKey", how="left")
        result = result[["Metadata_InChIKey", "InChIKey_Connectivity", "Metadata_JCP2022"]]

    return result


if __name__ == "__main__":
    # Example usage
    example_inchikeys = [
        "AAAHWCWPZPSPIW-UHFFFAOYSA-N",
        "AAAJHRMBUHXWLD-UHFFFAOYSA-N",
        "AAANUZMCJQUYNX-UHFFFAOYSA-N"
    ]

    result = translate_inchikey_to_jumpid(example_inchikeys)
    print(result)
    
    get_compound_compound_mapping = True
    get_compound_gene_mapping     = True
    
    ############################################################################
    
    if get_compound_compound_mapping:
        # Get all unique InChIKeys in MOTIVE compound-compound database
        df = pd.read_parquet("/work/datasets/jump_core/annotations/annotations_compound_compound.parquet")
        
        # Get unique InChIKeys from both columns (Interactions between compound A and B)
        unique_inchikeys_a = df["inchikey_a"].dropna().unique()
        unique_inchikeys_b = df["inchikey_b"].dropna().unique()
        
        # Find all unique InChIKeys
        unique_inchikeys = list(set(unique_inchikeys_a) | set(unique_inchikeys_b))
        print(f"Total unique InChIKeys in MOTIVE compound-compound database: {len(unique_inchikeys)}")

        # Translate all unique InChIKeys to JCP2022 IDs
        result = translate_inchikey_to_jumpid(unique_inchikeys)
        print(f"Total InChIKeys mapped to JCP2022 IDs: {result['Metadata_JCP2022'].notnull().sum()}")
        
        # Filter out all without mapping
        mapping_mask = result["Metadata_JCP2022"].notnull()
        result_cc = result[mapping_mask]
        
        # Save mapping to CSV
        result_cc.to_csv("/work/datasets/jump_core/annotations/inchikey_to_jcp2022_mapping_compound_compound.csv", index=False)
        
    
    ############################################################################
    
    if get_compound_gene_mapping:
        # Get all unique InChIKeys in MOTIVE compound-gene parquet
        df = pd.read_parquet("/work/datasets/jump_core/annotations/annotations_compound_gene.parquet")
        
        unique_inchikeys = df["inchikey"].dropna().unique()
        
        print(f"Total unique InChIKeys in MOTIVE compound-gene database: {len(unique_inchikeys)}")
        
        # Translate all unique InChIKeys to JCP2022 IDs
        result = translate_inchikey_to_jumpid(unique_inchikeys)
        print(f"Total InChIKeys mapped to JCP2022 IDs: {result['Metadata_JCP2022'].notnull().sum()}")
        
        # Filter out all without mapping
        mapping_mask = result["Metadata_JCP2022"].notnull()
        result_cg = result[mapping_mask]
        
        # Save mapping to CSV
        result_cg.to_csv("/work/datasets/jump_core/annotations/inchikey_to_jcp2022_mapping_compound_gene.csv", index=False)
        
        
    ############################################################################
    ############################################################################
    
    # Combined the two mappings 
    if get_compound_compound_mapping and get_compound_gene_mapping:
        combined_mapping = pd.concat([result_cc, result_cg]).drop_duplicates().reset_index(drop=True)
        print(f"Total unique InChIKeys mapped to JCP2022 IDs (combined): {len(combined_mapping)}")
        
        # Dropping inchikeys that have the same connectivity layer
        combined_mapping = combined_mapping.drop_duplicates(subset="Metadata_JCP2022").reset_index(drop=True)
        print(f"Total unique Metadata_JCP2022: {len(combined_mapping)}")
        
        
        # Dropping inchikeys that have the same connectivity layer
        # combined_mapping = combined_mapping.drop_duplicates(subset="InChIKey_Connectivity").reset_index(drop=True)
        # print(f"Total unique InChIKeys mapped to JCP2022 IDs (combined after dropping connectivity duplicates): {len(combined_mapping)}")
        
        # Save combined mapping to CSV
        combined_mapping.to_csv("/work/datasets/jump_core/annotations/inchikey_to_jcp2022_mapping_combined.csv", index=False)
        print("Combined mapping saved.")