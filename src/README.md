# Overview of core experiments


## $ Target2 plate key analysis
Plates 

JCPQC016
ACPJUM012
BR00121438
110000293081

### Command to copy plate data to correct folder
``` bash
fd "JCPQC016" | xargs -I {} cp {} /work/datasets/jump_target2_4plate/raw/
fd "ACPJUM012" | xargs -I {} cp {} /work/datasets/jump_target2_4plate/raw/
fd "BR00121438" | xargs -I {} cp {} /work/datasets/jump_target2_4plate/raw/
fd "110000293081" | xargs -I {} cp {} /work/datasets/jump_target2_4plate/raw/
```

## Downloading annotations from:


``` bash
wget "https://zenodo.org/records/18197517/files/annotations_compound_compound.parquet"
wget "https://zenodo.org/records/18197517/files/annotations_compound_gene.parquet"
wget "https://zenodo.org/records/18197517/files/annotations_compound_gene_curated.parquet"
wget "https://zenodo.org/records/18197517/files/annotations_gene_gene.parquet"
```

