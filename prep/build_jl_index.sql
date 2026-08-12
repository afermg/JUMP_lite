-- Build the URI manifest (jl_index_tidy.parquet) used by prep/download_raw.py.
--
-- Source: JUMP-cellpainting/datasets on GitHub (plate.csv.gz) +
--         per-plate load_data_with_illum.csv from cellpainting-gallery S3 +
--         JUMP-lite plate list from Zenodo (record 18705140 / jl_plates.csv).
--
-- Run from the directory where the parquet outputs should land, e.g.:
--   mkdir -p data/manifest && cd data/manifest && pixi run duckdb < ../../prep/build_jl_index.sql

.maxwidth 80
.maxrows 4
INSTALL httpfs;
LOAD httpfs;

CREATE OR REPLACE TABLE redlist AS (SELECT COLUMNS('^Metadata_(Source|Batch|Plate)$') FROM read_csv('https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz') WHERE
((Metadata_Source = 'source_4' AND suffix(Metadata_Batch, 'Batch12')) OR
(Metadata_Source = 'source_3' AND (regexp_matches(Metadata_Batch, '^CP_3[23456]_all_Phenix1$') OR Metadata_Batch in ['CP59', 'CP60'])) OR
((Metadata_Source = 'source_15') AND Metadata_Plate in ['PEP00004458', 'PEP00004421'])));
COPY redlist TO 'redlist.csv';
FROM redlist;

CREATE OR REPLACE TABLE graylist AS (SELECT COLUMNS('^Metadata_(Source|Batch|Plate)$') FROM read_csv('https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz') WHERE regexp_matches(Metadata_Plate, 'CP-CC9-R[123456]-28'));
COPY graylist TO 'graylist.csv';
FROM graylist;

CREATE OR REPLACE TABLE jump_plates AS (SELECT COLUMNS('^Metadata_(Source|Batch|Plate)$') FROM read_csv('https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz') ANTI JOIN (FROM 'graylist.csv' UNION ALL FROM 'redlist.csv') using(Metadata_Source,Metadata_Batch,Metadata_Plate));
FROM jump_plates;

.maxwidth 80
CREATE OR REPLACE TABLE loaddata_uris AS (SELECT *, format('s3://cellpainting-gallery/cpg0016-jump/{}/workspace/load_data_csv/{}/{}/load_data_with_illum.csv', Metadata_Source, Metadata_Batch, Metadata_Plate) AS uri FROM jump_plates);
FROM loaddata_uris;
SET VARIABLE csv_files = (SELECT list(uri) FROM loaddata_uris);

CREATE OR REPLACE TABLE jump_index AS (SELECT COLUMNS('^Metadata_(Source|Batch|Plate|Well|Site)$'),COLUMNS('URL_Orig(DNA|Mito|AGP|ER|RNA)')  FROM read_csv(getVariable('csv_files'), union_by_name=True));
FROM jump_index;

CREATE OR REPLACE TABLE jump_index_tidy AS (
UNPIVOT jump_index ON COLUMNS('URL_*') INTO NAME Metadata_Channel VALUE uri);
COPY jump_index_tidy TO 'jump_index_tidy.parquet';
FROM jump_index_tidy;

CREATE OR REPLACE TABLE jl_index AS (FROM jump_index JOIN (SELECT #2 AS Metadata_Plate FROM read_csv('https://zenodo.org/api/records/18705140/files/jl_plates.csv/content')) using(Metadata_Plate));
COPY jl_index TO 'jl_index.parquet';
FROM jl_index;

CREATE OR REPLACE TABLE jl_index_tidy AS (
UNPIVOT jl_index ON COLUMNS('URL_*') INTO NAME Metadata_Channel VALUE uri);
COPY jl_index_tidy TO 'jl_index_tidy.parquet';
FROM jl_index_tidy;

-- Build the broad plate-level source index and deterministically retain at most
-- four sites per well. The v1.0 release is the paper-cohort subset frozen in
-- metadata/jump_lite_v1_site_manifest.parquet; release builders must semi-join
-- to that manifest rather than upload every well produced here. Include the
-- complete location identity so identically named plates cannot mix.
CREATE OR REPLACE TABLE jl_index_sampled AS (
  SELECT * EXCLUDE rn
  FROM (
    SELECT
      *,
      row_number() OVER (
        PARTITION BY Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_Well
        ORDER BY Metadata_Site
      ) AS rn
    FROM jl_index
  )
  WHERE rn <= 4
);
COPY jl_index_sampled TO 'jl_index_sampled.parquet';
FROM jl_index_sampled;

CREATE OR REPLACE TABLE jl_index_sampled_tidy AS (
UNPIVOT jl_index_sampled On COLUMNS('URL_*') INTO NAME Metadata_Channel VALUE uri);
COPY jl_index_sampled_tidy TO 'jl_index_sampled_tidy.parquet';
FROM jl_index_sampled_tidy;
