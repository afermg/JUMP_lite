-- Feature extraction for DL models (morphem, dinov2, etc.) from aliby output
-- Produces the same columns as extract_features.py
-- Single streaming query — avoids materializing 176GB into memory.
--
-- Usage:
--   duckdb < analysis/get_perwell_dl.sql
--
-- To change model/dataset/compression, update the SET VARIABLE lines below.

SET enable_progress_bar = true;
SET temp_directory = '/tmp/duckdb_tmp';

-- Configuration
SET VARIABLE input_path = '/work/datasets/aliby_output/jump_lite_rerun/jump_lite_updated/morphem/jpegxl_lossy_mq.zarr/profiles/*.parquet';
SET VARIABLE model_name = 'morphem';
SET VARIABLE dataset_name = 'jump_lite_updated';
SET VARIABLE compression_name = 'jpegxl_lossy_mq.zarr';
SET VARIABLE output_path = 'data/features/jump_lite/morphem_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet';
SET VARIABLE output_dir = 'data/features/jump_lite';

-- Single streaming query: read → aggregate → pivot → write
-- No intermediate tables, DuckDB can optimize the full pipeline
COPY (
    SELECT
        well_id AS Metadata_id,
        string_split(well_id, '__')[1] AS Metadata_Source,
        string_split(well_id, '__')[2] AS Metadata_Batch,
        string_split(well_id, '__')[3] AS Metadata_Plate,
        string_split(well_id, '__')[4] AS Metadata_Well,
        '1' AS Metadata_Site,
        getvariable('model_name') AS Metadata_model,
        getvariable('dataset_name') AS Metadata_dataset,
        getvariable('compression_name') AS Metadata_compression,
        * EXCLUDE (well_id)
    FROM (
        PIVOT (
            SELECT
                string_split(parse_filename(filename, true), '__')[1] || '__' ||
                string_split(parse_filename(filename, true), '__')[2] || '__' ||
                string_split(parse_filename(filename, true), '__')[3] || '__' ||
                string_split(parse_filename(filename, true), '__')[4] AS well_id,
                branch || metric AS full_metric_name,
                object,
                mean(value) AS cvalue
            FROM read_parquet(getvariable('input_path'), filename=true)
            GROUP BY tp, well_id, branch, metric, object
        )
        ON object, full_metric_name
        USING any_value(cvalue)
    )
) TO 'data/features/jump_lite/morphem_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet' (FORMAT PARQUET);
