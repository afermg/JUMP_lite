SET enable_progress_bar = true;
COPY (PIVOT (
        SELECT
            filename,
            metric,
            any_value(object) AS model,
            MEAN(value) AS value,
            parse_filename(any_value(filename), true) AS site
        FROM read_parquet('/work/datasets/aliby_output/morphem/jump_core_annotated/jpegxl_lossy_mq.zarr/profiles/*.parquet', filename=True)
        GROUP BY filename, metric
    )
    ON metric
    USING any_value(value))
    TO 'perwell.parquet'
