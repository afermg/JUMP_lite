# Example
# bash get_perwell.sh /work/datasets/aliby_output/openphenom/jump_core_annotated/jpegxl_lossy_mq.zarr/profiles /work/datasets/aliby_output/openphenom/jump_core_annotated/output.parquet
mkdir -p $(dirname "${2}")
duckdb <<-EOF
	SET enable_progress_bar = true;
	COPY (
	  PIVOT (
	    SELECT
	      filename,
	      metric,
	      any_value(object) AS model,
	      MEAN(value) AS value,
	      parse_filename(any_value(filename), TRUE) AS site
	    FROM
	      read_parquet('${1}/*.parquet', filename = TRUE)
	    GROUP BY
	      filename,
	      metric
	  ) ON metric USING any_value(value)
	) TO '${2}';
EOF
