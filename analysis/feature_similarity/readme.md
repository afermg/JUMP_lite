# Do feature similarity analysis



Run:

cd analysis/feature_similarity/scripts
uv run python feature_correlation_cp_measure_script.py  



Run segementation overlap results

python analysis/compare_segmentations.py --root /work/datasets/aliby_output/cp_measure/jump_target2_4plate --ground-truth zstd.zarr --methods jpegxl_lossy_hq.zarr jpegxl_lossy_mq.zarr jpegxl_lossy_lq.zarr   jpegxl_lossy_effort_3.zarr --output result_summary/segmentation_comparison --workers 4 --segment-step segment_cell  

python analysis/compare_segmentations.py --root /work/datasets/aliby_output/cp_measure/jump_target2_4plate --ground-truth zstd.zarr --methods jpegxl_lossy_hq.zarr jpegxl_lossy_mq.zarr jpegxl_lossy_lq.zarr jpegxl_lossy_effort_3.zarr --output result_summary/segmentation_comparison --workers 4 --segment-step segment_cell  --well source_5__JUMPCPE-20210623-Run02_20210624_225846__ACPJUM012__F19__1 --visualize-sample