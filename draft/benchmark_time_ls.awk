# Print the time stats for folders listed using `ls`
# Call using `ls -lU --time-style=+%s /work/datasets/aliby_output/*/jump_target2_4plate/zstd.zarr/profiles  |awk -f benchmark_time.awk`
# --
# Example output:
# 
# *Model* *Images/min* *Hours(total)* *#images*
# openphenom 379.259 0.405 9216
# subcell 270.793 0.567222 9216
# cp_measure 0.508605 302.003 9216
# morphem 218.302 0.703611 9216
# dinov2_random 284.591 0.539722 9216

BEGIN {printf "*Model* *Images/min* *Hours(total)* *#images*\n"};
/^\// {
    group_starts=NR;
    {
	if (NR>1)
	    names[s[5]]=max-min;
    }
    split($0, s, "/");
    nfiles[s[5]] = 0;
} 
{
    if ($6>1767225600)
    {if (NR == group_starts+2) min = max = $6};
    if ($0 ~ /parquet/) {
	nfiles[s[5]]++;
	if ($6 < min) min = $6;
	if ($6 > max) max = $6;
    };
}
END {
	names[s[5]]=max-min;
	for (key in names) printf("%s %s %s %s\n", key, nfiles[key]/names[key] * 60, names[key]/3600, nfiles[key]) }
