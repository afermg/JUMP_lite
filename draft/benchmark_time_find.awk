# Print the time stats for folders listed using `find`
#
# Usage example
# 
# export LC_COLLATE=C
# find /work/datasets/aliby_output/*/jump_target2_4plate/zstd.zarr/profiles/  -type f -newermt "2025-12-01 0:00:00" -printf "%p %T@\n" | awk -f benchmark_time_find.awk | sort
# 
# #+RESULTS:
# | *Model*       | *Images/min* | *Hours(total)* | *#images* |
# | cp_measure    |    1.30622 |      53.7429 |    4212 |
# | dinov2_random |    284.631 |     0.539645 |    9216 |
# | morphem       |    218.363 |     0.703416 |    9216 |
# | openphenom    |    379.248 |     0.405012 |    9216 |
# | subcell       |    270.833 |     0.567138 |    9216 |

BEGIN {
    header = "Model Images/min Hours(total) #images"

    printf("%s\n", header);
};
 {
     split($1, d, "/")
     g = d[5]
     {
	 if (nfiles[g]==0)
	 {
	     min[g] = $2;
	     max[g] = $2;
	 }
	 else
	 {
	     if ($2 < min[g]) min[g] = $2;	
	     if ($2 > max[g]) max[g] = $2;
		
	 }
     }
     nfiles[g]++;
 };
 END {
     for (k in nfiles)
     {
	 delta = max[k]-min[k];
	 printf("%s %.1f %.1f %s\n", k, nfiles[k]/delta*60, delta/3600, nfiles[k]);};

 };
