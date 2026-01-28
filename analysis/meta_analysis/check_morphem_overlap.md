## Comparing training set overlap in JUMP for Morphem

Morphem was trained on the CHAMMI-75 dataset, consisting of a large colection of dataset including JUMP.
Given our use of the model in this performance comparison, we here check the overlap between the data used to train it.
We compare it to the subset included in JUMP-lite.

## Steps

### Install aws:

nix shell nixpkgs#awscli2

### Download training data metadata:

aws s3 cp --no-sign-request s3://chammi-data/CHAMMI-75/CHAMMI-75_train_metadata.csv ./

### Matching JUMP pattern and storing to occurrences.txt

rg jump CHAMMI-75_train_metadata.csv | awk -F',' '{print $26}' | awk -F'/' '{count[$8]++} END {for (item in count) print count[item], item}' | sort -n | tee occurrences.txt

### Checking annotated field for "experiment.study_type"

rg jump CHAMMI-75_train_metadata.csv| awk -F',' '{count[$NF]++;} END {for (item in count) print count[item], item}'

733705 Compound rows

### Compating overlap between JUMP data used in CHAMMI-75 and JUMP-lite:

```
duckdb -c " SELECT (SELECT COUNT(DISTINCT plate_id) FROM read_csv('occurrences.txt', delim=' ', header=false, columns={'count': 'INT', 'plate_id': 'VARCHAR'})) as txt_plates, (SELECT COUNT(DISTINCT Metadata_Plate) FROM '../metadata/metadata_filtered.parquet') as parquet_plates, (SELECT COUNT(DISTINCT t.plate_id) FROM read_csv('occurrences.txt', delim=' ', header=false, columns={'count': 'INT', 'plate_id': 'VARCHAR'}) t WHERE t.plate_id IN (SELECT DISTINCT Metadata_Plate FROM '../metadata/metadata_filtered.parquet')) as matching_plates "
```
``` markdown
┌────────────┬────────────────┬─────────────────┐
│ txt_plates │ parquet_plates │ matching_plates │
│   int64    │     int64      │      int64      │
├────────────┼────────────────┼─────────────────┤
│    1876    │      624       │       383       │
└────────────┴────────────────┴─────────────────┘
```

