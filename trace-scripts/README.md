## Trace File Analysis

Scripts to analyze and compare trace files from ONNX Runtime.

### Implemented:

So far, these features are implemented
* Single trace file analysis, either use the Python notebook or `perfstats.py`.
* `ratio_of_execution_time_in_[ex_provider]`
* `total_[ex_provider]_execution_time`
* `total_execution_time`
* `total_ops`
* `total_ops_in_[ex_provider]`
* ratio of op time per execution provider
* Comparing operations across two trace files (improvement on granularity is needed.)


## Analyze a Singe File
```
python perfstats-single.py [flags] [path to json trace file]
```

The available flags are:
* `-t`, breakdown by optype
* `-p`, breakdown by execution provider
* `-po`, breakdown by optype per execution provider
* `-n`, breakdown by node
* `-s`, breakdown by step

Use `-h` to see all available options.

## Compare Across Two Files

```
python perfstats-compare.py -t [path to accelerated json trace file] -b [path to baseline json trace file]
```

## TO-DO
* Enable per op time per execution provider breakdown in comparison script
* Integrate comparison scripts and single-run script together.