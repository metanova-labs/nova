slurm-example
=============

This directory contains an example for running BoltzGen on SLURM.

The approach we use is to launch a job array that runs many single-GPU jobs. Each job calls "boltzgen run" to go through all pipeline steps (design generation, refolding, etc.) for a small number of designs. After all the jobs have completed, on a login node we use "boltzgen merge" to merge all individual task results together, then "boltzgen run --steps filtering" to apply
filters to the combined set.

Note that other parallelization strategies could also work here. In particular, the individual BoltzGen pipeline steps (e.g. design generation, refolding) support
parallelization across multiple GPUs within the step.

You will need to modify [run.sh](run.sh) and probably also [run_job_array.slurm](run_job_array.slurm) for your site and analysis task.

After you've made your modifications, you can run:

```
$ bash run.sh submit
```

To submit the job array.

After that has finished, you can run:

```
$ bash run.sh process
```

to merge and filter the results.