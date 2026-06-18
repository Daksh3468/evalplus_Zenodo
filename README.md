# When Faster Isn’t Greener: The Hidden Costs of LLM-Based Code Optimization

This repository is the replication package for the paper "When Faster Isn’t Greener: The Hidden Costs of LLM-Based Code
Optimization" submitted at ASE'25. It is a fork from https://github.com/evalplus/evalplus/. We thank the authors of
EvalPlus and EvalPerf for their work which accelerated our research.

## Results analysis - Companion notebook

The notebook runs using Marimo (and not Jupyter). Thus, the outputs of the notebook are _not stored_, this means that in
order to view the outputs of the notebook, you need to run it yourself. We included a static view of the notebook in
`analysis/notebook.html` that you can view on your browser. To run the notebook, you will need Python 3.13

Setup:

To install the dependencies, as well as Marimo, run:

```shell
pip install -r requirements.txt
```

To view the notebook (the notebook will be computed at run time):

```shell
marimo run analysis/notebook.py
```

To edit the notebook:

```shell
marimo edit analysis/notebook.py
```

## How to run the experiment

### Optimization phase

#### Requirements

You must be on a machine with Linux, with the `perf` utility installed as well as `nvidia-smi`. This machine needs at
least one GPU. This experiment can technically run without GPUs, but it will be significantly slower, and the code may
need to be adapted. Python 3.13 is required as well as C++.

Make sure the path to this repository does not contain any space, otherwise some part of the cirron library will crash.

#### Running VLLM

Run VLLM on a server:

```shell
while true; do docker run --gpus all \
    -v $HUGGING_FACE_HOME:/root/.cache/huggingface \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.7.3 \
    --model $MODEL \
    -tp 4; sleep 10
done;
```

Set HUGGING_FACE_HOME to the root of your hugging face install so that the model is downloaded
there, if you don't want to save the model on the device, skip the `-v` option. Set MODEL to the huggingface name of the
LLM you want to use.
Set `-tp` to the number of GPUs you have.

The docker execution is in a loop to allow automatic rebooting in case of VLLM crashes.

#### Running the optimizations

**On the same machine**, in the root of this repository, run:

```shell
export SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 # Necessary to install with pip without a git repository
pip install ".[perf]"
echo "-1" | sudo tee -a /proc/sys/kernel/perf_event_paranoid # Allows energy measurements
```

Then, depending on the optimizer run the following command:

#### Single-code optimizers

```shell
for optimizer in "simple" "cod" "cot" "self-refine-exec-feedback" "self-refine-nl-feedback"; do evalplus.evalopti --model $MODEL --backend openai --base-url http://localhost:8000/v1 --temperature 0.2 --optimizer $optimizer; done;
```

#### LLM4EFFI

```shell
evalplus.evalopti --model $MODEL --backend openai --base-url http://localhost:8000/v1 --optimizer llm4effi --temperature 0.2 --n-samples 5 --max-profile 100 ;
```

#### EoH

```shell
evalplus.evalopti --model $MODEL --backend openai --base-url http://localhost:8000/v1 --optimizer eoh --temperature 0.2 --n-samples 3 --max-profile 1000
```

#### Simple10

```shell
evalplus.evalopti --model $MODEL --backend openai --base-url http://localhost:8000/v1 --optimizer simple10 --temperature 0.2 --n-samples 5 --max-profile 1000 --skip-iter 0 --skip-prev-config-check --force-n-samples-with-prev-results
```

### Program energy evaluation phase

#### Requirements

You must have a Linux machine with the `perf` utility installed. Python 3.13 is required

Make sure the repository is installed correctly and the machine allows for energy measurements using:

```shell
export SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 # Necessary to install with pip without a git repository
pip install ".[perf]"
echo "-1" | sudo tee -a /proc/sys/kernel/perf_event_paranoid
```

#### Running the evaluation

In the root of the repository, run:

```shell
evalplus.profile-energy --calibration-time 30 --nb-workers 5 --min-duration 1 --max-profile 1000
```

If you want to evaluate only a subset of the non-evaluated samples, you can filter using --pattern (e.g. --pattern "
evalperf/*simple10*evalopti_results.json" to evaluate only Simple10 results) or
split the work
between multiple machines using --split-start and --split-end (e.g. --split-start 20 --split-end 30 to only process 10
files.)

### Fetching the data

When the experiment is done running, make sure the data is downloaded on the machine you will use to run the notebook (
You don't need a big computer, only ~10GiB of free memory). The results of the experiments will be in
`evalplus_results/evalperf`.
