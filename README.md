# Incremental Attribute Reduction Under Two Dimontionally Variation

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/JsSparkyyx/IAR2DV
cd IAR2DV
```

Install dependencies by

```bash
pip install -r requirements.txt
```

## Run Experiments

### Run on the existing dataset

You can use `python src/main.py --exp <exp_name>` to run experiment 
AAO, DAO, IARAAO, IARDAO, IARAADO or IARDAAO on example dataset.

### Run on your own datasets

If you want to run experiment on your own dataset, you should first prepare the following file:
- data_name.csv: Each line represents an object, which contains several tokens `<attribute 1>,...,<attribute n>,<label>` where attribute and label should be a number. You should use `\t` as the delimiter.

Then you can run experiment on your own dataset by `python src/main.py --exp <exp_name> --data <data_path>`. 
