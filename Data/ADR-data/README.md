This directory contains the final processed data used in the paper.

The main data file is `data.jsonl`, which contains the following fields:

- `id`: Unique identifier for the ADR.
- `context`: The context of the ADR, which includes the title, description, and comments.
- `decision`: The decision made in the ADR.

This dataset was further divided into training, validation, and test sets. The splits are as follows:

- `data_train.jsonl`: Training set containing 80% of the data. (2946 ADRs)
- `data_val.jsonl`: Validation set containing 10% of the data. (982 ADRs)
- `data_test.jsonl`: Test set containing 10% of the data. (983 ADRs)
