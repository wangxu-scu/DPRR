# DRPP Project

## This project implements the cross-modal hashing method DRPP.

## Quick Start

1. **Enter the project directory**

   ```bash
   cd ./DRPP/
   ```

2. **Prepare the dataset**

   - Place dataset into the `./data/` directory.
   - Run the data preprocessing script:
     ```bash
     python ./utils/tools.py
     ```

3. **Generate noisy labels**

   - Run the following command to generate labels with simulated noise:
     ```bash
     python ./noise_label/generate.py
     ```

4. **Train the model**
   - Start training using the specified GPU and parameter:
     ```bash
     python ./train.py --gpus=0 --temperature=0.2 --lambda_class=0.5
     ```

---

## Directory Structure

```
DRPP/
├── data/                # Raw and processed datasets
├── utils/               # Utility functions
│   └── tools.py         # Data preprocessing script
├── noise_label/         # Noisy label generation
│   └── generate.py      # Script to generate noisy labels
├── train.py             # Model training script
└── README.md            # Project documentation
```
