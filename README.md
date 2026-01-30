# Factor Research Playground Project Setup and Usage

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/jerrychen04/data-quality.git
    cd data-quality
    ```

2. **Create a conda environment:**
    If you do not have conda installed, install it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Make sure you cd into the root of this repository before running this cmd:

    ```sh
    conda env create -f environment.yml
    ```

3. **Activate the conda virtual environment:**

    Using conda (make sure to activate everytime you create a new shell instance):
        
    ```sh
    conda activate data-quality
    ```

## Running the Main Script (sample mean reversion strategy implementation) as .py file

To run the main script, execute the following command:
```sh
python backtester/main.py
```

## Running sample research notebooks:

To run a .ipynb, like `bab.ipynb`, simply select the data-quality kernel and run the scripts. If not visible, select under > Select Another Kernel > Python Environments. **[IMPORTANT] When running on cached data, make sure date ranges align with your order generator**.

## Running Tests

To run the unit tests, use the following command. Note: tests are not freshly maintained at the moment:
```sh
python -m unittest discover -s unit_tests
```

## To Create A Strategy

Go to `order_generator.py` and create a new instance of `OrderGenerator` in a new file for cleanliness. See example code in main.py to run strategy. Write corresponding unit tests as needed. Run research runs in .ipynb and restart kernel when making package changes.
