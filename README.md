# stacking_2d_systems

**stacking_2d_systems** is a Python package designed to create different stacking configurations of 2D systems. It allows users to easily generate various configurations of 2D materials — such as **AA** and **AB** stacking — and apply **custom translations** along the *x*, *y*, or *xy* directions.

This package is especially useful for researchers who wish to create different stacking arrangement to match experimental **Powder X-ray Diffraction (PXRD)** with different stacking patterns of computationally generated structures.

---

## Features

- **AB and AA Stacking**
  Generate AB- and AA-stacked bilayer structures from a single monolayer input file.

- **Custom Translations**
  Create slip-stacked configurations by translating the second layer along the x, y, or xy directions.

- **Batch Generation**
  Automatically generate all possible stacking configurations with one command.

- **User-Friendly CLI**
  Generate stacking configurations directly from the terminal — no Python coding required.

- **Flexible Python Library**
  Use the `CreateStack` class in your Python scripts for full control and integration into custom workflows.

## Installation

### From PyPI

Install directly from [PyPI](https://pypi.org/project/stacking_2d_systems):

    ```bash
        pip install stacking_2d_systems
    ```

### From GitHub

    ```bash

        git clone https://github.com/bafgreat/stacking_2d_systems.git
        cd stacking_2d_systems
        pip install .
    ```

## Usage

### Command-Line Interface (CLI)

The stacking_2d_systems package provides a user-friendly command-line tool called create-stack for generating stacking configurations without needing to write Python code.

    ```bash
        create-stack path/to/monolayer.cif --stacking <type> [options]
    ```

#### Available Options

**filename:**
        Path to the input monolayer structure file (e.g., .cif).

**--interlayer-dist:**
        Distance between the layers in the bilayer structure. Default is 3.2.

**--output-dir:**
    The folder, which specifies where the generated files should be saved. The default is the current working directory where the code is run.

**--stacking:**
    Type of stacking configuration to generate. Options include:
        - ab: Generates an AB-stacked bilayer.
        - aa: Generates an AA-stacked bilayer.
        - x: Translates the second layer along the x-axis.
        - y: Translates the second layer along the y-axis.
        - xy: Translates the second layer along both x and y directions.
        - all: Generates all stacking configurations (AB, AA, x, y, xy).
**--max-length:**
        Maximum translation length along the specified axis (x, y, or xy). If not specified, defaults to half of the corresponding cell dimension

#### Examples

1. Generate AB Stacking

        ```bash
            create-stack path/to/monolayer.cif --stacking ab
        ```

if you wish to that the stack files should written in a particular directory

    ```bash
        create-stack path/to/monolayer.cif --stacking ab --output-dir path/to/output
    ```

2. Generate AA Stacking

    ```bash
    create-stack path/to/monolayer.cif --stacking aa
    ```

3. Custom Stacking Along the x-axis
To translate the second layer along the x-axis with a maximum translation length of 5.0 Angstroms:

    ```bash
        create-stack path/to/monolayer.cif --stacking x --max-length 5.0
    ```

This code will generate a series of files like monolayer_x_0.5.cif, monolayer_x_1.0.cif, etc.

4.Generate All Stacking Configurations
To automatically create AB, AA, and all custom translations along x, y, and xy

        ```bash
            create-stack path/to/monolayer.cif --stacking all
        ```

This code will generate all stacking configurations. At the maximum length for the custom translations will be half the size of the lattice length. i.e for translation along x, the maximum length is a/2.0 and b/2.0 along y and a+b/2.0 for xy.

## Library Usage

To have full control over the stacking process, the CreateStack class can be used directly in Python scripts.

    ```Python

    from stacking_2d_systems.slip_layers import CreateStack

    # Initialize the stack with a monolayer structure and interlayer distance
    stack = CreateStack('path/to/monolayer.cif', interlayer_dist=3.2)

    # Create an AB-stacked structure
    stack.create_ab_stacking()

    # Create an AA-stacked structure
    stack.create_aa_stacking()

    # Create custom stacking along the x-axis with a max length of 5.0 units
    stack.stack_along_x(max_length=5.0)
    # Create custom stacking along the y-axis
    stack.stack_along_y()
     # Create custom stacking along the xy-axis
    stack.stack_along_xy()
    ```

### Methods in CreateStack

- create_ab_stacking():
    Generates an AB-stacked structure and saves it as *_ab.cif.
- create_aa_stacking():
    Generates an AA-stacked structure and saves it as *_aa.cif.
- stack_along_x(max_length):
    Generates a series of x-axis translated structures.
- stack_along_y(max_length):
    Generates a series of y-axis translated structures.
- stack_along_xy(max_length):
    Generates a series of xy-diagonal translated structures.

## Contact

If you have any questions or need further help, feel free to contact the project maintainer at `bafgreat@gmail.com`.

## License

MIT License
