import os
from math import sqrt
import numpy as np
from ase.io import read, write


class CreateStack:
    """
    A class for creating and manipulating bilayers of 2D materials
    with various stacking configurations. The two main inputs are the
    filename and the interlayer distance. A default interlayer height of
    3.2 Angstroms is assumed, which is typical of most 2D COFs

    Attributes:
    -----------
        - filename : str
            The path to the input file containing the monolayer structure.
        - interlayer_dist : float
            The distance between layers in the bilayer structure.
        - base_name : str
            The base name derived from the filename for output file naming.
        - monolayer : ASE Atoms object
            The monolayer structure loaded from the file.
        - bilayer : ASE Atoms object
            The bilayer structure created by stacking the monolayer.
        - x : float
            The length of the unit cell along the x-axis.
        - y : float
            The length of the unit cell along the y-axis.
        - z : float
            The length of the unit cell along the z-axis.
    """

    def __init__(self, filename, interlayer_dist=4.5, output_dir="."):
        """
        Initializes the CreateStack object by reading the monolayer structure,
        creating a bilayer, and setting up necessary parameters.

        **Parameters:**
            - filename : str
                The path to the input file containing the monolayer structure.
            - interlayer_dist : float, optional
                The distance between layers in the bilayer (default is 3.2).
            - output_dir : str, optional
                The directory to store the output files (default is ".").
        """
        self.filename = filename
        self.interlayer_dist = interlayer_dist
        self.output_dir = output_dir
        self.base_name = os.path.basename(filename).split('.')[0]
        self.monolayer = read(filename)
        self.monolayer.cell[2, 2] = interlayer_dist
        self.bilayer = self.monolayer * (1, 1, 2)

        x, y, z = self.monolayer.cell.lengths().tolist()
        self.x = x
        self.y = y
        self.z = z

    def create_ab_stacking(self):
        """
        Creates an AB-stacked bilayer by shifting the second layer along the
        a- and b-cell vectors and writes the resulting structure to a CIF file.
        """
        cell_a = sqrt(self.monolayer.cell[0, 0]**2 + self.monolayer.cell[0, 1]**2 + self.monolayer.cell[0, 2]**2)
        cell_b = sqrt(self.monolayer.cell[1, 0]**2 + self.monolayer.cell[1, 1]**2 + self.monolayer.cell[1, 2]**2)
        ab_stack = self.bilayer.copy()
        ab_stack.positions[len(self.monolayer):] += [cell_a / 2, cell_b / 6 * sqrt(3), 0]
        output_file = os.path.join(self.output_dir, self.base_name + '_ab.cif')
        ab_stack.write(output_file)
        print(f"AB-stacked structure saved to: {output_file}")

    def create_aa_stacking(self):
        """
        Creates an AA-stacked bilayer without any relative shift between layers
        and writes the resulting structure to a CIF file.
        """
        output_file = os.path.join(self.output_dir, self.base_name + '_aa.cif')
        self.bilayer.write(output_file)
        print(f"AA-stacked structure saved to: {output_file}")

    def stack_along_x(self, max_length=None):
        """
        Creates a series of stacked structures by translating the second layer
        along the x-axis and writes each structure to a CIF file.

        Parameters:
        -----------
        max_length : float, optional
            The maximum translation length along the x-axis (default is half of the x-cell length).
        """
        if max_length is None:
            max_length = self.x / 2.0
        loop_values = np.arange(0.5, max_length, 0.5)
        x_bilayer = self.bilayer.copy()
        for index_ in loop_values:
            x_bilayer = self.bilayer.copy()
            x_bilayer.positions[len(self.monolayer):] += [index_, 0, 0]
            output_file = os.path.join(self.output_dir, f"{self.base_name}_x_{index_}.cif")
            x_bilayer.write(output_file)
            print(f"X-translated structure saved to: {output_file}")

    def stack_along_y(self, max_length=None):
        """
        Creates a series of stacked structures by translating the second layer
        along the y-axis and writes each structure to a CIF file.

        Parameters:
        -----------
        max_length : float, optional
            The maximum translation length along the y-axis (default is half of the y-cell length).
        """
        if max_length is None:
            max_length = self.y / 2.0
        loop_values = np.arange(0.5, max_length, 0.5)
        y_bilayer = self.bilayer.copy()
        for index_ in loop_values:
            y_bilayer = self.bilayer.copy()
            y_bilayer.positions[len(self.monolayer):] += [0, index_, 0]
            output_file = os.path.join(self.output_dir, f"{self.base_name}_y_{index_}.cif")
            y_bilayer.write(output_file)
            print(f"Y-translated structure saved to: {output_file}")

    def stack_along_xy(self, max_length=None):
        """
        Creates a series of stacked structures by translating the second layer
        along both the x- and y-axes and writes each structure to a CIF file.

        Parameters:
        -----------
        max_length : float, optional
            The maximum translation length along the xy-diagonal (default is the average of x and y cell lengths).
        """
        if max_length is None:
            max_length = (self.y + self.x) / 2.0
        loop_values = np.arange(0.5, max_length, 0.5)
        xy_bilayer = self.bilayer.copy()
        for index_ in loop_values:
            xy_bilayer = self.bilayer.copy()
            xy_bilayer.positions[len(self.monolayer):] += [index_, index_, 0]
            output_file = os.path.join(self.output_dir, f"{self.base_name}_xy_{index_}.cif")
            xy_bilayer.write(output_file)
            print(f"XY-translated structure saved to: {output_file}")