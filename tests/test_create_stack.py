# # tests/test_create_stack.py

# import os
# import pytest
# from stacking_2d_systems.slip_layers import CreateStack

# SAMPLE_CIF = "tests/monolayer.cif"

# @pytest.fixture
# def create_stack_instance(tmp_path):
#     """Fixture to create a CreateStack instance for testing."""
#     output_dir = tmp_path / "output"
#     output_dir.mkdir()
#     stack = CreateStack(SAMPLE_CIF, interlayer_dist=3.2, output_dir=output_dir)
#     return stack


# def test_create_ab_stacking(create_stack_instance):
#     """Test that the AB stacking is created correctly."""
#     create_stack_instance.create_ab_stacking()
#     output_file = os.path.join(create_stack_instance.output_dir, create_stack_instance.base_name + "_ab.cif")
#     assert os.path.exists(output_file), f"AB stacking file should be created at {output_file}."


# def test_create_aa_stacking(create_stack_instance):
#     """Test that the AA stacking is created correctly."""
#     create_stack_instance.create_aa_stacking()
#     output_file = os.path.join(create_stack_instance.output_dir, create_stack_instance.base_name + "_aa.cif")
#     assert os.path.exists(output_file), f"AA stacking file should be created at {output_file}."


# def test_stack_along_x(create_stack_instance):
#     """Test stacking along the x-axis."""
#     max_length = 5.0
#     create_stack_instance.stack_along_x(max_length=max_length)
#     expected_files = [
#         os.path.join(create_stack_instance.output_dir, create_stack_instance.base_name + f"_x_{i/2}.cif")
#         for i in range(1, int(max_length * 2) + 1)
#     ]
#     for file in expected_files:
#         assert os.path.exists(file), f"File {file} should be created for x translation."


# def test_stack_along_y(create_stack_instance):
#     """Test stacking along the y-axis."""
#     max_length = 4.0
#     create_stack_instance.stack_along_y(max_length=max_length)
#     expected_files = [
#         os.path.join(create_stack_instance.output_dir, create_stack_instance.base_name + f"_y_{i/2}.cif")
#         for i in range(1, int(max_length * 2) + 1)
#     ]
#     for file in expected_files:
#         assert os.path.exists(file), f"File {file} should be created for y translation."


# def test_stack_along_xy(create_stack_instance):
#     """Test stacking along the xy-direction."""
#     max_length = 6.0
#     create_stack_instance.stack_along_xy(max_length=max_length)
#     expected_files = [
#         os.path.join(create_stack_instance.output_dir, create_stack_instance.base_name + f"_xy_{i/2}.cif")
#         for i in range(1, int(max_length * 2) + 1)
#     ]
#     for file in expected_files:
#         assert os.path.exists(file), f"File {file} should be created for xy translation."
