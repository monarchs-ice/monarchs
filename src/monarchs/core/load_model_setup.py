"""
Module containing the ModelSetup class, used to load in and hold the
configuration used by MONARCHS.
"""

import importlib.util
import ast

# List of safe imports that are allowed in a model setup script.
SAFE_IMPORTS = {
    "monarchs",
    "numpy",
    "netCDF4",
    "math",
    "matplotlib",
    "scipy",
    "datetime",
    "pandas",
}
MODULE_NAME = "monarchs.core.load_model_setup"


class ModelSetup:
    """
    Class to load in the model setup from a user-specified Python script.
    """

    def __init__(self, script_path):
        self.errors = []
        self.script_path = script_path
        print(f"Loading model setup from {self.script_path}")
        # Run validation checks before we use importlib to load it in.
        self.validate_model_setup()
        spec = importlib.util.spec_from_file_location(
            "model_setup", self.script_path
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        for var_name in dir(config_module):
            # don't load in dunder attributes like __name__
            if var_name.startswith("__"):
                continue
            var_value = getattr(config_module, var_name)
            # don't load in functions or classes
            if not callable(var_value):
                setattr(self, var_name, var_value)

    def validate_model_setup(self):
        """
        Run the validation checks. If any of them fail, print out all the
        errors found for each.
        """
        method_name = (
            "monarchs.core.load_model_setup.ModelSetup.validate_model_setup"
        )
        self.check_file_exists()
        self.check_for_key_variables()
        self.check_for_unexpected_imports()
        if self.errors:
            error_message = "\n".join(self.errors)
            raise ValueError(
                f"{method_name}: Errors found in model setup:\n"
                f"{error_message}"
            )

    def check_file_exists(self):
        """Check to see whether the user-provided runscript works."""
        method_name = (
            "monarchs.core.load_model_setup.ModelSetup.check_file_exists"
        )
        try:
            with open(self.script_path, "r", encoding="utf-8") as file:
                file.read()
        except FileNotFoundError:
            self.errors.extend(
                [
                    f"{method_name}: Path to runscript"
                    f" ({self.script_path}) not found. Please either run from a"
                    " directory containing a valid model_setup.py, or pass the -i"
                    " flag with a valid runscript path."
                ]
            )

    def check_for_key_variables(self):
        """
        Ensure that certain variables are defined within the model setup
        script before we import it. The intent is to prevent the model from
        progressing if variables that are required to run are not present,
        and goes some way to preventing unexpected code from running.
        """
        method_name = (
            "monarchs.core.load_model_setup.ModelSetup.check_for_key_variables"
        )
        # List of variables that are required to be present in order for
        # MONARCHS to accept the model setup script
        required_vars = [
            "row_amount",
            "col_amount",
            "vertical_points_firn",
            "vertical_points_lake",
            "vertical_points_lid",
            "num_days",
        ]

        with open(self.script_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=self.script_path)
            read_vars = set()
            # Walk through the AST and find the Assignments.
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        # for each target (i.e. variable), check
                        # if it is one of the required variables
                        # and if so add it to the list
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            if var_name in required_vars:
                                read_vars.add(var_name)

            missing_vars = set(required_vars) - read_vars
            if missing_vars:
                self.errors.extend(
                    [
                        f"{method_name}: The following required variables are"
                        f" missing from the model setup script {self.script_path}:"
                        f" {', '.join(missing_vars)}. Please check that "
                        f"{self.script_path} is a valid MONARCHS configuration "
                        f"script."
                    ]
                )

    def check_for_unexpected_imports(self):
        """
        We want to ensure that only certain imports are allowed in a model
        setup script. This further prevents unexpected code from being run
        when importing the script. This means that use of modules like
        "os" and "sys" are not supported in runscripts by default.

        If you are getting an error here due a module that you know is safe
        to execute, then add it to SAFE_IMPORTS below.
        """
        method_name = (
            "monarchs.core.load_model_setup.ModelSetup."
            "check_for_unexpected_imports"
        )

        with open(self.script_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=self.script_path)
        # Track whether we get any errors. If so, add an extra developer aid
        # message to tell people where they can add new safe imports.
        flag = False
        for node in ast.walk(tree):
            # Look at any nodes that are Imports.
            # If the name is not in SAFE_IMPORTS, return an error.
            if isinstance(node, ast.Import):
                for node_name in node.names:
                    # look at just the top-level module
                    if node_name.name.split(".")[0] not in SAFE_IMPORTS:
                        flag = True
                        self.errors.extend(
                            [
                                f"{method_name}: Unsafe import '{node_name.name}'"
                                f" found in model setup script {self.script_path}."
                            ]
                        )
            # Also look at ImportFrom nodes, and check whether the parent
            # is in SAFE_IMPORTS.
            elif isinstance(node, ast.ImportFrom):
                parent_name = node.module.split(".")[0] if node.module else ""
                if parent_name not in SAFE_IMPORTS:
                    self.errors.append(
                        f"{method_name}: Unsafe import from '{node.module}' found. "
                    )
                    flag = True
        if flag:
            self.errors.extend(
                [
                    f"Only the following imports are allowed: "
                    f"{', '.join(SAFE_IMPORTS)}."
                ]
            )
            self.errors.extend(
                [
                    f"If you are confident that your configuration script "
                    f"import(s) are safe, please add them to SAFE_IMPORTS "
                    f"in {MODULE_NAME}."
                ]
            )


def get_model_setup(model_setup_path):
    """
    Load in the model setup from the specified path,
    and return an instance of the ModelSetup class.
    """
    model_setup = ModelSetup(model_setup_path)
    return model_setup
