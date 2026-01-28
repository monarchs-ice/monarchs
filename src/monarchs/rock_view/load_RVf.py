"""
Load in the rock view fraction CSV to be used by the model grid.
"""

import numpy as np
import os
import os.path 

def load_RVf(RVf_filepath):
    """
    Load in the rock view fraction (RVf) CSV and convert to a NumPy array, perform logic checks.
    
    Parameters
    ----------
    RVf_filepath :  str
        Filepath for the rock view fraction CSV.
        The file specified must be a .csv, with all values being between 0 and 1.
        The grid must be the same size as the model grid specified in model_setup.py. Currently only square grids are supported.
    
    Returns
    -------
    RVf : np.array
        A NumPy array of rock view fraction values.
    """
    
    # Checking for the existence of the file and that it can be read.
    if os.path.isfile(RVf_filepath) is True:
        if os.access(RVf_filepath, os.R_OK):
            RVf = np.loadtxt(RVf_filepath, dtype=np.float64, delimiter=',')
        else:
            raise IOError(f"The file {RVf_filepath} is not readable.")
        
        # Checking that the RVf grid is the same shape as the model grid.
        if RVf.shape != (num_rows, num_cols): # TO DO: Need to include intiliase_iceshelf? Or grid[num_rows, num_cols]?
            raise ValueError("The RVf grid is the wrong shape!")
        
        
        # Checking that all values in the grid are between 0 and 1.
        if np.any((RVf<0) | (RVf>1)):
            raise ValueError("All grid values must be between 0 and 1!")
            
    return RVf
