# dprep/post_analysis_tools.py (or wherever this file resides)

import os
import re
import shutil
import sys
import traceback
from collections import defaultdict, namedtuple, Counter # Added Counter
from pathlib import Path
import math # Added math
from scipy.special import erf
from scipy.optimize import brentq, minimize_scalar

import pandas as pd
import numpy as np
import ase.db # Added ase.db
from ase.db import connect
from ase.data import chemical_symbols, atomic_numbers, atomic_names # Added atomic_numbers, atomic_names

import matplotlib
matplotlib.use('Agg') # Set backend *before* importing pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as colors # Added colors
import matplotlib.patches as patches # Added patches

from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
# --- AbacusTest Imports ---


# =============================================================================
# Constants
# =============================================================================

# --- Periodic Table Layout for Heatmaps (col, row) ---
# (Using 1-based indexing for columns, 0-based for rows, top row is 0)
ELEMENT_POSITIONS = {
    1: (1, 0), 2: (18, 0),  # H, He
    3: (1, 1), 4: (2, 1), 5: (13, 1), 6: (14, 1), 7: (15, 1), 8: (16, 1), 9: (17, 1), 10: (18, 1), # Li-Ne
    11: (1, 2), 12: (2, 2), 13: (13, 2), 14: (14, 2), 15: (15, 2), 16: (16, 2), 17: (17, 2), 18: (18, 2), # Na-Ar
    19: (1, 3), 20: (2, 3), 21: (3, 3), 22: (4, 3), 23: (5, 3), 24: (6, 3), 25: (7, 3), 26: (8, 3), 27: (9, 3), 28: (10, 3), 29: (11, 3), 30: (12, 3), 31: (13, 3), 32: (14, 3), 33: (15, 3), 34: (16, 3), 35: (17, 3), 36: (18, 3), # K-Kr
    37: (1, 4), 38: (2, 4), 39: (3, 4), 40: (4, 4), 41: (5, 4), 42: (6, 4), 43: (7, 4), 44: (8, 4), 45: (9, 4), 46: (10, 4), 47: (11, 4), 48: (12, 4), 49: (13, 4), 50: (14, 4), 51: (15, 4), 52: (16, 4), 53: (17, 4), 54: (18, 4), # Rb-Xe
    55: (1, 5), 56: (2, 5), # Cs, Ba
    57: (3, 7.5), 58: (4, 7.5), 59: (5, 7.5), 60: (6, 7.5), 61: (7, 7.5), 62: (8, 7.5), 63: (9, 7.5), 64: (10, 7.5), 65: (11, 7.5), 66: (12, 7.5), 67: (13, 7.5), 68: (14, 7.5), 69: (15, 7.5), 70: (16, 7.5), 71: (17, 7.5), # Lanthanides La-Lu (placed below)
    72: (4, 5), 73: (5, 5), 74: (6, 5), 75: (7, 5), 76: (8, 5), 77: (9, 5), 78: (10, 5), 79: (11, 5), 80: (12, 5), 81: (13, 5), 82: (14, 5), 83: (15, 5), 84: (16, 5), 85: (17, 5), 86: (18, 5), # Hf-Rn
    87: (1, 6), 88: (2, 6), # Fr, Ra
    89: (3, 8.5), 90: (4, 8.5), 91: (5, 8.5), 92: (6, 8.5), 93: (7, 8.5), 94: (8, 8.5), 95: (9, 8.5), 96: (10, 8.5), 97: (11, 8.5), 98: (12, 8.5), 99: (13, 8.5), 100: (14, 8.5), 101: (15, 8.5), 102: (16, 8.5), 103: (17, 8.5), # Actinides Ac-Lr (placed below)
    104: (4, 6), 105: (5, 6), 106: (6, 6), 107: (7, 6), 108: (8, 6), 109: (9, 6), 110: (10, 6), 111: (11, 6), 112: (12, 6), 113: (13, 6), 114: (14, 6), 115: (15, 6), 116: (16, 6), 117: (17, 6), 118: (18, 6) # Rf-Og
}

# Build symbol -> atomic_number map (using ase.data imported above)
SYMBOL_TO_Z = {symbol: i for i, symbol in enumerate(chemical_symbols)}


# =============================================================================
# Data Structures
# =============================================================================

# Define a structure to hold the error results
BandErrorMetrics = namedtuple("BandErrorMetrics",
                              ["mae",  # Mean Absolute Error
                               "rmse",  # Root Mean Squared Error
                               "max_ae"])  # Maximum Absolute Error

# Optional: Define a structure for extracted data if needed consistently
CutoffData = namedtuple("CutoffData", ["bands", "efermi", "kpt_lines"])


# =============================================================================
# Visualization Functions
# =============================================================================
def get_element_z_map():
    """
    Build a mapping from element symbols to atomic numbers.

    Returns:
        dict: A dictionary mapping element symbols to their atomic numbers
    """
    return {symbol: i for i, symbol in enumerate(chemical_symbols)}


def create_periodic_heatmap(element_values, title="Periodic Table Heatmap",
                            colormap='plasma', log_scale=True, output_file=None,
                            colorbar_label="Value", missing_value_text="-",
                            include_zero_values=True):
    """
    Create a heatmap visualization of the periodic table with custom values.

    Args:
        element_values (dict): Dictionary mapping element symbols to their values
        title (str): Title of the plot
        colormap (str): Matplotlib colormap name to use
        log_scale (bool): Whether to use logarithmic color scale
        output_file (str): Path to save the image, None to display instead
        colorbar_label (str): Label for the colorbar
        missing_value_text (str): Text to display for elements with no value
        include_zero_values (bool): Whether to color elements with value 0

    Returns:
        tuple: (fig, ax) - The matplotlib figure and axis objects
    """
    # Build the map from symbol to atomic number
    symbol_to_z = get_element_z_map()

    # Calculate min/max values for normalization
    present_elements = set(element_values.keys())
    max_value = 0
    min_value_nonzero = float('inf')

    # Prepare data for plotting
    plot_data = {}
    all_elements = set(chemical_symbols[1:119])  # Elements 1 to 118

    for symbol in all_elements:
        z = symbol_to_z.get(symbol)
        if not z or z > 118:  # Skip dummy elements or beyond Oganesson
            continue

        value = element_values.get(symbol, 0)
        pos = ELEMENT_POSITIONS.get(z)

        if pos:
            plot_data[symbol] = {'z': z, 'pos': pos, 'value': value}
            if value > 0:
                max_value = max(max_value, value)
                min_value_nonzero = min(min_value_nonzero, value)

    # Handle edge cases
    if max_value == 0:
        print("Warning: No positive values found in the dataset.")
        max_value = 1

    if min_value_nonzero == float('inf'):
        min_value_nonzero = 1

    # Choose colormap
    cmap = plt.get_cmap(colormap)

    # Prepare normalization
    if log_scale:
        # Better handling of log scale range
        if min_value_nonzero == max_value:
            # If all values are the same, create a small range around that value
            vmin_norm = max(0.9 * min_value_nonzero, 1e-10)  # Ensure positive for log scale
            vmax_norm = 1.1 * max_value
        else:
            # Calculate logarithmic range with better precision
            log_min = math.log10(max(min_value_nonzero, 1e-10))
            log_max = math.log10(max_value)

            # Ensure a reasonable range between min and max
            if log_max - log_min < 1:
                # If range is too small, expand it slightly
                log_min = max(log_min - 0.5, -10)  # Don't go below 1e-10
                log_max = log_max + 0.5

            vmin_norm = 10 ** math.floor(log_min)
            vmax_norm = 10 ** math.ceil(log_max)

        # Ensure vmin is positive for LogNorm
        vmin_norm = max(vmin_norm, 1e-10)
        norm = colors.LogNorm(vmin=vmin_norm, vmax=vmax_norm)
    else:
        # Linear scale with improved range handling
        if min_value_nonzero == max_value and max_value > 0:
            # If all values are the same, create a range
            vmin_norm = 0
            vmax_norm = max_value * 1.1  # Add 10% to max for better visualization
        else:
            vmin_norm = 0
            vmax_norm = max_value

        norm = colors.Normalize(vmin=vmin_norm, vmax=vmax_norm)

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_aspect('equal')

    # Plot element boxes
    for symbol, data in plot_data.items():
        value = data['value']
        col, row = data['pos']
        z = data['z']

        # Default colors
        color = 'white'  # Default for zero/missing values
        edge_color = 'gray'
        text_color = 'black'

        # Apply color based on value
        if value > 0 or (value == 0 and include_zero_values):
            try:
                # Safe color mapping to handle edge cases
                if log_scale and value <= 0:
                    # For log scale, set zero/negative values to minimum
                    color = cmap(0)  # Minimum of colormap
                else:
                    # For valid values, use proper normalization
                    mapped_value = max(value, vmin_norm if log_scale and value > 0 else 0)
                    color = cmap(norm(mapped_value))

                # Adjust text color for contrast
                r, g, b, _ = color
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = 'white' if luminance < 0.5 else 'black'
            except Exception as e:
                print(f"Warning: Color mapping error for {symbol} (value={value}): {e}")
                color = 'lightgray'

        # Draw element rectangle
        rect = patches.Rectangle(
            (col - 0.95, row - 0.95),  # Bottom-left corner
            0.9, 0.9,  # Width, height
            linewidth=0.5,
            edgecolor=edge_color,
            facecolor=color,
            alpha=0.8
        )
        ax.add_patch(rect)

        # Add element symbol
        ax.text(col - 0.5, row - 0.3, symbol,
                ha='center', va='center', fontsize=10, weight='bold', color=text_color)

        # Add element value
        if value > 0:
            # Format the value
            if value >= 1000 or (value < 0.1 and value > 0):
                value_str = f"{value:.1e}"  # Scientific notation for large/small values
            elif isinstance(value, int):
                value_str = f"{value}"
            else:
                value_str = f"{value:.2f}" if value < 10 else f"{value:.1f}"  # Better precision

            ax.text(col - 0.5, row - 0.7, value_str,
                    ha='center', va='center', fontsize=7, color=text_color)
        elif value == 0 and symbol in present_elements:
            ax.text(col - 0.5, row - 0.7, "0",
                    ha='center', va='center', fontsize=7, color=text_color)
        else:
            ax.text(col - 0.5, row - 0.7, missing_value_text,
                    ha='center', va='center', fontsize=7, color='gray')

    # Customize plot layout
    # Determine axis limits based on element positions
    min_col = min(pos[0] for pos in ELEMENT_POSITIONS.values())
    max_col = max(pos[0] for pos in ELEMENT_POSITIONS.values())
    min_row = min(pos[1] for pos in ELEMENT_POSITIONS.values())
    max_row = max(pos[1] for pos in ELEMENT_POSITIONS.values())

    ax.set_xlim(min_col - 1.5, max_col + 0.5)
    ax.set_ylim(max_row + 0.5, min_row - 1.5)  # Invert y-axis for typical table layout

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')  # Turn off the frame

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02, fraction=0.05, aspect=30)
    cbar.set_label(colorbar_label, size=12)

    # Set up ticks for log scale if needed
    if log_scale and max_value > 0:
        if vmin_norm < vmax_norm:
            # Create appropriate logarithmic ticks
            if vmax_norm / vmin_norm > 1000:
                # For wide ranges, use order-of-magnitude ticks
                log_min = math.floor(math.log10(vmin_norm))
                log_max = math.ceil(math.log10(vmax_norm))
                ticks = [10 ** i for i in range(log_min, log_max + 1)]
            else:
                # For narrower ranges, use more granular ticks
                log_min = math.log10(vmin_norm)
                log_max = math.log10(vmax_norm)
                # Determine appropriate tick spacing
                span = log_max - log_min
                if span <= 2:
                    # More granular ticks for small spans
                    positions = np.linspace(log_min, log_max, 5)
                    ticks = [10 ** p for p in positions]
                else:
                    # Whole powers of 10 for larger spans
                    ticks = [10 ** i for i in range(int(log_min), int(log_max) + 1)]

            cbar.set_ticks(ticks)
            # Format tick labels nicely (1, 10, 100, etc.)
            cbar.set_ticklabels([f"{t:g}" for t in ticks])

    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=600)
        print(f"Heatmap saved to {output_file}")

    return fig, ax


# =============================================================================
# Core Analysis Functions
# =============================================================================

def find_vbm_index(bands_shifted, tolerance=1e-4):
    """
    Finds the index of the Valence Band Maximum (VBM).
    (Code as provided in the original script)
    """
    num_bands = bands_shifted.shape[0]
    vbm_index = -1
    for i in range(num_bands):
        if np.max(bands_shifted[i, :]) <= tolerance:
            vbm_index = i
        else:
            break
    return vbm_index


def calculate_band_errors(bands1_shifted, bands2_shifted, band_indices):
    """
    Calculates error metrics between two sets of shifted band structures.
    (Code as provided in the original script - slightly adapted bounds check)
    """
    if not band_indices or len(band_indices) == 0:
        return BandErrorMetrics(np.nan, np.nan, np.nan)

    max_idx = max(band_indices) if band_indices else -1 # Handle empty list safely
    if max_idx >= bands1_shifted.shape[0] or max_idx >= bands2_shifted.shape[0]:
        print(f"Warning: Max band index {max_idx} requested is out of bounds "
              f"for bands shape {bands1_shifted.shape} or {bands2_shifted.shape}. Clamping indices.")
        band_indices = [idx for idx in band_indices if idx < bands1_shifted.shape[0] and idx < bands2_shifted.shape[0]]
        if not band_indices:
             print(f"Warning: No valid overlapping band indices found after clamping.")
             return BandErrorMetrics(np.nan, np.nan, np.nan)

    errors = []
    for i in band_indices:
        delta_e = bands1_shifted[i, :] - bands2_shifted[i, :]
        errors.extend(delta_e)

    if not errors:
        return BandErrorMetrics(np.nan, np.nan, np.nan)

    errors = np.array(errors)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    max_ae = np.max(np.abs(errors))

    return BandErrorMetrics(mae=mae, rmse=rmse, max_ae=max_ae)


def delta_band(band_energy1, band_energy2, n_elec, wk, smearing, smearing_sigma, efermi_shift = 0, return_all = False):
    '''
    Calculate the "distance" between two band structures.
    Parameters
    ----------
        band_energy1, band_energy2 : list
            Nested lists that contain the band energies.
            band_energy[ispin][ik][iband] specifies the band energy of a state.
        wk : list
            Weight of k-points.
        n_elec : float or tuple
            Total number of electrons used to determine the Fermi level.
            If it is a tuple, the first element is for band_energy1 and
            the second is for band_energy2.
        smearing : str
            Smearing method, can be 'gaussian' or 'fermi-dirac'
        smearing_sigma : float
            Smearing parameter.
        efermi_shift : float
            Energy shift of the Fermi level.
        return_all : bool
            If True, return a tuple (eta, efermi1, efermi2, omega) where
            eta is the distance, efermi1 and efermi2 are the Fermi levels,
            and omega is the optimized energy shift.
            If False, return only eta.
    '''
    # occupation function
    def f_occ(x, x0):
        if smearing == 'gaussian':
            return 0.5 * (1.0 - erf((x - x0) / smearing_sigma)) \
                if smearing_sigma > 0 else 0.5 * (1 - np.sign(x - x0))
        elif smearing == 'fermi-dirac':
            return 1.0 / (1.0 + np.exp((x - x0) / smearing_sigma)) \
                if smearing_sigma > 0 else 0.5 * (1 - np.sign(x - x0))
        else:
            raise ValueError('Unknown smearing method: %s'%smearing)
    def efermi(wk, be, n_elec):
        _nmax = np.sum(wk * f_occ(be, np.max(be)))
        _delta = (_nmax - n_elec) / n_elec
        if np.abs(_delta) < 1e-4 or np.abs(_delta * n_elec) <= 0.01: # 0.1% error: the case where all bands are occupied
            print(f"WARNING: all bands are occupied in band_energy1, error of this estimation: {_delta:.4%}")
            return np.max(be)
        else: # if error is too large, let it directly fail
            if _delta < 0:
                raise ValueError(f"""WARNING: maximum possible number of electrons in band structure not-enough:
{n_elec:.4f} vs. {_nmax:.4f} (nelec vs. nmax). This is always because of too small basis size and all
bands are occupied, otherwise please check your data.""")
            return brentq(lambda x: np.sum(wk * f_occ(be, x)) - n_elec, np.min(be), np.max(be))

    min_n_band = min(band_energy1.shape[0], band_energy2.shape[0])
    band_energy1 = band_energy1[:min_n_band]
    band_energy2 = band_energy2[:min_n_band]

    # convert to arrays for convenience
    be1 = np.expand_dims(band_energy1.T, 0)
    be2 = np.expand_dims(band_energy2.T, 0)
    # convert spinless weight to the one with spin
    nspin = len(be1)
    wk = [1] * band_energy1.shape[1]
    wk = np.array(wk).reshape(1, len(wk), 1) * (2 / nspin)
    wk = 2 * wk/np.sum(wk) # normalize the weight
    n_elec1, n_elec2 = n_elec if isinstance(n_elec, tuple) else (n_elec, n_elec)
    # if be1.shape != be2.shape:
    #     raise TypeError(f'Error: Inconsistent shape between two band structures: {be1.shape} vs {be2.shape}.')
    # assert be1.shape[1] == wk.shape[1]
    assert smearing_sigma >= 0 and n_elec1 > 0 and n_elec2 > 0
    # determine the Fermi levels for two band structures by root finding
    efermi1 = efermi(wk, be1, n_elec1)
    efermi2 = efermi(wk, be2, n_elec2)
    # geometrically averaged occupation (under shifted Fermi level)
    f_avg = np.sqrt(f_occ(be1, efermi1 + efermi_shift) * f_occ(be2, efermi2 + efermi_shift))
    res = minimize_scalar(lambda omega: np.sum(wk * f_avg * (be1 - be2 + omega)**2), \
            (-10, 10), method='brent')
    omega = res.x
    eta = np.sqrt(res.fun / np.sum(wk * f_avg))
    eta_max = np.max(np.abs(be1 - be2 + omega))
    # for ispin in range(nspin):
    #     for ik in range(len(be1[ispin])):
    #         delta = np.array(be1[ispin][ik]) - np.array(be2[ispin][ik]) + omega
    #         # zval = 19, natom = 4, nocc = 19*2 = 38
    #         if np.linalg.norm(delta, np.inf)*1e3 > 100:
    #             print(f_occ(be1, efermi1 + efermi_shift)[ispin][ik], delta)
    #             print(ispin, ik, np.linalg.norm(delta, np.inf))
    return (eta, eta_max) if not return_all else (eta, eta_max, efermi1, efermi2, omega)


def quantify_band_error(bands1, e_vbm_max1, bands2, e_vbm_max2, vbm_index,
                        num_valence_near_ef=5, num_conduction_near_ef=5, smearing='gaussian', smearing_sigma=0.002):
    """
    Quantifies the error between two band structures.
    (Code as provided in the original script - adapted for shape check/VBM warning)
    """
    # Ensure minimum number of bands and kpoints for comparison
    if bands1.shape[1] != bands2.shape[1]:
         print(f"Error: K-point number mismatch! {bands1.shape[1]} vs {bands2.shape[1]}")
         return None
    n_elec = (vbm_index + 1) * 2
    eta, eta_max = delta_band(
        band_energy1=bands1,
        band_energy2=bands2,
        n_elec=n_elec,
        wk=[],
        smearing='gaussian',
        smearing_sigma=0.002,
        efermi_shift=0,
        return_all=False
    )

    eta_10, eta_max_10 = delta_band(
        band_energy1=bands1,
        band_energy2=bands2,
        n_elec=n_elec,
        wk=[],
        smearing='gaussian',
        smearing_sigma=0.002,
        efermi_shift=10,
        return_all=False
    )

    num_bands, num_kpoints = bands1.shape

    bands1_shifted = bands1 - e_vbm_max1
    bands2_shifted = bands2 - e_vbm_max2

    cbm_index = vbm_index + 1

    valence_near_ef_indices = list(range(max(0, vbm_index - num_valence_near_ef + 1), vbm_index + 1)) if vbm_index >= 0 else []

    if cbm_index < num_bands:
        conduction_near_ef_indices = list(range(cbm_index, min(num_bands, cbm_index + num_conduction_near_ef)))
    else:
        print(f"Warning: CBM index ({cbm_index}) potentially out of bounds ({num_bands} common bands). No conduction bands near Ef considered.")
        conduction_near_ef_indices = []

    results = {}
    results['eta'] = {'mae': eta}
    results['eta_10'] = {'mae': eta_10}
    results['eta_max'] = {'mae': eta_max}
    results['eta_max_10'] = {'mae': eta_max_10}

    results['vbm_index_ref'] = vbm_index
    occupied_indices = list(range(vbm_index + 1))
    results['occupied'] = calculate_band_errors(bands1_shifted, bands2_shifted, occupied_indices)
    all_indexes = list(range(num_bands))
    results['all_band'] = calculate_band_errors(bands1_shifted, bands2_shifted, all_indexes)
    # Handle case where CBM index might be >= num_bands if VBM wasn't found properly or only occupied bands exist
    results['un_occupied'] = calculate_band_errors(bands1_shifted, bands2_shifted, all_indexes[cbm_index:]) if cbm_index < num_bands else BandErrorMetrics(np.nan, np.nan, np.nan)
    results['valence_near_ef'] = calculate_band_errors(bands1_shifted, bands2_shifted, valence_near_ef_indices)
    results['conduction_near_ef'] = calculate_band_errors(bands1_shifted, bands2_shifted, conduction_near_ef_indices)
    near_ef_indices = valence_near_ef_indices + conduction_near_ef_indices
    results['near_ef'] = calculate_band_errors(bands1_shifted, bands2_shifted, near_ef_indices)
    occupied_half_un_occupied_indices = list(range(int(len(occupied_indices)*1.5)))
    results['n_occupied_and_0.5n_un_occupied'] = calculate_band_errors(bands1_shifted, bands2_shifted, occupied_half_un_occupied_indices)
    occupied_1_un_occupied_indices = list(range(len(occupied_indices)+1))
    results['n_occupied_and_1_un_occupied'] = calculate_band_errors(bands1_shifted, bands2_shifted, occupied_1_un_occupied_indices)
    occupied_3_un_occupied_indices = list(range(len(occupied_indices)+3))
    results['n_occupied_and_3_un_occupied'] = calculate_band_errors(bands1_shifted, bands2_shifted, occupied_3_un_occupied_indices)
    occupied_5_un_occupied_indices = list(range(len(occupied_indices)+5))
    results['n_occupied_and_5_un_occupied'] = calculate_band_errors(bands1_shifted, bands2_shifted, occupied_5_un_occupied_indices)

    return results


# =============================================================================
# File Handling / Data Extraction Helpers (Example: ABACUS)
# =============================================================================
def simple_get_n_occupied(log_nscf_path):
    occupied_bands = None
    pattern = re.compile(r"occupied bands\s*=\s*(\d+)")
    with open(log_nscf_path, 'r', encoding='utf-8') as f:  # 使用 utf-8 编码打开
        for line in f:
            match = pattern.search(line)
            if match:
                occupied_bands = int(match.group(1))
    return occupied_bands


def _extract_band_data(job_dir_path: Path, target_band_file: str = "BANDS_1.dat",
                             target_log_nscf: str = "running_nscf.log",
                             inputfile_name: str = "INPUT.nscf",
                             kptfile_name: str = "KPT.nscf"):
    from abacustest.lib_model.model_012_band import PostBand, ReadInput, ReadKpt
    """
    Internal helper: Extracts band data for a single ABACUS job directory.
    Finds standard INPUT, KPT, OUT.*, band and log files.

    Args:
        job_dir_path (Path): Path to the calculation directory.
        target_band_file (str): Name of the band data file.
        target_log_nscf (str): Name of the NSCF log file.
        inputfile_name (str): Name of the input file.
        kptfile_name (str): Name of the kpoint file.

    Returns:
        tuple: (bands_ev, kpt_lines, efermi)
               bands_ev are in eV. Returns None for data points on error.
    """
    job_dir = str(job_dir_path)
    # Use PostBand/ReadInput/ReadKpt from abacustest
    post_instance = PostBand([job_dir])  # May not need path if methods are static

    input_param = ReadInput(str(job_dir_path / inputfile_name))
    kpt_reader_instance = ReadKpt(str(job_dir_path / kptfile_name))  # Try instantiation
    # Check if result is directly available or needs method call (adapt as needed)
    if hasattr(kpt_reader_instance, 'kpt_data'):  # Example attribute access
        kpt_data = kpt_reader_instance.kpt_data
    elif callable(kpt_reader_instance):  # Example if instance is callable
        kpt_data = kpt_reader_instance()
    else:  # Default assumption based on original code context
        kpt_data = kpt_reader_instance

    kpt_lines = kpt_data[0]

    # Find band and log files within OUT.* directory
    suffix = input_param.get("suffix", "ABACUS")
    band_file = job_dir_path / f"OUT.{suffix}" / target_band_file
    log_file_nscf = job_dir_path / f"OUT.{suffix}" / target_log_nscf

    n_occupied_bands = simple_get_n_occupied(log_file_nscf)
    bands_ev = PostBand.get_band(str(band_file))  # assuming get_band returns eV
    e_vbm_max = float(max(bands_ev[n_occupied_bands-1]))
    efermi_ev = post_instance.get_efermi(str(log_file_nscf))  # Assuming instance method from earlier

    # Return data: Bands (eV), KPT lines, Efermi (eV), No error
    return bands_ev, np.array(kpt_lines, dtype=object), efermi_ev, n_occupied_bands, e_vbm_max


# =============================================================================
# Other Utility Functions (Previously Provided)
# =============================================================================
# (copy_failed_folders, find_copy_rename_recursive, process_band_data,
#  plot_band_comparisons, run_band_comparison_workflow)
# These functions might need adjustments if they rely on specific details
# changed above, but their core logic remains the same based on the
# original provided code. Keep them here if they are part of your toolset.


def merge_svgs_to_pdf(svg_files, output_pdf):
    # Total number of SVG files
    num_files = len(svg_files)

    # Estimate grid dimensions (rows and columns) to be as square as possible.
    # Here we use rows = ceil(sqrt(n)) and cols = ceil(n / rows)
    rows = math.ceil(math.sqrt(num_files))
    cols = math.ceil(num_files / rows)

    # Use the first SVG to estimate the dimensions of a single image
    sample_drawing = svg2rlg(svg_files[0])
    svg_width = sample_drawing.width
    svg_height = sample_drawing.height

    # Calculate the overall PDF page dimensions (here we keep the original SVG sizes)
    page_width = cols * svg_width
    page_height = rows * svg_height

    # Create a PDF canvas with the computed page size
    c = canvas.Canvas(output_pdf, pagesize=(page_width, page_height))

    # Iterate through all SVG files and draw each on the PDF
    for index, svg_file in enumerate(svg_files):
        drawing = svg2rlg(svg_file)

        # Calculate the column and row index for the current image.
        # PDF origin is at the bottom left.
        col_index = index % cols
        # For row, we start from the top, so we calculate rows - 1 - (index // cols)
        row_index = rows - 1 - (index // cols)

        # Calculate the lower-left coordinates for drawing the image
        x = col_index * svg_width
        y = row_index * svg_height

        # Optionally scale the drawing to fit exactly into the cell if needed
        scale = min(svg_width / drawing.width, svg_height / drawing.height)
        drawing.width *= scale
        drawing.height *= scale
        drawing.scale(scale, scale)

        # Draw the drawing object on the canvas at the specified position
        renderPDF.draw(drawing, c, x, y)

    # Save the generated PDF file
    c.save()
    print(f"PDF saved to: {output_pdf}")


def copy_failed_folders(source_dir, target_folder_name, check_file_name, dump_dir):
    """
    Recursively search through `source_dir` for folders named `target_folder_name`.
    If such a folder is found and does NOT contain the file `check_file_name`,
    copy its parent directory (considered the subtask ID) to `dump_dir`.

    Parameters:
        source_dir (str or Path): The root directory to start the search from.
        target_folder_name (str): The name of the folder to look for.
        check_file_name (str): The file whose presence is checked inside the target folder.
        dump_dir (str or Path): The destination directory to copy the parent folders into.
    """
    source_dir = Path(source_dir)
    dump_dir = Path(dump_dir)
    if not source_dir.is_dir():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    dump_dir.mkdir(parents=True, exist_ok=True)

    found_count = 0
    copied_count = 0

    print(f"Searching in: {source_dir}")
    print(f"Looking for folders named: '{target_folder_name}'")
    print(f"Checking for absence of file: '{check_file_name}' inside those folders")
    print(f"Copying the parent folder (subtask ID) to: {dump_dir}")
    print("-" * 40)

    for root, dirs, files in os.walk(source_dir):
        # If the current directory name matches the target
        if os.path.basename(root) == target_folder_name:
            found_count += 1
            folder_path = Path(root)
            parent_folder = folder_path.parent  # The subtask folder (one level up)
            task_id = parent_folder.name

            # Check if the file is missing
            if check_file_name not in os.listdir(folder_path):
                print(f"  · Missing file '{check_file_name}' in: {folder_path}")
                destination = dump_dir / task_id

                if destination.exists():
                    print(f"    - Skipped: destination '{destination}' already exists.")
                else:
                    try:
                        shutil.copytree(parent_folder, destination)
                        copied_count += 1
                        print(f"    - Successfully copied to: '{destination}'")
                    except Exception as e:
                        print(f"    - Failed to copy: {e}")

    print("\n" + "-" * 40)
    print(f"Found {found_count} folders named '{target_folder_name}'.")
    print(f"Copied {copied_count} parent folders to '{dump_dir}'.")
    if copied_count == 0:
        print("No missing files found, nothing was copied.")


def find_copy_rename_recursive(source_dir, filename_to_find, dest_dir, id_prefix):
    """(Code as provided in the original script)"""
    source_path = Path(source_dir); dest_path = Path(dest_dir)
    if not source_path.is_dir(): print(f"Error: Source directory '{source_dir}' not found."); return
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"Searching in: {source_path}"); print(f"Looking for:  {filename_to_find}")
    print(f"Copying to:   {dest_path}"); print(f"ID prefix:    '{id_prefix}'"); print("-" * 20)
    found_count = 0; copied_count = 0
    for found_file in source_path.rglob(filename_to_find):
        found_count += 1; # print(f"Found: {found_file}")
        id_name = None
        for parent in found_file.parents:
            if parent.name.startswith(id_prefix): id_name = parent.name; break
            if parent == source_path: break
        if id_name:
            original_extension = found_file.suffix; new_filename = f"{id_name}{original_extension}"
            destination_file_path = dest_path / new_filename
            try:
                # print(f"  -> Copying to: {destination_file_path}")
                shutil.copy2(found_file, destination_file_path); copied_count += 1
            except Exception as e: print(f"  -> Error copying {found_file} to {destination_file_path}: {e}")
        # else: print(f"  -> Warning: Could not find parent directory starting with '{id_prefix}' for {found_file}. Skipping.")
    print("-" * 20); print(f"Search complete. Found {found_count} file(s). Copied and renamed {copied_count} file(s).")


def count_elements_in_db(db_file):
    """
    Count the number of structures in which each element appears.

    Args:
        db_file (str): Path to the ASE database file

    Returns:
        dict: Dictionary mapping element symbols to their occurrence counts
    """
    print(f"Connecting to database: {db_file}")
    element_counts = Counter()
    total_structures = 0

    try:
        with ase.db.connect(db_file) as db:
            total_structures = len(db)
            print(f"Found {total_structures} structures. Counting elements...")

            for i, row in enumerate(db.select()):
                atoms = row.toatoms()
                # Count unique elements in each structure
                unique_elements = set(atoms.get_chemical_symbols())
                element_counts.update(unique_elements)

                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{total_structures} structures...")

    except Exception as e:
        print(f"Error accessing database: {e}")
        return {}

    print("Finished counting.")
    return element_counts


def get_atom_counts_from_elementary_db(db_file):
    """
    Extract the number of atoms in each elementary substance from the database.

    Args:
        db_file (str): Path to the database containing elementary substances

    Returns:
        dict: Dictionary mapping element symbols to their atom counts
    """
    print(f"Connecting to elementary substances database: {db_file}")
    element_atom_counts = {}

    try:
        with connect(db_file) as db:
            total_structures = len(db)
            print(f"Found {total_structures} structures in the database.")

            for i, row in enumerate(db.select()):
                try:
                    atoms = row.toatoms()
                    symbols = atoms.get_chemical_symbols()

                    # Verify this is an elementary substance
                    unique_elements = set(symbols)
                    if len(unique_elements) != 1:
                        print(f"Warning: Structure {row.id} is not an elementary substance. Skipping.")
                        continue

                    element_symbol = list(unique_elements)[0]
                    atom_count = len(atoms)

                    # Only record if we don't have this element yet or
                    # if this structure has fewer atoms than previously recorded
                    if element_symbol not in element_atom_counts or atom_count < element_atom_counts[element_symbol]:
                        element_atom_counts[element_symbol] = atom_count

                except Exception as e:
                    print(f"Error processing structure {row.id}: {e}")

                if (i + 1) % 20 == 0 or i == total_structures - 1:
                    print(f"Processed {i + 1}/{total_structures} structures...")

    except Exception as e:
        print(f"Error accessing database: {e}")
        return {}

    print(f"Found atom counts for {len(element_atom_counts)} elements.")

    # Optionally, print the results
    for element, count in sorted(element_atom_counts.items()):
        print(f"{element}: {count} atoms")

    return element_atom_counts


# --- process_band_data and plot_band_comparisons use _extract_band_data ---
def extract_symbols_from_kpoints(kpoints, label_dict, threshold=1e-5):
    """
    Extract symbol indices and symbol names from kpoints array by comparing with label_dict.

    Args:
        kpoints (list): List of k-point coordinates in direct coordinates
        label_dict (dict): Dictionary mapping symbol names to positions
        threshold (float): Distance threshold to consider a kpoint matching a label position

    Returns:
        tuple: (symbol_index, symbols) where:
            - symbol_index is a list of indices where labeled points are found
            - symbols is a list of the corresponding symbol labels
    """
    import numpy as np

    symbol_index = []
    symbols = []
    last_symbol = None

    # Process each kpoint
    for i, kpt in enumerate(kpoints):
        for label, pos in label_dict.items():
            # Check if the kpoint is close to a labeled position
            if np.allclose(kpt, pos, atol=threshold):
                # Convert '\Gamma' to 'G'
                symbol = 'G' if label == '\\Gamma' else label

                # Add to the lists (only if different from the previous symbol)
                if symbol != last_symbol:
                    symbol_index.append(i)
                    symbols.append(symbol)
                    last_symbol = symbol
                break

    return symbol_index, symbols


def save_direct_kpoints(kpoints):
    """
    Saves k-point data to a custom text file format and band data to a .npy file.

    Args:
        kpoints (list): A list of k-point coordinates, e.g., [[0.0, 0.0, 0.0], ...].
                        Assumed to be in Direct coordinates.
    """
    # Ensure output directory exists
    num_kpoints = len(kpoints)
    weight = 1.0 / num_kpoints

    with open('old_kpoints', 'w') as f:
        # Write header information
        f.write("K_POINTS\n")
        f.write(f"{num_kpoints} //total number of k-point\n")
        f.write("Direct //'Direct' coordinate\n")

        # Write k-point coordinates and weights
        for kpt in kpoints:
            # Format: kx ky kz weight (using fixed precision for neatness)
            f.write(f"{kpt[0]:.8f} {kpt[1]:.8f} {kpt[2]:.8f} {weight:.8f}\n")


def process_band_data(
        workspace_root: str,
        plot_data_dir: str,
        job_types: list = ['pw', 'lcao'],
        id_prefix: str = 'id_',
        target_folder: str = 'OUT.ABACUS',
        target_band_file: str = 'BANDS_1.dat',
        force_reprocess: bool = False
):
    """(Code as provided in the original script, uses _extract_band_data)"""
    workspace_path = Path(workspace_root); plot_data_path = Path(plot_data_dir)
    plot_data_path.mkdir(parents=True, exist_ok=True)
    print("\n--- Starting Data Processing (process_band_data) ---"); print(f"Workspace: {workspace_path}"); print(f"Job Types: {job_types}"); print(f"ID Prefix: '{id_prefix}'"); print(f"Target:    '{target_folder}/{target_band_file}'"); print(f"Cache Dir: {plot_data_path}"); print(f"Force Reprocess: {force_reprocess}"); print("-" * 30)
    found_jobs = defaultdict(lambda: {job_type: None for job_type in job_types}); raw_band_files_count = 0
    for job_type in job_types:
        search_dir = workspace_path / job_type
        if not search_dir.is_dir(): print(f"Warning: Search directory not found: {search_dir}"); continue
        for band_file_path in search_dir.rglob(f"**/{target_folder}/{target_band_file}"):
            raw_band_files_count += 1; out_folder = band_file_path.parent; id_dir = out_folder.parent
            if id_dir.name.startswith(id_prefix):
                id_name = id_dir.name
                if job_type in found_jobs[id_name]: found_jobs[id_name][job_type] = id_dir
    print(f"Found {raw_band_files_count} raw band files across specified job types.")
    processed_ids = []; missing_pair_ids = []; error_ids = defaultdict(list); skipped_ids = []
    sorted_ids = sorted(found_jobs.keys())
    for id_name in sorted_ids:
        job_paths = found_jobs[id_name]
        if not all(job_paths[jt] for jt in job_types): missing_pair_ids.append(id_name); continue
        output_npz_path = plot_data_path / f"{id_name}.npz"
        if output_npz_path.exists() and not force_reprocess: skipped_ids.append(id_name); continue
        print(f"  Processing {id_name}...")
        data_to_save = {}
        kpt_lines_ref_storage = None # To store kpt_lines from first job type
        for i, job_type in enumerate(job_types):
            job_dir = job_paths[job_type]
            # Use the internal helper _extract_band_data which uses abacustest
            bands_ev, kpt_lines, efermi, n_occupied_bands, e_vbm_max = _extract_band_data(job_dir, target_band_file)
            data_to_save[f'bands_{job_type}'] = bands_ev
            data_to_save[f'efermi_{job_type}'] = efermi
            data_to_save[f'e_vbm_max_{job_type}'] = e_vbm_max
            data_to_save[f'n_occupied_bands_{job_type}'] = n_occupied_bands
            if i == 0: kpt_lines_ref_storage = kpt_lines # Store kpt from first type
        data_to_save['kpt_lines_ref'] = np.array(kpt_lines_ref_storage, dtype=object)  # Save kpt_lines
        np.savez_compressed(output_npz_path, **data_to_save)
        processed_ids.append(id_name)

    print("-" * 30); print("Processing Summary:"); print(f"- IDs processed and saved: {len(processed_ids)}"); print(f"- IDs skipped (already processed): {len(skipped_ids)}"); print(f"- IDs missing a required job type: {len(missing_pair_ids)}"); print(f"- IDs with errors during data extraction/saving: {len(error_ids)}")
    if error_ids:
        print("  Error Details:")
        for id_name, errors in error_ids.items(): print(f"    - {id_name}: {'; '.join(errors)}")
    print("-" * 30)
    return {'processed': processed_ids, 'missing_pair': missing_pair_ids, 'errors': list(error_ids.keys()), 'skipped': skipped_ids}


def plot_band_comparisons(
        plot_data_dir: str,
        pics_dir: str,
        job_types: list = ['pw', 'lcao'],
        id_prefix: str = 'id_',
        ase_db_path: str = None,
        plot_styles: dict = {'pw': {'color': 'blue', 'linestyle': '-', 'label': 'PW'},
                             'lcao': {'color': 'red', 'linestyle': '--', 'label': 'LCAO'}},
        plot_ylim: list = [-5, 5],
        plot_filename_suffix: str = '_compare.png',
        force_replot: bool = False
):
    from abacustest.lib_model.model_012_band import PostBand, ReadInput, ReadKpt

    plot_data_path = Path(plot_data_dir)
    pics_path = Path(pics_dir)
    pics_path.mkdir(parents=True, exist_ok=True)
    if not plot_data_path.is_dir(): print(f"Error: Plot data directory not found: {plot_data_path}"); return

    # Initialize a flag for ASE DB availability
    db_available = False

    print("\n--- Starting Plot Generation (plot_band_comparisons) ---")
    print(f"Reading data from: {plot_data_path}")
    print(f"Saving plots to:   {pics_path}")
    print(f"Plot Y limits:     {plot_ylim}")

    # Check ASE DB availability
    if ase_db_path:
        ase_db_file = Path(ase_db_path)
        if not ase_db_file.is_file():
            print(f"Warning: ASE DB file not found at '{ase_db_path}'. Cannot fetch formulas.")
        else:
            try:
                # Just test if connection works
                with connect(ase_db_file) as _:
                    db_available = True
                    print(f"ASE DB available: {ase_db_file.name}")
            except Exception as e:
                print(f"Warning: Failed to connect to ASE DB '{ase_db_path}': {e}")

    print(f"ASE DB for titles: {'Yes (' + Path(ase_db_path).name + ')' if db_available else 'No'}")
    print(f"Force Replot:      {force_replot}")
    print("-" * 30)

    plot_count = 0
    plot_errors = 0
    skipped_count = 0
    npz_files = sorted(list(plot_data_path.glob(f"{id_prefix}*.npz")))
    if not npz_files: print("No processed .npz files matching the ID prefix found to plot."); print("-" * 30); return

    for npz_file in npz_files:
        id_name = npz_file.stem
        print(f"  Plotting {id_name}...")
        try:
            data = np.load(npz_file, allow_pickle=True)
            # Now we need both Fermi level and VBM max for each job type
            required_keys = ['kpt_lines_ref'] + \
                            [f'bands_{jt}' for jt in job_types] + \
                            [f'e_vbm_max_{jt}' for jt in job_types] + \
                            [f'efermi_{jt}' for jt in job_types]

            if not all(key in data for key in required_keys):
                print(f"    Error: Missing required data keys in {npz_file.name}. Skipping.")
                plot_errors += 1
                continue

            kpt_lines_ref = data['kpt_lines_ref']
            # Use data from the first job type as reference shape for rearrange_plotdata
            bands_ref = data[f'bands_{job_types[0]}']  # Bands already in eV if processed correctly
            e_vbm_max_ref = data[f'e_vbm_max_{job_types[0]}']
            n_occupied_bands = data[f'n_occupied_bands_{job_types[0]}']
            # Ensure kpt_lines_ref is usable
            if kpt_lines_ref is None or kpt_lines_ref.size == 0 or (
                    kpt_lines_ref.ndim > 0 and kpt_lines_ref[0] is None):
                print(f"    Error: Invalid kpt_lines_ref in {npz_file.name}. Skipping plot.")
                plot_errors += 1
                continue

            # Call PostBand.rearrange_plotdata - assumes it handles None kpt_lines gracefully if it occurs
            band_idx, symbol_index, symbols = PostBand.rearrange_plotdata(bands_ref, kpt_lines_ref)  # Pass bands for shape info

            plot_title_id = id_name
            if db_available and ase_db_path:
                try:
                    numeric_id_str = id_name.replace(id_prefix, "")
                    if numeric_id_str.isdigit():
                        ase_id_one_based = int(numeric_id_str) + 1
                        # Use 'with connect()' pattern here
                        with connect(ase_db_path) as db_conn:
                            row = db_conn.get(id=ase_id_one_based)
                            if row and hasattr(row, 'formula'):
                                plot_title_id = row.formula
                except Exception as db_err:
                    print(f"    Warning: Error querying ASE DB for {id_name}: {db_err}")

            plot_filename = pics_path / (plot_title_id + plot_filename_suffix)
            if plot_filename.exists() and not force_replot:
                skipped_count += 1
                continue

            fig, ax = plt.subplots(figsize=(6, 5))
            fermi_ylim_list = []
            cbm_max_ylim_list = []
            # Plot all job types
            plotted_labels = set()  # To avoid duplicate legend entries
            for job_type in job_types:
                bands = data[f'bands_{job_type}']  # Should be in eV already
                e_vbm_max = data[f'e_vbm_max_{job_type}']  # Use VBM as reference
                efermi = data[f'efermi_{job_type}']  # Get Fermi level for horizontal line

                style = plot_styles.get(job_type, {})
                label_base = style.get('label', job_type)
                color = style.get('color', 'black')
                ls = style.get('linestyle', '-')

                if bands.shape[1] != bands_ref.shape[1]:
                    print(f"    Warning: K-point mismatch for {job_type} in {id_name}. Skipping {job_type} plot.")
                    continue

                # Plot bands relative to VBM instead of Fermi level
                bands_shifted = bands - e_vbm_max
                cbm_max = float(max(bands_shifted[n_occupied_bands]))
                cbm_max_ylim_list.append(cbm_max + 2)
                for band_num, iband in enumerate(bands_shifted):
                    label_to_use = label_base if (label_base not in plotted_labels and band_num == 0) else ""
                    if label_to_use: plotted_labels.add(label_base)
                    try:
                        for i, idx in enumerate(band_idx):  # Use band_idx from reference
                            if idx[2] < len(iband) and idx[3] <= len(iband):  # Basic bounds check
                                ax.plot(range(idx[0], idx[1]), iband[idx[2]:idx[3]], linestyle=ls, color=color,
                                        linewidth=1.0, alpha=0.8, label=label_to_use if i == 0 else "")
                            else:  # Handle potential index mismatch more gracefully
                                # print(f"    Debug: Index mismatch plotting band {band_num} for {job_type} idx={idx} band_len={len(iband)}")
                                pass  # Skip segment if indices out of bound
                    except IndexError as idx_err:
                        print(
                            f"    Warning: IndexError plotting band {band_num} for {job_type}. Idx: {idx}, Band len: {len(iband)}. Error: {idx_err}")
                    except Exception as plot_err:
                        print(f"    Warning: Unexpected error plotting band {band_num} for {job_type}: {plot_err}")

                # Add Fermi level horizontal line relative to VBM
                fermi_relative_to_vbm = efermi - e_vbm_max
                fermi_ylim_list.append(fermi_relative_to_vbm - 2)

                if band_idx:
                    x_min, x_max = 0, band_idx[-1][1]
                    ax.plot([x_min, x_max], [fermi_relative_to_vbm, fermi_relative_to_vbm],
                            linestyle=':', color=color, linewidth=1.2, alpha=0.7)

            if band_idx: ax.set_xlim(0, band_idx[-1][1])  # Use max k index from rearrange
            ax.set_ylim(min(plot_ylim[0], min(fermi_ylim_list)), max(plot_ylim[1], max(cbm_max_ylim_list)))
            if symbols is not None and symbol_index is not None and len(symbol_index) == len(symbols):
                ax.set_xticks(symbol_index)
                ax.set_xticklabels(symbols)
                for index in symbol_index[1:-1]: ax.axvline(x=index, color='gray', linestyle=':', linewidth=0.6)
            ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel("K points")
            ax.set_ylabel("Energy (E - E$_{VBM_{MAX}}$, eV)")  # Updated y-label as requested
            title_note = "\n(Dash line: E$_{VBM_{MAX}}$; Dot line: E$_{fermi}$)"
            ax.set_title(f"Band Comparison: {plot_title_id}"+title_note)
            ax.legend()
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300)
            plt.close(fig)
            plot_count += 1
        except Exception as e:
            print(f"    Error plotting {id_name}: {type(e).__name__} - {e}")
            plot_errors += 1
            # traceback.print_exc() # Uncomment for detailed debug
            plt.close('all')  # Close any potentially open figures from the error

    print("-" * 30)
    print("Plotting Summary:")
    print(f"- Plots generated: {plot_count}")
    print(f"- Plots skipped (already exist): {skipped_count}")
    print(f"- Errors during plotting: {plot_errors}")
    print("-" * 30)


def run_band_comparison_workflow(
        workspace_root: str,
        base_output_dir: str,
        ase_db_path: str = None,
        job_types: list = ['pw', 'lcao'],
        id_prefix: str = 'db_seq_id_',
        target_folder: str = 'OUT.ABACUS',
        target_band_file: str = 'BANDS_1.dat',
        plot_styles: dict = {'pw': {'color': 'blue', 'linestyle': '-', 'label': 'PW'},
                             'lcao': {'color': 'red', 'linestyle': '--', 'label': 'LCAO'}},
        plot_ylim: list = [-5, 5],
        plot_filename_suffix: str = '.png',
        force_reprocess: bool = False,
        force_replot: bool = False
):
    """(Code as provided in the original script)"""
    workspace_path = Path(workspace_root); base_output_path = Path(base_output_dir)
    if not base_output_path.is_absolute(): base_output_path = os.path.abspath(base_output_path)
    plot_data_dir = os.path.join(base_output_path, 'plot_data')
    pics_dir = os.path.join(base_output_path, 'pics')
    print("=" * 40); print("Starting Band Comparison Workflow"); print("=" * 40)
    # Step 1: Process data
    process_results = process_band_data(workspace_root=str(workspace_path), plot_data_dir=str(plot_data_dir), job_types=job_types, id_prefix=id_prefix, target_folder=target_folder, target_band_file=target_band_file, force_reprocess=force_reprocess)
    # Step 2: Generate plots
    plot_band_comparisons(plot_data_dir=str(plot_data_dir), pics_dir=str(pics_dir), job_types=job_types, id_prefix=id_prefix, ase_db_path=ase_db_path, plot_styles=plot_styles, plot_ylim=plot_ylim, plot_filename_suffix=plot_filename_suffix, force_replot=force_replot)
    print("\nWorkflow Complete."); print("=" * 40)
    return plot_data_dir, pics_dir


def round_sig(x, sig=3):
    if x == 0:
        return 0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)


def find_failed_jobs_directories(start_path):
    """
    Find all directories named 'failed_jobs' and check if they have content.
    If they have content, print their absolute paths.
    """
    # Ensure the starting path exists
    if not os.path.exists(start_path):
        print(f"Error: Path '{start_path}' does not exist")
        return

    # Counters
    empty_count = 0
    non_empty_count = 0

    print(f"Starting to search for 'failed_jobs' directories under '{start_path}'...")

    # Walk through the directory tree
    for root, dirs, files in os.walk(start_path):
        # Check if the current directory contains a subdirectory named 'failed_jobs'
        if 'failed_jobs' in dirs:
            failed_jobs_path = os.path.join(root, 'failed_jobs')

            # Check if this directory has content
            has_content = False

            # Check for files or subdirectories
            failed_jobs_files = os.listdir(failed_jobs_path)
            if failed_jobs_files:
                has_content = True
                non_empty_count += 1
                print(f"Non-empty failed_jobs directory: {failed_jobs_path}")
            else:
                empty_count += 1

    print(f"\nSearch completed. Found {empty_count + non_empty_count} 'failed_jobs' directories:")
    print(f"- Directories with content: {non_empty_count}")
    print(f"- Empty directories: {empty_count}")


def rescue_jobs(log_file_path="dpdispatcher.log", output_dir="./rescue"):
    """
    Extract job IDs from dpdispatcher.log and download them for rescue

    Args:
        log_file_path: Path to the dpdispatcher log file
        output_dir: Directory to save rescued jobs
    """
    # Extract job IDs using regex pattern
    job_ids = []
    pattern = r'job_id is (\d+):'

    with open(log_file_path, 'r') as file:
        for line in file:
            if 'job_id is' in line:
                match = re.search(pattern, line)
                if match:
                    job_id = match.group(1)
                    job_ids.append(job_id)

    print(f"Found {len(job_ids)} job IDs: {', '.join(job_ids) if job_ids else 'None'}")

    # Create rescue directory and download jobs
    if job_ids:
        os.makedirs(output_dir, exist_ok=True)
        current_dir = os.getcwd()
        os.chdir(output_dir)

        # Download jobs using lbg command
        download_cmd = 'lbg job download ' + ' '.join(job_ids)
        print(f"Executing: {download_cmd}")
        os.system(download_cmd)

        # Return to original directory
        os.chdir(current_dir)
        print(f"Jobs downloaded to {os.path.abspath(output_dir)}")

    return job_ids


# Helper function for ASE DB lookup (using 'with', no close needed)
def get_element_info(ase_db_path: str, id_name: str, id_prefix: str, symbol_to_z_global: dict):
    """
    Fetches formula, symbol, and Z number from ASE DB or guesses.
    Handles DB connection safely using 'with' statement.

    Args:
        ase_db_path (str): Path to the ASE database file. Can be None.
        id_name (str): The identifier name (e.g., 'db_seq_id_0').
        id_prefix (str): The prefix used in the identifier (e.g., 'db_seq_id_').
        symbol_to_z_global (dict): Precomputed map from element symbol to atomic number.

    Returns:
        tuple: (plot_title_id, formula, element_symbol, atomic_number)
    """
    formula = "Unknown"; atomic_number = 999; element_symbol = "X"; plot_title_id = id_name
    if ase_db_path and Path(ase_db_path).is_file():
        try:
            # 'with' statement ensures the database connection is closed automatically
            with connect(ase_db_path) as db:
                numeric_id_str = id_name.replace(id_prefix, "")
                if numeric_id_str.isdigit():
                    try:
                       # Assuming id_prefix relates to 1-based DB ID
                       ase_db_id_key = int(numeric_id_str) + 1
                       row = db.get(id=ase_db_id_key)
                    except (ValueError, TypeError, KeyError): # Catch common DB key errors
                       try: row = db.get(id_name) # Fallback: try id_name as key
                       except: row = None # Give up if both fail
                    except Exception as get_err: # Catch other potential db.get errors
                       print(f"    Warning: DB get error for key '{ase_db_id_key}' or '{id_name}': {get_err}")
                       row = None

                    if row and hasattr(row, 'formula'):
                        formula = row.formula; plot_title_id = formula
                        match = re.match(r"([A-Z][a-z]?)", formula)
                        if match:
                            element_symbol = match.group(1)
                            atomic_number = symbol_to_z_global.get(element_symbol, 999)
        except Exception as db_err:
             print(f"    Warning: DB access error '{ase_db_path}': {db_err}")
    # Fallback guess if DB unavailable or lookup failed
    if formula == "Unknown":
        potential_symbol = id_name.replace(id_prefix, '')
        match_symbol_only = re.match(r"([A-Z][a-z]?)", potential_symbol)
        if match_symbol_only:
             guessed_symbol = match_symbol_only.group(1)
             if symbol_to_z_global.get(guessed_symbol):
                 element_symbol = guessed_symbol
                 atomic_number = symbol_to_z_global.get(element_symbol, 999)
                 formula = element_symbol; plot_title_id = formula
    return plot_title_id, formula, element_symbol, atomic_number


# Function to consolidate results for a single job type pair
def consolidate_results_for_job_type_pair(
        comparison_pair: list,  # [comp_job_type, ref_job_type]
        pair_plot_data_dir: str,
        pair_pics_svg_dir: str,
        pair_summary_output_dir: str,
        id_prefix: str,
        ase_db_path: str,
        composite_periodic_filename: str = 'composite_periodic_order.pdf',
        composite_mae_filename: str = 'composite_mae_order.pdf',
        heatmap_filename_prefix: str = 'heatmap_error',  # Prefix for heatmap filenames
        element_mae_csv_filename: str = 'element_mae_data.csv',  # For saving element MAE data
):
    """
    Consolidates results (MAE, SVGs) for a single job type comparison pair
    and saves summary plots into the pair's specific summary directory.
    """
    comp_jt, ref_jt = str(comparison_pair[0]), str(comparison_pair[1])
    pair_label = f"{comp_jt}_vs_{ref_jt}"
    plot_data_path = Path(pair_plot_data_dir)
    pics_svg_path = Path(pair_pics_svg_dir)
    summary_output_path = Path(pair_summary_output_dir)
    summary_output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Consolidating Summary Plots for Pair: {pair_label} ---")
    print(f"Reading data from: {plot_data_path}")
    print(f"Reading SVGs from: {pics_svg_path}")
    print(f"Saving summary plots to: {summary_output_path}")

    # Basic check if source dirs contain necessary files
    if not any(plot_data_path.glob(f"{id_prefix}*.npz")):
        print(f"Warning: No .npz files found in {plot_data_path}. Skipping consolidation for {pair_label}.")
        return None, None, None
    if not any(pics_svg_path.glob(f'*_compare.svg')):
        print(f"Warning: No '*_compare.svg' files found in {pics_svg_path}. Skipping consolidation for {pair_label}.")
        return None, None, None

    symbol_to_z_global = get_element_z_map()
    analysis_results = {}

    # Dictionary to store MAE values for different metrics by element
    metric_keys = ['near_ef', 'occupied', 'all_band', 'n_occupied_and_0.5n_un_occupied', 'n_occupied_and_3_un_occupied', 'eta', 'eta_10', 'eta_max', 'eta_max_10']
    element_metrics = {key: {} for key in metric_keys}

    npz_files = sorted(list(plot_data_path.glob(f"{id_prefix}*.npz")))

    for npz_file in npz_files:
        id_name = npz_file.stem
        data = np.load(npz_file, allow_pickle=True)

        # Extract band data
        bands1 = data[f'bands_{comp_jt}']
        bands2 = data[f'bands_{ref_jt}']
        e_vbm_max1 = data[f'e_vbm_max_{comp_jt}']
        e_vbm_max2 = data[f'e_vbm_max_{ref_jt}']
        n_occupied_bands = data[f'n_occupied_bands_{ref_jt}']

        # Calculate error metrics
        error_metrics = quantify_band_error(
            e_vbm_max1=e_vbm_max1,
            e_vbm_max2=e_vbm_max2,
            bands1=bands1,
            bands2=bands2,
            vbm_index=n_occupied_bands - 1,
            num_valence_near_ef=5,
            num_conduction_near_ef=5
        )

        if not error_metrics:
            print(f"    Warning: No error metrics calculated for {id_name}")
            continue

        # Get element info and SVG path
        plot_title_id, formula, symbol, z = get_element_info(ase_db_path, id_name, id_prefix, symbol_to_z_global)
        svg_path_final = pics_svg_path / (formula + '_compare.svg')

        # Extract MAE values for each metric
        mae_values = {}
        for key in metric_keys:
            if key in error_metrics and error_metrics[key]:
                metric = error_metrics[key]
                if isinstance(metric, tuple) and hasattr(metric, 'mae'):
                    mae_values[key] = metric.mae
                elif isinstance(metric, dict) and 'mae' in metric:
                    mae_values[key] = metric['mae']
                else:
                    mae_values[key] = np.nan
            else:
                mae_values[key] = np.nan

        # Store analysis results
        analysis_results[id_name] = {
            'formula': formula,
            'z': z,
            'symbol': symbol,
            'mae': mae_values['eta'],  # Primary MAE for sorting
            'plot_path': svg_path_final,
            'mae_values': mae_values
        }

        # Store MAE values by element for each metric
        if symbol != "X":
            for key in metric_keys:
                if not np.isnan(mae_values[key]):
                    element_metrics[key][symbol] = round_sig(mae_values[key], 3)

    print(f"Gathered summary data for {len(analysis_results)} IDs for pair {pair_label}.")

    # --- Generate Composite Plots ---
    if analysis_results:
        valid_plots = [res['plot_path'] for res in analysis_results.values() if res['plot_path'] is not None]
        if valid_plots:
            print(f"Generating composite plots for {pair_label} using {len(valid_plots)} SVGs...")

            # Sort for periodic table order
            plots_sorted_periodic = sorted(analysis_results.values(), key=lambda x: x['z'])
            valid_paths_periodic = [p['plot_path'] for p in plots_sorted_periodic if p['plot_path'] is not None]

            # Sort for MAE order
            plots_sorted_mae = sorted(analysis_results.values(), key=lambda x: (np.isnan(x['mae']), x['mae']))
            valid_paths_mae = [p['plot_path'] for p in plots_sorted_mae if p['plot_path'] is not None]

            composite_periodic_path = str(summary_output_path / composite_periodic_filename)
            composite_mae_path = str(summary_output_path / composite_mae_filename)

            merge_svgs_to_pdf(valid_paths_periodic, composite_periodic_path)
            merge_svgs_to_pdf(valid_paths_mae, composite_mae_path)
        else:
            print(f"Skipping composite SVG generation for {pair_label}: No valid plot paths found.")
    else:
        print(f"Skipping composite SVG generation for {pair_label}: No analysis results.")

    # --- Generate Heatmaps for Different Metrics ---
    titles = {
        'near_ef': f"Band MAE near E$_f$ ({comp_jt} vs {ref_jt})",
        'occupied': f"Occupied Bands MAE ({comp_jt} vs {ref_jt})",
        'all_band': f"All Bands MAE ({comp_jt} vs {ref_jt})",
        'n_occupied_and_0.5n_un_occupied': f"Occ + 0.5×Unocc Bands MAE ({comp_jt} vs {ref_jt})",
        'n_occupied_and_3_un_occupied': f"Occ + 3 Unocc Bands MAE ({comp_jt} vs {ref_jt})",
        'eta': f"$\\eta_v$ (Valence Bands) MAE ({comp_jt} vs {ref_jt})",
        'eta_10': f"$\\eta_{{10}}$ (Valence + Conduction up to 10 eV) MAE ({comp_jt} vs {ref_jt})",
        'eta_max': f"max $\\eta$ (Maximum Band Difference) ({comp_jt} vs {ref_jt})",
        'eta_max_10': f"max $\\eta_{{10}}$ (Maximum Band Difference up to 10 eV) ({comp_jt} vs {ref_jt})"
    }

    for metric_key in metric_keys:
        element_data = element_metrics[metric_key]
        if element_data:
            print(f"\nGenerating {metric_key} heatmap for {pair_label} using {len(element_data)} elements...")

            heatmap_path = summary_output_path / f"{heatmap_filename_prefix}_{metric_key}.png"
            fig, ax = create_periodic_heatmap(
                element_values=element_data,
                title=titles[metric_key],
                output_file=str(heatmap_path),
                colormap='viridis',
                log_scale=True,
                colorbar_label=f"{metric_key} MAE (eV)"
            )
            plt.close(fig)
            print(f"  Saved heatmap: {heatmap_path.name}")
        else:
            print(f"\nSkipping {metric_key} heatmap for {pair_label}: No element data.")

    # --- Save Element MAE Data to CSV ---
    for metric_key in metric_keys:
        element_data = element_metrics[metric_key]
        if element_data:
            csv_path = summary_output_path / f"{metric_key}_{element_mae_csv_filename}"
            df = pd.DataFrame([element_data], index=[pair_label])
            df.to_csv(csv_path)
            print(f"  Saved {metric_key} element MAE data to: {csv_path.name}")

    print(f"--- Finished Summary Consolidation for Pair: {pair_label} ---")
    return summary_output_path, element_metrics


def create_consolidated_mae_reports(all_pairs_element_metrics, master_summary_path):
    """
    Creates consolidated CSV files for element MAE data across all comparison pairs.

    Args:
        all_pairs_element_metrics (dict): Dictionary with pair_label keys and element metrics values
        master_summary_path (Path): Path to master summary directory
    """
    # Process the near_ef metric for the main comparison
    all_pairs_element_mae = {pair: metrics['eta'] for pair, metrics in all_pairs_element_metrics.items()}

    # Create main consolidated CSV with all pairs' element MAE data
    consolidated_csv_path = master_summary_path / "all_pairs_element_mae.csv"
    df_all = pd.DataFrame.from_dict(all_pairs_element_mae, orient='index')
    df_all.index.name = 'Comparison'
    df_all.to_csv(consolidated_csv_path)
    print(f"\nSaved consolidated element MAE data for all pairs to: {consolidated_csv_path}")

    # Create sorted element MAE CSV based on average values
    element_avg_mae = df_all.mean(axis=0)
    sorted_columns = element_avg_mae.sort_values(ascending=False).index.tolist()
    df_sorted = df_all[sorted_columns]
    df_sorted.loc['Average'] = element_avg_mae[sorted_columns]
    sorted_csv_path = master_summary_path / "all_pairs_element_mae_sorted.csv"
    df_sorted.to_csv(sorted_csv_path)
    print(f"\nSaved sorted element MAE data (ordered by average MAE) to: {sorted_csv_path}")

    # For each metric type, create a separate CSV
    metric_keys = next(iter(all_pairs_element_metrics.values())).keys()
    for metric_key in metric_keys:
        metric_data = {pair: metrics[metric_key] for pair, metrics in all_pairs_element_metrics.items()}
        metric_csv_path = master_summary_path / f"{metric_key}_all_pairs.csv"
        pd.DataFrame.from_dict(metric_data, orient='index').to_csv(metric_csv_path)
        print(f"Saved {metric_key} metric data for all pairs to: {metric_csv_path}")


def compare_multiple_job_pairs_workflow(
        workspace_root: str,
        base_output_dir: str,
        job_types: list,
        ref_job_type: str,
        ase_db_path: str = None,
        id_prefix: str = 'db_seq_id_',
        target_folder: str = 'OUT.ABACUS',
        target_band_file: str = 'BANDS_1.dat',
        plot_ylim: list = [-10, 10],
        plot_styles: dict = None,
        force_reprocess: bool = False,
        force_replot: bool = False,
):
    """
    Main workflow comparing multiple job types against a reference job type.

    Runs separate comparisons (job_type vs ref_job_type) and generates summary
    plots for each pair, saving them within the pair's specific output directory.

    Args:
        workspace_root (str): Root directory containing job_type subdirectories.
        base_output_dir (str): Base directory to store all analysis outputs.
        job_types (list): List of strings identifying the job types (subdirs).
        ref_job_type (str): The string identifying the reference job type.
        ase_db_path (str, optional): Path to ASE database.
        id_prefix (str): Prefix for material ID directories.
        target_folder (str): Name of output folder within ID dir.
        target_band_file (str): Name of band structure file.
        plot_ylim (list): Y-axis limits for band plots.
        plot_styles (dict, optional): Define custom plot styles for job types.
        force_reprocess (bool): Force reprocessing of raw data.
        force_replot (bool): Force regeneration of plots.
    """
    print("=" * 40)
    print("Starting Pairwise Job Type Comparison Workflow (Summary per Pair)")
    print("=" * 40)

    workspace_path = Path(workspace_root)
    base_output_path = Path(base_output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)

    # Create master summary directory
    master_summary_path = base_output_path / "all_pairs_summary"
    master_summary_path.mkdir(parents=True, exist_ok=True)

    # Ensure job types are strings
    job_types_str = [str(jt) for jt in job_types]
    ref_job_type_str = str(ref_job_type)

    # Input validation
    if ref_job_type_str not in job_types_str:
        print(f"Error: Reference job type '{ref_job_type_str}' not found in job_types list: {job_types_str}")
        return

    valid_comp_job_types = [jt for jt in job_types_str if jt != ref_job_type_str]
    if not valid_comp_job_types:
        print("Error: No comparison job types found other than the reference.")
        return

    comparison_pairs = [[jt, ref_job_type_str] for jt in valid_comp_job_types]

    # Prepare plot styles
    internal_plot_styles = {}
    if plot_styles:
        for k, v in plot_styles.items():
            if isinstance(v, dict):
                internal_plot_styles[str(k)] = v.copy()
            else:
                internal_plot_styles[str(k)] = {'color': str(v)}

    # Assign default styles
    colors = plt.cm.viridis(np.linspace(0, 1, len(job_types_str)))
    job_type_indices = {jt: i for i, jt in enumerate(job_types_str)}

    for jt in job_types_str:
        if jt not in internal_plot_styles:
            internal_plot_styles[jt] = {}
        style = internal_plot_styles[jt]
        if 'color' not in style: style['color'] = colors[job_type_indices[jt]]
        if 'linestyle' not in style: style['linestyle'] = '-'
        if 'label' not in style: style['label'] = jt

    # Make reference style distinct
    ref_style = internal_plot_styles[ref_job_type_str]
    if 'linestyle' not in ref_style or ref_style['linestyle'] == '-':
        ref_style['linestyle'] = '--'
    if ref_style['label'] == ref_job_type_str:
        ref_style['label'] += ' (Ref)'

    # Print configuration summary
    print(f"Workspace: {workspace_path}")
    print(f"Base Output: {base_output_path}")
    print(f"Job Types: {job_types_str}")
    print(f"Reference Job Type: {ref_job_type_str}")
    print(f"Comparison Pairs: {comparison_pairs}")
    print(f"ID Prefix: '{id_prefix}'")
    print(f"Plot Y Lim: {plot_ylim}")
    print(f"ASE DB: {ase_db_path}")
    print(f"Force Reprocess: {force_reprocess}, Force Replot: {force_replot}")
    print("Plot Styles Applied:")
    for jt, style in internal_plot_styles.items(): print(f"  - {jt}: {style}")
    print("-" * 30)

    # Run pairwise comparisons and consolidate
    summary_paths_all = {}
    all_pairs_element_metrics = {}  # Store all element metrics for all pairs

    for pair in comparison_pairs:
        comp_jt, ref_jt = pair
        pair_label = f"{comp_jt}_vs_{ref_jt}"
        output_dir_pair = base_output_path / f"compare_{pair_label}"

        # Extract styles for this pair only
        pair_styles_for_run = {
            comp_jt: internal_plot_styles[comp_jt],
            ref_jt: internal_plot_styles[ref_jt]
        }

        # Run comparison for this pair
        plot_data_dir, pics_svg_dir = run_band_comparison_workflow(
            workspace_root=str(workspace_path),
            base_output_dir=str(output_dir_pair),
            job_types=pair,
            id_prefix=id_prefix,
            target_folder=target_folder,
            target_band_file=target_band_file,
            ase_db_path=ase_db_path,
            plot_ylim=plot_ylim,
            plot_styles=pair_styles_for_run,
            plot_filename_suffix='_compare.svg',
            force_reprocess=force_reprocess,
            force_replot=force_replot
        )

        # Consolidate results for this pair
        pair_summary_output_dir = output_dir_pair / "summary"
        summary_info, element_metrics = consolidate_results_for_job_type_pair(
            comparison_pair=pair,
            pair_plot_data_dir=str(plot_data_dir),
            pair_pics_svg_dir=str(pics_svg_dir),
            pair_summary_output_dir=str(pair_summary_output_dir),
            id_prefix=id_prefix,
            ase_db_path=ase_db_path,
        )

        if summary_info:
            summary_paths_all[pair_label] = summary_info
            if element_metrics:
                all_pairs_element_metrics[pair_label] = element_metrics

    # Copy individual pair summaries to master directory
    print("\n--- Consolidating All Pairs' Summaries to Master Directory ---")
    for pair_label, summary_path in summary_paths_all.items():
        pair_master_summary_dir = master_summary_path / pair_label
        pair_master_summary_dir.mkdir(parents=True, exist_ok=True)

        # Copy files from pair's summary directory to master directory
        source_dir = Path(summary_path)
        for file_path in source_dir.glob('*'):
            if file_path.is_file():
                shutil.copy2(file_path, pair_master_summary_dir)
        print(f"  Copied summary files for {pair_label} to {pair_master_summary_dir}")

    # Create consolidated MAE reports if we have data
    if all_pairs_element_metrics:
        create_consolidated_mae_reports(all_pairs_element_metrics, master_summary_path)

    # Final report
    print("\n" + "=" * 40)
    print("Workflow Summary:")
    if summary_paths_all:
        print("\nSummary plots generated in respective comparison directories:")
        for pair_label, summary_dir in summary_paths_all.items():
            print(f"  - {pair_label}: {summary_dir}")
        print(f"\nMaster summary directory with all pairs: {master_summary_path}")
    else:
        print("No summary plots were generated.")
    print("\nWorkflow Complete.")
    print("=" * 40)

# =============================================================================
# Test Execution Block
# =============================================================================

if __name__ == "__main__":
    # --- Configure and Run ---
    run_band_comparison_workflow(
        workspace_root='/root/compare_band_mp_20',
        base_output_dir='band_data',
        # --- ASE Database for Titles ---
        ase_db_path='/root/compare_band_mp_20/lcao/elementary_substances.db',
        # --- Job Types and IDs ---
        job_types=['pw', 'lcao'],
        id_prefix='db_seq_id_',
        # --- Plotting Options ---
        plot_ylim=[-10, 10],
        plot_styles={'pw': {'color': '#1f77b4', 'linestyle': '-', 'label': 'PW'},
                     'lcao': {'color': '#ff7f0e', 'linestyle': '--', 'label': 'LCAO'}},
        # --- Caching Control ---
        force_reprocess=False,
        force_replot=False
    )
