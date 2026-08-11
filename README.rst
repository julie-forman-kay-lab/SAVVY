SAVVY
=====

**Surface Accessibility Via Vector Yield** is a Python workflow for
quantifying the local steric approachability of selected residues or
multi-residue regions across structural ensembles.

Overview
--------
SAVVY samples uniform approach directions around a user-defined
target residue and determines whether a spherical probe can traverse a finite,
straight approach vector without sterically clashing with other protein heavy atoms.

The primary endpoint is the **Accessible Vector Yield Fraction**:
::
   accessible_fraction = unblocked_directions / sampled_directions

For Serine and Threonine residues, the default target origin is the side-chain
hydroxyl oxygen (OG/OG1). For other amino acids, the default target origin is
the geometric centroid of the side-chain heavy atoms.

For binding analyses, SAVVY evaluates three conditions:

* ``APO``: the unbound protein ensemble.
* ``BOUND_ONLY``: the bound-state protein coordinates with binding partner chains removed.
* ``BOUND_FULL``: the intact protein-partner complex.

The resulting effects are decomposed as:
::
   conformational = BOUND_ONLY - APO
   direct_partner = BOUND_FULL - BOUND_ONLY
   total          = BOUND_FULL - APO

Negative values indicate reduced approach accessibility.

SAVVY measures probe-dependent geometric approachability. It is not a direct
measure of solvent-accessible surface area, binding affinity, catalytic
geometry, reaction rate, or binding kinetics.

Requirements
------------
SAVVY requires Python 3.9 or newer and the following Python packages:
::
   numpy
   pandas
   scipy
   matplotlib
   tqdm

Clone the repository and install the required packages:

.. code-block:: bash

   git clone https://github.com/julie-forman-kay-lab/SAVVY
   cd SAVVY
   python -m pip install numpy pandas scipy matplotlib tqdm

Usage Examples
--------------
Basic APO-versus-bound ensemble analysis:

.. code-block:: python

   python run_savvy.py \
       --apo-dir ./apo_pdbs \
       --bound-dir ./bound_pdbs \
       --apo-target-chain A \
       --bound-target-chain B \
       --bound-partner-chains all \
       --target-residues 38,47,66,71,84 \
       --outdir ./savvy_results \
       --n-workers 8

The default primary calculation uses a 4 Å spherical probe, a 20 Å approach
path, 2,048 sampled spherical Fibonacci directions, and exclusion of the target
residue plus two neighboring residues on each side from the target.

Analyzing 5 targets across 200 conformations (100 apo + 100 bound) with 10 CPU workers takes ~10 minutes.

Multi-Residue Region Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A residue group can be summarized within each conformer before ensemble
bootstrapping as a protein region:

.. code-block:: python

   python run_savvy.py \
       --apo-dir ./apo_pdbs \
       --bound-dir ./bound_pdbs \
       --target-residues 117,118,119,120,121 \
       --region-name FEMDI \
       --region-residues 117,118,119,120,121 \
       --outdir ./savvy_femdi_results

Regenerate Plots Without Repeating Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you would like to regenerate plots with specfic plotting parameters without
repeating the calculations, you can use the ``--redo-plots`` option:

.. code-block:: python

   python run_savvy.py \
       --redo-plots ./savvy_results \
       --outdir ./savvy_replotted \
       --figure-dpi 600 \
       --residue-label-shift -1

The residue-label shift changes only the labeled residue numbering on the plots
and does not alter the target residues used in the calculation.

Citation
--------
If you use SAVVY, please cite the following publication where SAVVY was first described:

.. code-block:: rst

    Smyth S., Liu Z.H., Tsangaris T.E., Head-Gordon T., Forman-Kay J.D., Gradinaru C.C., (2026). Conformational Ensembles of the Disordered 4E-BP2:eIF4E Complex Restrained by smFRET Experiments. bioRxiv 2026.04.21.719986; doi: https://doi.org/10.64898/2026.04.21.719986

Version
-------
v0.4.0
