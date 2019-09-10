# ReGroup
Python library for reconstruction of a finite matrix group known approximately

## Prereqisites:
  numpy and scipy

## Use:

'''
import grpcorr
'''

The module grpcorr.py contains three driver routines:

multab_group_correction - performes multiplication table based reconstruction

lsf_group_correction    - performes simultaneous least squares fit to the set of given matrices

abfit_group_correction  - performes simultaneous least squares fit with a set of vectors

See the comments inside.
