"""Disabled: designer rejected strip fitting for contour/hierarchy regression."""
from textile.reference_repair import fit_reference

if __name__=='__main__':
    # Fail before opening targets or producing any new raster.
    fit_reference(None,None)
