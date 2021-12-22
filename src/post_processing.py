import numpy as np
from stats import calculate_regions

def get_outlier_regions(regions, area_threshold=3101, eccen_threshold=(0.33, 0.95)):
    """
        Get regions of a prediction whose areas and eccentricities fall outside of the 99-th quantile
        of the distribution from the training dataset

        Parameters
        ---------------
        regions:
            the list of regions from a mask
        area_threshold: int, optional
            the upper threshold corresponding to the 99th quantile of the area distribution
        eccen_threshold: (float, float), optional
            the thresholds corresponding to the 1st and 99th quantile of the eccentricity distribution

        Returns
        ---------------
        the outliers regions int the mask
    """

    outliers = []
    for r in regions:
        if r.area > area_threshold or r.eccentricity < eccen_threshold[0] or r.eccentricity > eccen_threshold[1]:
            outliers.append(r)
    return outliers

def draw_outliers_regions(mask, area_threshold=3101, eccen_threshold=[0, 0.95]):
    """
        Change the color of outliers regions into red

        Parameters
        ---------------
        mask:   np.ndarray
            the input mask
        area_threshold: int, optional
            the upper threshold corresponding to the 99th quantile of the area distribution
        eccen_threshold: (float, float), optional
            the thresholds corresponding to the 1st and 99th quantile of the eccentricity distribution

        Returns
        ----------------
        the recolored mask
    """
    modified = np.dstack(([mask, mask, mask]))
    regions = calculate_regions(mask)[0]
    outliers = get_outlier_regions(regions, area_threshold, eccen_threshold)

    for out in outliers:
        for c in out.coords:
            modified[c[0], c[1]] = [1, 0, 0]

    return modified