import numpy as np
from skimage.measure import label, regionprops
from stats import calculate_regions

def get_outlier_regions(regions, area_threshold=3101, eccen_threshold=[0.33, 0.95]):
    outliers = []
    for r in regions:
        if r.area > area_threshold or r.eccentricity < eccen_threshold[0] or r.eccentricity > eccen_threshold[1]:
            outliers.append(r)
    return outliers

def draw_outliers_regions(input_mask, area_threshold=3101, eccen_threshold=[0, 0.95]):
    modified = np.dstack(([input_mask, input_mask, input_mask]))
    regions = calculate_regions(input_mask)[0]
    outliers = get_outlier_regions(regions, area_threshold, eccen_threshold)

    for out in outliers:
        for c in out.coords:
            modified[c[0], c[1]] = [1, 0, 0]

    return modified