from skimage.measure import label, regionprops


def identify_fasicle_regions(annotation, return_num_regions=False, minimum_fasicle_area=15):
    regions = regionprops(label(annotation))
    regions = [x for x in regions if x.area > minimum_fasicle_area]
    return len(regions) if return_num_regions else regions
