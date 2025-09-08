import rioxarray as rioxr
import richdem as rd
import scipy.ndimage as nd
import numpy as np

def map_lakes(topg_files, e):

    '''
    Provide a list of paths to subglacial topography files, such as those from
    the data supplement of "Unveiling the Hidden Lake-Rich Landscapes Under Earthś Glaciers".

    Writes .tif file of lake depths (m).
    '''
    
    for topg_file in topg_files:
        topg = rioxr.open_rasterio(topg_file)
        buffer_sea_level = _internal_buffer(2, topg.data[0] != 0)
        rd_topg = rd.rdarray(topg.data[0], no_data = 0)
        rd_topg.projection = topg.rio.crs.to_wkt()
        rd_topg.geotransform = topg.rio.transform().to_gdal()

        # run sink-fill algorithm from richdem
        topg_filled = rd.FillDepressions(rd_topg, in_place=False)
        lakes = np.array(topg_filled) - topg
        labelled_lakes, n_features = nd.label(lakes.data)

        # remove lakes shallower than e m
        lakes.data[0][lakes.data[0] <= e] = np.nan
        labelled_lakes[0][lakes.data[0] <= e] = 0

        lakes_touching_ocean = np.unique(labelled_lakes * buffer_sea_level)

        resolution = topg.rio.resolution()[0]
        for f in np.unique(labelled_lakes):
            # remove lakes smaller than 0.05 km2
            if (np.sum(labelled_lakes == f) * resolution**2) < 5e4:
                lakes.data[0][labelled_lakes[0] == f] = np.nan

            # remove lakes within two pixels of ocean
            if f in lakes_touching_ocean:
                lakes.data[0][labelled_lakes[0] == f] = np.nan


        lakes.rio.to_raster(topg_file.split('_')[0] + '_lakes.tif')


def _internal_buffer(bw, mask):

    '''
    Creates buffer inside a mask,
    Bw is buffer width in pixels.

    '''
    
    mask_iter = mask == 1
    mask_bw = ~mask_iter
    buffer = np.zeros_like(mask_iter)
    for i in range(bw):
        boundary_mask = mask_bw==0
        k = np.ones((3,3),dtype=int)
        boundary = nd.binary_dilation(boundary_mask==0, k) & boundary_mask
        mask_bw = np.where(boundary, 1, mask_bw)
    buffer = ((mask_bw + mask_iter)-1)
    
    return buffer

        
if __name__ == '__main__':
    map_lakes(['./Data/RGI60-13.13272_topg.tif', './Data/RGI60-06.00377_topg.tif'], 5)
