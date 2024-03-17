import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
from glob import glob
import xarray as xr
import rioxarray as rxr
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio


path = 'data/LC08_L2SP_204025_20230904_20230912_02_T1/'
complete_dataset = os.listdir(path)
complete_dataset = [path + x for x in complete_dataset]
print(complete_dataset)

def show_rgb(bands_list, red=4, green=3, blue=2):
    stack = []
    colors = [red, green, blue]
    colors = ['B' + str(x) for x in colors]
    for color in colors:
        for band in bands_list:
            if color in band:
                with rio.open(band) as src:
                    array = src.read(1)
                    stack.append(array)
                break
                
    stack = np.dstack(stack)
    print(stack)
    datastack = stack
    
    stack = np.clip(stack*0.0000275-0.2, 0, 1)

    # Clip to enhance contrast
    stack = np.clip(stack,0,0.2)/0.2

    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(stack)
    plt.show()

show_rgb(complete_dataset)

#show_rgb(complete_dataset, red=7, green=6, blue=4)
#----------------------------------------------------

#--------------------------------K-nearest means-----------------------------
'''this part is a customized 
input = 7 bands of lansat 9 images
output = k-cluster images, 
'''
class ClusteredBands:
    
    def __init__(self, rasters_list):
        self.rasters = rasters_list
        self.model_input = None
        self.width = 0
        self.height = 0
        self.depth = 0
        self.no_of_ranges = None
        self.models = None
        self.predicted_rasters = None
        self.s_scores = []
        self.inertia_scores = []
        
    def set_raster_stack(self):
        band_list = []
        for image in self.rasters:
            with rio.open(image, 'r') as src:
                band = src.read(1)
                band = np.nan_to_num(band)
                band_list.append(band)
        bands_stack = np.dstack(band_list)
        
        # Prepare model input from bands stack
        self.width, self.height, self.depth = bands_stack.shape
        self.model_input = bands_stack.reshape(self.width * self.height, self.depth)
        
    def build_models(self, no_of_clusters_range):
        self.no_of_ranges = no_of_clusters_range
        models = []
        predicted = []
        inertia_vals = []
        s_scores = []
        for n_clust in no_of_clusters_range:
            kmeans = KMeans(n_clusters=n_clust)
            y_pred = kmeans.fit_predict(self.model_input)
            
            # Append model
            models.append(kmeans)
            
            # Calculate metrics
            s_scores.append(self._calc_s_score(y_pred))
            inertia_vals.append(kmeans.inertia_)
            
            # Append output image (classified)
            quantized_raster = np.reshape(y_pred, (self.width, self.height))
            predicted.append(quantized_raster)
            print("yo1")
            
        # Update class parameters
        self.models = models
        self.predicted_rasters = predicted
        self.s_scores = s_scores
        self.inertia_scores = inertia_vals
        
    def _calc_s_score(self, labels):
        s_score = silhouette_score(self.model_input, labels, sample_size=1000)
        return s_score
        
    def show_clustered(self):
        for idx, no_of_clust in enumerate(self.no_of_ranges):
            title = 'Number of clusters: ' + str(no_of_clust)
            image = self.predicted_rasters[idx]
            print("yo2")
            plt.figure(figsize = (15,15))
            plt.axis('off')
            plt.title(title)
            plt.imshow(image, cmap='Accent')
            plt.colorbar()
            plt.show()
            
    def show_inertia(self):
        plt.figure(figsize = (10,10))
        plt.title('Inertia of the models')
        plt.plot(self.no_of_ranges, self.inertia_scores)
        plt.show()
        print("yo3")
        
    def show_silhouette_scores(self):
        plt.figure(figsize = (10,10))
        plt.title('Silhouette scores')
        plt.plot(self.no_of_ranges, self.s_scores)
        plt.show()
        print("yo4")

clustered_models = ClusteredBands(complete_dataset)
clustered_models.set_raster_stack()

ranges = np.arange(3,10,1)
clustered_models.build_models(ranges)
clustered_models.show_clustered()
clustered_models.show_inertia()
clustered_models.show_silhouette_scores()
