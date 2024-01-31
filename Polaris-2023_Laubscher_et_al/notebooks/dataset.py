import os
import imageio
import numpy as np
import pandas as pd
from deepcell.datasets.dataset import Dataset

DATA_URL = "https://deepcell-data.s3.us-west-1.amazonaws.com/spot_detection/PolarisPublicationData.zip"
DATA_HASH = "8b82b6bd3e4aca811464279487630c22"

class PolarisPublicationData(Dataset):
    def __init__(self):
        super().__init__(
            url=DATA_URL,
            file_hash=DATA_HASH,
            secure=False
        )

    def load_data(self, figure):
        """Load the specified data file required for reproducing a publication figure from
        Laubscher et al. (2023).

        Args:
            figure (str): Data split to load from `['1', '2', 'S2', 'S4', 'S5', 'S6', 'S7',
            'S8', 'S9', 'S11']`.

        Raises:
            ValueError: Figure must be one of `['1', '2', 'S2', 'S4', 'S5', 'S6', 'S7', 'S8',
            'S9', 'S11']`.
        """
        if figure not in ['1', '2', 'S2', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']:
            raise ValueError('Figure must be one of 1, 2, S2, S4, S5, S6, S7, S8, S9, S10')

        if figure=='1':
            fname = 'MERFISH_cropped.tiff'
            fpath = os.path.join(self.path, fname)
            return self._load_tif(fpath)
        
        if figure=='2':
            results_fname = 'Petukhov_results_226.csv'
            results_fpath = os.path.join(self.path, results_fname)
            results = self._load_csv(results_fpath)
            results = results.reset_index(drop=True)

            segmentation_fname = 'Petukhov_results_226.npy'
            segmentation_fpath = os.path.join(self.path, segmentation_fname)
            segmentation = self._load_npy(segmentation_fpath)
            
            return results, segmentation
            
        if figure=='S2':
            MERFISH_fname = 'MERFISH_image.tiff'
            MERFISH_fpath = os.path.join(self.path, MERFISH_fname)
            MERFISH_im = self._load_tif(MERFISH_fpath)

            ISS_fname = 'ISS_image.tiff'
            ISS_fpath = os.path.join(self.path, ISS_fname)
            ISS_im = self._load_tif(ISS_fpath)

            splitFISH_fname = 'splitFISH_image.tiff'
            splitFISH_fpath = os.path.join(self.path, splitFISH_fname)
            splitFISH_im = self._load_tif(splitFISH_fpath)
            splitFISH_im = np.expand_dims(splitFISH_im, axis=[0,-1])

            seqFISH_fname = 'seqFISH_image.tiff'
            seqFISH_fpath = os.path.join(self.path, seqFISH_fname)
            seqFISH_im = self._load_tif(seqFISH_fpath)
            seqFISH_im = np.expand_dims(seqFISH_im, axis=[0,-1])

            SunTag_fname = 'SunTag_image.tiff'
            SunTag_fpath = os.path.join(self.path, SunTag_fname)
            SunTag_im = self._load_tif(SunTag_fpath)
            SunTag_im = SunTag_im[40]
            SunTag_im = np.expand_dims(SunTag_im, axis=[0,-1])
            
            return MERFISH_im, ISS_im, splitFISH_im, seqFISH_im, SunTag_im
            
        if figure=='S4':
            DoG_fname = 'DoG_coords.npy'
            DoG_fpath = os.path.join(self.path, DoG_fname)
            DoG_coords = self._load_npy(DoG_fpath)

            LoG_fname = 'LoG_coords.npy'
            LoG_fpath = os.path.join(self.path, LoG_fname)
            LoG_coords = self._load_npy(LoG_fpath)

            PLM_fname = 'PLM_coords.npy'
            PLM_fpath = os.path.join(self.path, PLM_fname)
            PLM_coords = self._load_npy(PLM_fpath)

            trackpy_fname = 'trackpy_coords.npy'
            trackpy_fpath = os.path.join(self.path, trackpy_fname)
            trackpy_coords = self._load_npy(trackpy_fpath)

            polaris_fname = 'polaris_coords.npy'
            polaris_fpath = os.path.join(self.path, polaris_fname)
            polaris_coords = self._load_npy(polaris_fpath)
            
            airloc_fname = 'airlocalize_coords.npy'
            airloc_fpath = os.path.join(self.path, airloc_fname)
            airloc_coords = self._load_npy(airloc_fpath)

            all_coords = {
                'DoG': DoG_coords,
                'LoG': LoG_coords,
                'PLM': PLM_coords,
                'trackpy': trackpy_coords,
                'airloc': airloc_coords,
                'polaris': polaris_coords
            }
            
            return all_coords
            
        if figure=='S5':
            fname = 'receptive_field_data.csv'
            fpath = os.path.join(self.path, fname)
            return self._load_csv(fpath)

        if figure=='S6':
            density_fname_model = 'density_model_benchmarking.csv'
            density_fpath_model = os.path.join(self.path, density_fname_model)
            density_data_model = self._load_csv(density_fpath_model)

            intensity_fname_model = 'intensity_model_benchmarking.csv'
            intensity_fpath_model = os.path.join(self.path, intensity_fname_model)
            intensity_data_model = self._load_csv(intensity_fpath_model)
            
            density_fname_method = 'density_method_benchmarking.csv'
            density_fpath_method = os.path.join(self.path, density_fname_method)
            density_data_method = self._load_csv(density_fpath_method)

            intensity_fname_method = 'intensity_method_benchmarking.csv'
            intensity_fpath_method = os.path.join(self.path, intensity_fname_method)
            intensity_data_method = self._load_csv(intensity_fpath_method)
            
            return (density_data_model,
                    intensity_data_model,
                    density_data_method,
                    intensity_data_method)
            
        if figure=='S7':
            fname = 'dropout_benchmarking_data.csv'
            fpath = os.path.join(self.path, fname)
            return self._load_csv(fpath)

        if figure=='S8':
            MERFISH_im_fname = 'liu_MERFISH_im.tiff'
            MERFISH_im_fpath = os.path.join(self.path, MERFISH_im_fname)
            MERFISH_im = self._load_tif(MERFISH_im_fpath)

            MERFISH_results_fname = 'liu_MERFISH_all_genes.csv'
            MERFISH_results_fpath = os.path.join(self.path, MERFISH_results_fname)
            MERFISH_results = self._load_csv(MERFISH_results_fpath)

            seqFISH_im_fname = 'macrophage_seqFISH_im.tiff'
            seqFISH_im_fpath = os.path.join(self.path, seqFISH_im_fname)
            seqFISH_im = self._load_tif(seqFISH_im_fpath)

            seqFISH_results_fname = 'macrophage_all_genes.csv'
            seqFISH_results_fpath = os.path.join(self.path, seqFISH_results_fname)
            seqFISH_results = self._load_csv(seqFISH_results_fpath)
            
            return MERFISH_im, MERFISH_results, seqFISH_im, seqFISH_results

        if figure=='S9':
            Petukhov_data_fname = 'petukhov_polaris_comparison.csv'
            Petukhov_data_fpath = os.path.join(self.path, Petukhov_data_fname)
            Petukhov_data = self._load_csv(Petukhov_data_fpath)

            Liu_data_fname = 'liu_polaris_comparison.csv'
            Liu_data_fpath = os.path.join(self.path, Liu_data_fname)
            Liu_data = self._load_csv(Liu_data_fpath)

            seqFISH_data_fname = 'seqfish_polaris_comparison.csv'
            seqFISH_data_fpath = os.path.join(self.path, seqFISH_data_fname)
            seqFISH_data = self._load_csv(seqFISH_data_fpath)
            
            all_data = {
                'Petukhov et al.': Petukhov_data,
                'Liu et al.': Liu_data,
                'seqFISH': seqFISH_data
            }
            
            return all_data

        if figure=='S10':
            results_fname = 'feldman_iss_results.csv'
            results_fpath = os.path.join(self.path, results_fname)
            results = self._load_csv(results_fpath)

            segmentation_fname = 'feldman_iss_segmentation.tiff'
            segmentation_fpath = os.path.join(self.path, segmentation_fname)
            segmentation = self._load_tif(segmentation_fpath)

            bulk_counts_fname = 'feldman_iss_bulk_counts.csv'
            bulk_counts_fpath = os.path.join(self.path, bulk_counts_fname)
            bulk_counts = self._load_csv(bulk_counts_fpath)
            
            return results, segmentation, bulk_counts
        
        
    def _load_tif(self, fpath):
        data = imageio.volread(fpath)
        data = np.array(data)

        return data
    
    def _load_npy(self, fpath):
        data = np.load(fpath, allow_pickle=True)

        return data
    
    def _load_csv(self, fpath):
        data = pd.read_csv(fpath, index_col=0)

        return data