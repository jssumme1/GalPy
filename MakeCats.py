import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import math
from glob import glob
from astropy.coordinates import SkyCoord
from tqdm import tqdm

class MakeCats:

    ### INITIALIZE
    
    def __init__(self, field, cat_dir, catfile, pixsize):
        #Initialize catalog locations and field name
        self.field = field
        self.cat_dir = cat_dir
        self.catfile = catfile
        self.catname = f'{self.field}_total_cat.txt'

        if type(pixsize) == float:
            self.sw_pixel = pixsize
            self.lw_pixel = pixsize
        else:
            self.sw_pixel = pixsize[0]
            self.lw_pixel = pixsize[1]

        #Create empty data structures
        self.filts = []

        self.ra_dict, self.dec_dict, self.mag_dict, self.fwhm_dict = {}, {}, {}, {}
        self.x_dict, self.y_dict, self.err_dict, self.num_dict = {}, {}, {}, {}
        
        self.ra = Table()
        self.dec = Table()
        self.mag = Table()
        self.fwhm = Table()
        self.x = Table()
        self.y = Table()
        self.err = Table()
        self.num = Table()

        ### REMOVE PREEXISTING TOTAL CATALOG
        if os.path.exists(os.path.join(cat_dir, self.catname)):
            os.system(f'rm {os.path.join(cat_dir, self.catname)}')

        ### GET SEXTRACTOR CATALOGS
        ### Using directory and naming scheme fed to the constructor
        self.cats = glob(os.path.join(self.cat_dir, self.catfile))
        self.cats.sort()

        ### GET FILTERS CORRESPONDING TO EACH CATALOG
        for cat in self.cats:
            catname = cat.lower().split('/')[-1]
            for i in range(len(catname)):
                    if catname[i] == 'f' and (catname[i+4] == 'w' or catname[i+4] == 'm') and catname[i+1] in '0 1 2 3 4 5 6 7 8 9'.split():
                        filt = catname[i:i+5]
                        self.filts.append(filt.upper())
                        break

        ### PRINT
        print(f'Initialized. Found {len(self.cats)} catalogs.')

    ### Method photometry
    ### returns a table of mag, magerr, and fwhm

    def photometry(self, table, filt):
        photom = Table()
        
        idx = np.where((table['MAGERR_AUTO'] < 5) & (table['ALPHA_J2000'] != 0) & (table['DELTA_J2000'] != 0))
        
        photom['mag'] = table['MAG_AUTO'][idx]
        photom['err'] = table['MAGERR_AUTO'][idx]
        photom['ra'] = table['ALPHA_J2000'][idx]
        photom['dec'] = table['DELTA_J2000'][idx]
        photom['x'] = table['X_IMAGE'][idx]
        photom['y'] = table['Y_IMAGE'][idx]
        photom['num'] = table['NUMBER'][idx]
        
        if int(filt[1:4]) < 250:
            photom['fwhm'] = self.sw_pixel * table['FWHM_IMAGE'][idx]
            
        else:
            photom['fwhm'] = self.lw_pixel * table['FWHM_IMAGE'][idx]

        return photom

    ### Method photom_dict
    ### gets photometry for all filters and puts them in a dictionary for each attribute

    def photom_dicts(self):
        for i in tqdm(range(len(self.cats)),desc='Getting photometry...'):
            filt = self.filts[i]
            t = Table.read(self.cats[i], format='ascii')
            photom = self.photometry(t, filt)
            self.ra_dict[filt] = photom['ra']
            self.dec_dict[filt] = photom['dec']
            self.mag_dict[filt] = photom['mag']
            self.fwhm_dict[filt] = photom['fwhm']
            self.x_dict[filt] = photom['x']
            self.y_dict[filt] = photom['y']
            self.err_dict[filt] = photom['err']
            self.num_dict[filt] = photom['num']
    
    ### Method align
    ### matches individual catalogs to each other
    ### det_filt is the primary filter which all other filters are matched to

    def align(self, det_filt):
        #begin table from detection filter
        self.ra[det_filt] = self.ra_dict[det_filt]
        self.dec[det_filt] = self.dec_dict[det_filt]
        self.mag[det_filt] = self.mag_dict[det_filt]
        self.fwhm[det_filt] = self.fwhm_dict[det_filt]
        self.x[det_filt] = self.x_dict[det_filt]
        self.y[det_filt] = self.y_dict[det_filt]
        self.err[det_filt] = self.err_dict[det_filt]
        self.num[det_filt] = self.num_dict[det_filt]

        #create skycoord object with detection filter
        c = SkyCoord(ra=self.ra[det_filt], dec=self.dec[det_filt], unit='deg')

        for i in tqdm(self.filts,desc='SkyCoord Match...'):
            if i != det_filt:
                #convert to arrays for easy data manipulation
                ra1 = np.array(self.ra_dict[i])
                dec1 = np.array(self.dec_dict[i])
                mag1 = np.array(self.mag_dict[i])
                fwhm1 = np.array(self.fwhm_dict[i])
                x1 = np.array(self.x_dict[i])
                y1 = np.array(self.y_dict[i])
                magerr1 = np.array(self.err_dict[i])
                num1 = np.array(self.num_dict[i])

                #match catalogs
                catalog = SkyCoord(ra=ra1, dec=dec1, unit='deg')
                idx, d2d, d3d = c.match_to_catalog_sky(catalog)
                
                #add matched data to the existing table
                self.ra[i] = ra1[idx]
                self.dec[i] = dec1[idx]
                self.mag[i] = mag1[idx]
                self.fwhm[i] = fwhm1[idx]
                self.x[i] = x1[idx]
                self.y[i] = y1[idx]
                self.err[i] = magerr1[idx]
                self.num[i] = num1[idx]

        ### Method save_table
        ### removes extraneous detections from the tables
        ### makes sure the objects are close enough - tolerance in degrees
        ### removes any objects detected in fewer filters than min_filters
        ### also saves the matched catalog file

    def match(self, det_filt, tolerance=0.0001, min_filters=4, flux=False):
        #call previous methods to make matched tables
        self.photom_dicts()
        self.align(det_filt)

        proj = abs(np.cos(self.dec[det_filt][0]*2*math.pi/360))

        for filt in self.filts:
            idx = np.where((np.absolute(self.ra[filt] - self.ra[det_filt]) > tolerance/proj) | (np.absolute(self.dec[filt] - self.dec[det_filt]) > tolerance))
            self.ra[filt][idx] = 0
            self.dec[filt][idx] = 0
            self.mag[filt][idx] = 0
            self.fwhm[filt][idx] = 0
            self.x[filt][idx] = 0
            self.y[filt][idx] = 0
            self.err[filt][idx] = 0
            self.num[filt][idx] = 0

        for i in tqdm(range(len(self.mag[det_filt]) - 1, -1, -1), desc='Filtering Non-Detections...'):
            #Count number of filters that didn't detect each object
            nondetections = 0
            for filt in self.filts:
                nondetections += int(self.mag[filt][i] == 0)

            #Filter out filters with detections < min_filters
            if len(self.filts) - nondetections < min_filters:
                self.mag.remove_row(i)
                self.ra.remove_row(i)
                self.dec.remove_row(i)
                self.fwhm.remove_row(i)
                self.x.remove_row(i)
                self.y.remove_row(i)
                self.err.remove_row(i)
                self.num.remove_row(i)
            
        #Create overall table
        self.final_table = Table()
        self.final_table['#id'] = range(1, len(self.mag[det_filt])+1)
        self.final_table['RA'] = self.ra[det_filt]
        self.final_table['DEC'] = self.dec[det_filt]
        self.final_table['X'] = self.x[det_filt]
        self.final_table['Y'] = self.y[det_filt]

        #put magnitudes in table
        for filt in self.filts:
            self.final_table[filt] = self.mag[filt]
            self.final_table[filt + '_err'] = self.err[filt]
            self.final_table[filt + '_num'] = self.num[filt]
        
        if flux == True:
            for filt in self.filts: # erg/s/cm^2???????????????????
                for i in range(len(self.final_table[filt])):
                    self.final_table[filt][i] = (10**(-0.4*(self.final_table[filt][i]+48.6)))*1e-7*1e4*1e26*1e6
                for i in range(len(self.final_table[filt + '_err'])):
                    self.final_table[filt + '_err'][i] = (2.5/np.log(10))*self.final_table[filt + '_err'][i]*(self.final_table[filt][i])
            
        #Save catalog
        if flux == True:
            self.catname = f'{self.field}_total_flux_cat.txt'
            
        print(f'SAVING {os.path.join(self.cat_dir, self.catname)}') 
        self.final_table.write(os.path.join(self.cat_dir, self.catname), format='ascii', overwrite=True)
