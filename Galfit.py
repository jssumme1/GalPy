import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from astropy.visualization import simple_norm
from astropy.table import Table
import math
from glob import glob
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
matplotlib.rc('font', family='serif', serif='cm')
matplotlib.rc('text', usetex=True)
matplotlib.rc('ps', usedistiller='xpdf')

from photutils.background import Background2D
from photutils.isophote import EllipseGeometry
from photutils.isophote import Ellipse

import SExtractor
import MakeCats

class Galfit:

    def __init__(self, field, images, cats, master_cat, img_size=200, psf_size=150, star=False,
                 image_dir='mosaics', object_dir='objects', psf_dir='psfs', cat_dir='cats'):
        self.field = field
        self.images = images
        self.images = images
        self.cats = cats
        self.master_cat = Table.read(master_cat, format='ascii')
        self.img_size = img_size
        self.psf_size = psf_size
        self.star = star
        self.image_dir = image_dir
        self.object_dir = object_dir
        self.psf_dir = psf_dir
        self.cat_dir = cat_dir

        # make the directory structure in cwd if it doesn't exist
        if not os.path.exists(self.image_dir):
            os.mkdir(self.image_dir)
        if not os.path.exists(self.object_dir):
            os.mkdir(self.object_dir)
        if not os.path.exists(self.psf_dir):
            os.mkdir(self.psf_dir)
        if not os.path.exists(self.cat_dir):
            os.mkdir(self.cat_dir)

        ###   
        ### make smaller psf images for convolution
        ###
        for im in tqdm(self.images, desc='Creating PSF cutouts...'):
            filt = im.split('_')[-4]
            psf = os.path.join(self.psf_dir, f'{filt}_webbpsf.fits')

            with fits.open(psf) as hdul:
                # use same header for both extensions--science header has WCS info
                data = hdul['OVERSAMP'].data
                min = math.floor(166.5 - psf_size)
                max = math.ceil(166.5 + psf_size)
                hdul['OVERSAMP'].data = data[min:max, min:max] # SIZE OF PSF
                hdul.writeto(psf.replace('webbpsf', 'webbpsf_cutout'), overwrite=True)



    ###
    ### # make sci, err, and mask files for the object
    ###
    def make_object(self, obj_id, overwrite=True):
        # NEED TO MAKE MASK FROM SEGMENTATION MAP

        # parameters of the object
        x_obj = self.master_cat[obj_id-1]['X']
        y_obj = self.master_cat[obj_id-1]['Y']
        ra_obj = self.master_cat[obj_id-1]['RA']
        dec_obj = self.master_cat[obj_id-1]['DEC']

        # create new directory for all files of the object
        if not os.path.exists(os.path.join(self.object_dir, f'obj_{obj_id}')):
            os.mkdir(os.path.join(self.object_dir, 'obj_'+str(obj_id)))
            
        for im in tqdm(self.images, desc='Creating object cutouts...'):
            filt = im.split('_')[-4]
            if not os.path.exists(os.path.join(self.object_dir, f'obj_{obj_id}/{filt}')):
                os.mkdir(os.path.join(self.object_dir, f'obj_{obj_id}/{filt}'))

            # make sci and err/sigma images
            with fits.open(im) as hdul:
                # use same header for both extensions--science header has WCS info
                header = hdul['sci'].header[:]
                self.field_RA = header['RA_V1']
                self.field_DEC = header['DEC_V1']

                # save science for fitting and err for uncertainty ("sigma image")
                for extension in 'sci err'.split():
                    output_image = os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/obj_{obj_id}_{filt}_{extension}.fits')
                    header['EXTNAME'] = extension
                    header['EXPTIME'] = 1 # from galfit readme

                    # cutout the data and fix the WCS pixel
                    data = hdul[extension].data[int(y_obj-self.img_size):int(y_obj+self.img_size), int(x_obj-self.img_size):int(x_obj+self.img_size)]
                    header['CRPIX1'] = y_obj
                    header['CRPIX2'] = x_obj
                    header['CRVAL1'] = dec_obj
                    header['CRVAL2'] = ra_obj

                    # save cutout
                    fits.writeto(output_image, data, header, overwrite=overwrite)

            # make mask file
            segm = os.path.join(self.cat_dir, f'{self.field.upper()}_{filt.upper()}_segm.fits')
            filt_obj_id = self.master_cat[obj_id-1][f'{filt.upper()}_num']

            with fits.open(segm) as hdul:
                header = hdul[0].header[:]
                output_image = os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/obj_{obj_id}_{filt}_mask.fits')
                data = hdul[0].data[int(y_obj-self.img_size):int(y_obj+self.img_size), int(x_obj-self.img_size):int(x_obj+self.img_size)]
                data[data == filt_obj_id] = 0
                data[data != 0] = 1294124012

                # save cutout
                fits.writeto(output_image, data, header, overwrite=overwrite)

        # initialize string of components to fit
        self.components = ''
        self.galfit_images = []
        self.constraints_list = ''
        self.constraint = False

    ###
    ### begin galfit parameters
    ###
    def galfit_params(self, obj_id, filt, zp=28.0865):
        # test image
        proj = abs(np.cos(self.field_DEC*2*math.pi/360)) # projection due to target RA
        # write some parameters for feedme file
        params = {}
        params['A'] = f'obj_{obj_id}_{filt}_sci.fits'                          # Input data image (FITS file)
        params['B'] = f'obj_{obj_id}_{filt}_imgblock01.fits'                   # Output data image block
        params['C'] = f'obj_{obj_id}_{filt}_err.fits'                          # Sigma image name (made from data if blank or "none") 
        params['D'] = f'../../../psfs/{filt}_webbpsf_cutout.fits'              # Input PSF image and (optional) diffusion kernel
        params['E'] = 1                                                        # PSF fine sampling factor relative to data 
        params['F'] = f'obj_{obj_id}_{filt}_mask.fits'                         # Bad pixel mask (FITS image or ASCII coord list)
        
        if self.constraint == True:
            params['G'] = 'constraints.txt'                                    # File with parameter constraints (ASCII file) 
        else:
            params['G'] = 'none'                                               # File with parameter constraints (ASCII file) 
        params['H'] = f'1 {2*self.img_size} 1 {2*self.img_size}'               # Image region to fit (xmin xmax ymin ymax)
        params['I'] = f'{2*self.psf_size} {2*self.psf_size}'                   # Size of the convolution box (x y)
        params['J'] = zp                                                       # Magnitude photometric zeropoint 
        params['K'] = f'{0.03/proj} 0.03'                                      # Plate scale (dx dy)    [arcsec per pixel]
        params['O'] = 'regular'                                                # Display type (regular, curses, both)
        params['P'] = 0                                                        # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps

        return params

    ###
    ### add galfit component
    ###
    def add_component(self, obj_id, filt, form, bulge=False):
        if form == 'psf':
            self.components += ("\n# Object number: 3\n"
                        +"0) psf                    # object type\n"
                        +f"1) {self.img_size}  {self.img_size}  1 1    # position x, y\n"
                        +f"3) {self.master_cat[obj_id-1][filt.upper()]} 1 # total magnitude\n"
                        +"Z) 0                      # Skip this model in output image?  (yes=1, no=0)\n")
            
        elif form == 'sersic':

            for cat in self.cats:
                if filt.upper() in cat:
                    goodcat = cat
                    break
            
            # get some object parameters
            cat = Table.read(goodcat, format='ascii')
            onum = self.master_cat[f'{filt.upper()}_num'][obj_id-1]
            hlr = float(cat['FWHM_IMAGE'][cat['NUMBER'] == onum]) / 3 # rough conversion from FWHM to half light radius
            ar = float(cat['B_IMAGE'][cat['NUMBER'] == onum] / cat['A_IMAGE'][cat['NUMBER'] == onum])
            pa = float(cat['THETA_IMAGE'][cat['NUMBER'] == onum])

            if bulge == False:
                self.components += ("\n# Object number: 1\n"
                            +"0) sersic                 #  object type\n"
                            +f"1) {self.img_size}  {self.img_size}  1 1    #  position x, y\n"
                            +f"3) {self.master_cat[obj_id-1][filt.upper()]} 1 #  Integrated magnitude\n"
                            +f"4) {hlr}      1             #  R_e (half-light radius)   [pix]\n"
                            +"5) 1.0      1             #  Sersic index n (de Vaucouleurs n=4)\n"
                            +"6) 0.0000      0          #  -----\n"
                            +"7) 0.0000      0          #  -----\n" 
                            +"8) 0.0000      0          #  -----\n"
                            +f"9) {ar}      1             #  axis ratio (b/a)\n"
                            +f"10) {(pa-90)%360}    1             #  position angle (PA) [deg: Up=0, Left=90]\n"
                            +"Z) 0                      #  output option (0 = resid., 1 = Don't subtract)\n")
                
            elif bulge == True:
                self.components += ("\n# Object number: 1\n"
                            +"0) sersic                 #  object type\n"
                            +f"1) {self.img_size}  {self.img_size}  1 1    #  position x, y\n"
                            +f"3) {self.master_cat[obj_id-1][filt.upper()]} 1 #  Integrated magnitude\n"
                            +f"4) {hlr/2}      1             #  R_e (half-light radius)   [pix]\n"
                            +"5) 5.0      1             #  Sersic index n (de Vaucouleurs n=4)\n"
                            +"6) 0.0000      0          #  -----\n"
                            +"7) 0.0000      0          #  -----\n" 
                            +"8) 0.0000      0          #  -----\n"
                            +f"9) {ar}      1             #  axis ratio (b/a)\n"
                            +f"10) {(pa-90)%360}    1             #  position angle (PA) [deg: Up=0, Left=90]\n"
                            +"Z) 0                      #  output option (0 = resid., 1 = Don't subtract)\n")
                
        else:
            print('Please enter a valid component name.')

    ###
    ### add constraints to fit
    ###
    def add_constraints(self, obj_id, filt, constraint):
        self.constraint = True
        self.constraints_list += constraint + '\n'

    ###
    ### create feedme file
    ###
    def run(self, obj_id, filt, num='01', overwrite=False):

        gfile = os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/galfit.{num}')
        feedme_file = os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/galfit.feedme')
        constraint_file = os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/constraints.txt')

        self.galfit_images.append(os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/obj_{obj_id}_{filt}_imgblock{num}.fits'))

        # delete old galfit file
        if os.path.isfile(gfile):
            if overwrite == True:
                os.remove(gfile) 
                os.remove(feedme_file)
            else:
                print(f'{gfile} already exists')
                return os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/obj_{obj_id}_{filt}_imgblock{num}.fits')

        # get params
        params = self.galfit_params(obj_id, filt)

        with open(feedme_file, 'w') as f:
            # write parameters to feedme file
            for param, value in params.items():
                f.write(f'{param}) {value}\n')

            # write components to feedme file
            f.write(self.components)

        if self.constraint == True:
            with open(constraint_file, 'w') as f:
                f.write(self.constraints_list)

        self.components = ''

        os.chdir(os.path.join(self.object_dir, f'obj_{obj_id}/{filt}'))
        os.system('../../../../galfit galfit.feedme')

        return os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/obj_{obj_id}_{filt}_imgblock{num}.fits')
    
    def create_templates(self, obj_id, filt, num='02', oldnum='01', overwrite=False):

        gfile = os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/galfit.{num}')
        feedme_file = os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/galfit.{oldnum}')

        # delete old galfit file
        if os.path.isfile(gfile):
            if overwrite == True:
                os.remove(gfile) 
                os.remove(feedme_file)
            else:
                print(f'{gfile} already exists')
                return os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/obj_{obj_id}_{filt}_imgblock{num}.fits')

        # get old output file and edit a couple things
        feedme_edit = ''
        with open(feedme_file, 'r') as f:
            for line in f:
                if 'B) ' in line:
                    line = line.replace(oldnum, num)
                if 'P) ' in line:
                    line = line.replace(' 0 ', ' 3 ')
                feedme_edit += line
        
        # write to a new feedme file
        with open(os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/galfit.feedme'), 'w') as f:
            f.write(feedme_edit)

        os.chdir(os.path.join(self.object_dir, f'obj_{obj_id}/{filt}'))
        os.system('../../../../galfit galfit.feedme')

        return os.path.join(self.object_dir, f'obj_{obj_id}/{filt}/obj_{obj_id}_{filt}_imgblock{num}.fits')
    
    ###
    ### plot galfit image, template, and residual
    ###
    def plot_galfit(self, im, field, obj_id, dirname, num='01', s=None):
        filt = im.split('_')[-2].upper()

        if s == None:
            s = self.img_size

        # make sure galfit didn't fail
        if not os.path.isfile(im):
            print('NO SUCH FILE!!!', im)
            return

        # get info about fit
        with open(os.path.join(os.path.dirname(im), f'galfit.{num}')) as f:
            chi2_red = f.readlines()[3].split()[3].replace(',', '')

        # get images
        with fits.open(im) as hdul:
            im = hdul[1].data[(self.img_size-s):(self.img_size+s),(self.img_size-s):(self.img_size+s)]
            temp = hdul[2].data[(self.img_size-s):(self.img_size+s),(self.img_size-s):(self.img_size+s)]
            res = hdul[3].data[(self.img_size-s):(self.img_size+s),(self.img_size-s):(self.img_size+s)]
        
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(15,5)

        n = simple_norm(temp,stretch="log",percent=100-1)
        im1 = ax[1].imshow(temp, origin="lower", norm=n, cmap="inferno")
        ax[1].set_title('Template', fontsize=18)
        cbar1 = ax[1].figure.colorbar(im1, ax=ax[1])
        cbar1.ax.set_ylabel('MJy/Sr', rotation=-90, va="bottom")
        
        n = simple_norm(im,stretch="log",percent=100-1)
        im0 = ax[0].imshow(im, origin="lower", norm=n, cmap="inferno")
        ax[0].set_title('Image', fontsize=18)
        cbar0 = ax[0].figure.colorbar(im0, ax=ax[0])
        cbar0.ax.set_ylabel('MJy/Sr', rotation=-90, va="bottom")

        n = simple_norm(res,stretch="log",percent=100-2)
        im2 = ax[2].imshow(res, origin="lower", norm=n, cmap="inferno")
        ax[2].set_title('Residual', fontsize=18)
        cbar2 = ax[2].figure.colorbar(im2, ax=ax[2])
        cbar2.ax.set_ylabel('MJy/Sr', rotation=-90, va="bottom")

        fig.suptitle(f'Field: {field.upper()}, Object: {obj_id}, Filter: {filt}, '+r'$\chi^2_\nu=$'+str(chi2_red), fontsize=25)
        fig.savefig(os.path.join(dirname, f'galfit_{field}_{obj_id}_{filt}.pdf'))

    def plot_all(self, obj_id, field, num='01', s=None):

        if s == None:
            s = self.img_size

        savedir = os.path.join(self.object_dir, f'obj_{obj_id}/sersic_results')

        if not os.path.exists(savedir):
            os.mkdir(savedir)

        for im in tqdm(self.galfit_images, desc='Making plots...'):
            self.plot_galfit(im, field, obj_id, savedir, num='01', s=s)

    ###
    ### Get fit info from galfit files
    ###
    def get_galfit_info(self, im, num='01'):
        galfit_info = {}
        # get info from galfit fit
        with open(os.path.join(os.path.dirname(im), f'galfit.{num}')) as f:
            lines = f.readlines()
            galfit_info['chi2'] = float(lines[3].split()[3].replace(',', ''))
            galfit_info['X'] = float(lines[40].split()[1])
            galfit_info['Y'] = float(lines[40].split()[2])
            galfit_info['mag'] = float(lines[41].split()[1])
            galfit_info['HL_radius'] = float(lines[42].split()[1])
            galfit_info['sersic_n'] = float(lines[43].split()[1])
            galfit_info['axis_ratio'] = float(lines[47].split()[1])
            galfit_info['PA'] = float(lines[48].split()[1])
            return galfit_info

    def plot_apertures(self, im, field, obj_id, dirname, num='01', components=2, frac_hl=0.5):
        filt = im.split('_')[-2].upper()
        galfit_info = self.get_galfit_info(im, num=num)

        # fit elliptical apertures to the data itself
        with fits.open(im) as hdul:
            data = hdul[1].data
            bkg = Background2D(data, [5,5])
            rms = bkg.background_rms_median
            sky = bkg.background_median
        
            # now fit the elliptical annuli
            geometry = EllipseGeometry(x0=galfit_info['X'], y0=galfit_info['Y'], sma=galfit_info['HL_radius'], eps=np.sqrt(1-galfit_info['axis_ratio']**2),pa=((galfit_info['PA']+90) % 360)*np.pi/180)
            ellipse = Ellipse(data-sky, geometry)
            try: #sometimes fit does not converge
                isolist = ellipse.fit_image()
                np.max(isolist.sma[isolist.intens*0==0]) # random thing to see if works
            except:
                return
        
            # convert units
            ps = 0.03 # arcsec per pixel
            zp = 28.0865
            sb = zp - 2.5 * np.log10(isolist.intens/ps**2)
            sb_bkg = zp - 2.5 * np.log10(rms/ps**2)
            # errorbars
            upper = zp - 2.5 * np.log10((isolist.intens+np.sqrt(isolist.int_err**2 + rms**2))/ps**2)
            etop = sb - upper
            lower = zp - 2.5 * np.log10((isolist.intens-np.sqrt(isolist.int_err**2 + rms**2))/ps**2)
            ebot = lower - sb
            bad = np.where(np.isnan(lower))
            if bad != [] : # get rid of nans from log
                ebot[bad] = 5
        
            # fit ellipcical apertures to the model fit
            fit = hdul[2].data
            ellipse_fit = Ellipse(fit-sky, geometry)
            try: # sometimes fit does not converge
                isolist_fit = ellipse_fit.fit_image()
                np.max(isolist_fit.sma[isolist_fit.intens*0==0]) # random thing to see if works
            except:
                return
            sb_fit = zp - 2.5 * np.log10(isolist_fit.intens/ps**2)
            
            # get residual intensity
            # calculate SB but do it once for >0 and once for <0, then plot both individually
            # also normalize it by data (if it doesn't work, try model)
            maxnum = np.min([isolist.intens.shape, isolist_fit.intens.shape])
            intens_res = isolist.intens[:maxnum] - isolist_fit.intens[:maxnum]
            int_err_res = np.sqrt(isolist.int_err[:maxnum]**2 + isolist_fit.int_err[:maxnum]**2)   
            res = intens_res / isolist_fit.intens[:maxnum]
            nonnegative = isolist_fit.intens[:maxnum].copy()
            nonnegative[nonnegative <= 0] = 1
            res_err = int_err_res / nonnegative
            
        
        fig, ax = plt.subplots(2,1, sharex=True)
        fig.subplots_adjust(hspace=0)
        fig.set_size_inches(10,6)
        
        ml2 = MultipleLocator(0.5)
            
        ax[0].errorbar(isolist.sma*ps, sb, yerr=[etop, ebot], fmt='o', capsize=4, ms=5, color='black', label='Data')
        ax[0].plot(isolist_fit.sma*ps, sb_fit, color='blue', label='Sersic')
        ax[0].plot([0, np.max(isolist.sma[sb*0==0]*ps)],[sb_bkg,sb_bkg], '--', color='black', label='1$\sigma$ sky-err')
        
        ax[1].errorbar(isolist.sma[:maxnum]*ps, res, yerr=res_err, fmt='o', capsize=4, ms=5, color='black')
        ax[1].plot([0, np.max(isolist.sma[sb*0==0]*ps)],[0,0], ls='--', color='black')
        
        ax[1].set_xlabel('semimajor axis [arcsec]', fontsize=20)
        ax[0].set_ylabel('SB [mag arcsec$^{-2}$]', fontsize=15)
        ax[1].set_ylabel('residual/model intensity', fontsize=15)
        ax[0].text(galfit_info['HL_radius']*ps/200*frac_hl, sb_bkg-0.4, '$1\sigma$ sky-err', fontsize=15)
        ax[0].text(galfit_info['HL_radius']*ps/3*frac_hl, np.min(sb_fit[sb_fit*0==0])+1.3, 'data \& model', fontsize=15)
        ax[0].set_ylim(np.min(sb_fit[sb_fit*0==0]), np.max(sb_fit[sb_fit*0==0]))
        ax[1].set_ylim(np.min(res[isolist.sma*ps < galfit_info['HL_radius']*ps*frac_hl]), np.max(res[isolist.sma*ps < galfit_info['HL_radius']*ps*frac_hl]))
        ax[0].set_xlim(isolist.sma[1]*ps, galfit_info['HL_radius']*ps*frac_hl)
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].invert_yaxis()
        ax[0].yaxis.set_minor_locator(ml2)

        fig.suptitle(f'Field: {field.upper()}, Object: {obj_id}, Filter: {filt}, '+r'$\chi^2_\nu=$'+str(galfit_info['chi2']), fontsize=25)
        
        for axis in ax:
            axis.tick_params(axis='both', which='major', length=10, width=1, direction='in', labelsize='large', 
                            bottom=True, top=True, left=True, right=True)
            axis.tick_params(axis='both', which='minor', length=5, width=1, direction='in', labelsize='large', 
                            bottom=True, top=True, left=True, right=True)
        
        fig.savefig(os.path.join(dirname, f'galfit_apertures_{field}_{obj_id}_{filt}.pdf'))

    def plot_all_apertures(self, obj_id, field, num='01', frac_hl=1.5):
        # make aperture plots of galfit run 1 results
        dirname = os.path.join(self.object_dir, f'obj_{obj_id}/sersic_results')

        if not os.path.exists(dirname):
            os.mkdir(dirname)

        for im in tqdm(self.galfit_images, desc='Making plots...'):
            self.plot_apertures(im, field, obj_id, dirname, num=num, frac_hl=frac_hl)