################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import numpy as np
import matplotlib.pylab as plt
import astropy.io.fits as pyfits
from scipy.interpolate import RegularGridInterpolator

# --------------------------------------------------------------------------------------------------
# --- CLASS :  FitsBeam
# --------------------------------------------------------------------------------------------------

class FitsBeam(object):
    """MeerKAT FITS file handler class.

    This class facilitates the loading of a MeerKAT .fits or .npz primary beam file 
    and sample it in a required manner for use or display.

    Parameters
    ----------
    filename : str
        Name and path of fits file to load.

    interpmethod : str
        Interpolation method to use, either 'linear' or 'cubic'

    Raises
    ------
    ValueError
        If `filename` is of unknown format

    Example usage
    -------------

    .. code:: python

      from katbeam import FitsBeam

      fb=FitsBeam('beam_mavg_L.fits')
      fb.showbeam(freqMHz=1500,pol='HV',component='abs',colmap='viridis')

    """

    def __init__(self, filename, interpmethod = 'linear'):
        self.filename = filename
        self.interpmethod = interpmethod
        self.freqMHz, self.l_deg, self.m_deg, self.HHbeam, self.HVbeam, self.VHbeam, self.VVbeam, self.Ibeam = FitsBeam.read_beam_file(filename)
        self.reHHinterp = RegularGridInterpolator((self.freqMHz, self.m_deg, self.l_deg), np.real(self.HHbeam), method=self.interpmethod,bounds_error=False)
        self.imHHinterp = RegularGridInterpolator((self.freqMHz, self.m_deg, self.l_deg), np.imag(self.HHbeam), method=self.interpmethod,bounds_error=False)
        self.reHVinterp = RegularGridInterpolator((self.freqMHz, self.m_deg, self.l_deg), np.real(self.HVbeam), method=self.interpmethod,bounds_error=False)
        self.imHVinterp = RegularGridInterpolator((self.freqMHz, self.m_deg, self.l_deg), np.imag(self.HVbeam), method=self.interpmethod,bounds_error=False)
        self.reVHinterp = RegularGridInterpolator((self.freqMHz, self.m_deg, self.l_deg), np.real(self.VHbeam), method=self.interpmethod,bounds_error=False)
        self.imVHinterp = RegularGridInterpolator((self.freqMHz, self.m_deg, self.l_deg), np.imag(self.VHbeam), method=self.interpmethod,bounds_error=False)
        self.reVVinterp = RegularGridInterpolator((self.freqMHz, self.m_deg, self.l_deg), np.real(self.VVbeam), method=self.interpmethod,bounds_error=False)
        self.imVVinterp = RegularGridInterpolator((self.freqMHz, self.m_deg, self.l_deg), np.imag(self.VVbeam), method=self.interpmethod,bounds_error=False)
        self.absIinterp = RegularGridInterpolator((self.freqMHz, self.m_deg, self.l_deg), self.Ibeam, method=self.interpmethod,bounds_error=False)

    @staticmethod
    def read_beam_file(filename):
        if filename.lower().endswith('.npz'): #read npz beam file
            d=np.load(filename)
            beam=d['beam']
            #Note npz file polarisation ordering: HH,HV,VH,VV specified in d['pols']
            HHbeam,HVbeam,VHbeam,VVbeam=beam[0,-1,:,:,:],beam[1,-1,:,:,:],beam[2,-1,:,:,:],beam[3,-1,:,:,:]#dimensions: pol,ant,freq,m,l
            l_deg,m_deg=d['margin_deg'],d['margin_deg']
            freqMHz=d['freq_MHz']
        elif filename.lower().endswith('.fits'): #read fits beam file that is FORTRAN order on disk, C-order in Python array memory (swaps interpretation of order of dimensions)
            d=pyfits.open(filename)[0]
            l_deg=d.header['CDELT1']*(np.arange(1,d.data.shape[-1]+1)-d.header['CRPIX1'])+d.header['CRVAL1']#first FORTRAN order dimension on disk = last C-order dimension in Python array
            m_deg=d.header['CDELT2']*(np.arange(1,d.data.shape[-2]+1)-d.header['CRPIX2'])+d.header['CRVAL2']
            freq=d.header['CDELT3']*(np.arange(1,d.data.shape[-3]+1)-d.header['CRPIX3'])+d.header['CRVAL3']
            freqMHz=freq/1e6
            #Note fits file polarisation ordering: VV,HH,VH,HV differs from npz file pol ordering
            VVbeam=d.data[0,0,:,:,:]+1j*d.data[1,0,:,:,:]#dimensions: complex,pol,freq,m,l
            HHbeam=d.data[0,1,:,:,:]+1j*d.data[1,1,:,:,:]
            VHbeam=d.data[0,2,:,:,:]+1j*d.data[1,2,:,:,:]
            HVbeam=d.data[0,3,:,:,:]+1j*d.data[1,3,:,:,:]
        else:
            raise Exception('Unknown file format')
        Ibeam=0.5*(np.abs(HHbeam)**2+np.abs(VVbeam)**2+np.abs(HVbeam)**2+np.abs(VHbeam)**2)
        return freqMHz,l_deg,m_deg,HHbeam,HVbeam,VHbeam,VVbeam,Ibeam

    @staticmethod
    def broadcast(l_deg, m_deg, freqMHz):
        """Calculate broadcasted arrays from margins
        """
        ffreqMHz,mm_deg,ll_deg=np.meshgrid(freqMHz,m_deg,l_deg,indexing='ij') #for clarity when working with 3 dimensions avoid default 'xy' indexing which swaps first two dimensions
        return ll_deg,mm_deg,ffreqMHz

    def HH(self, ll_deg, mm_deg, freqMHz):
        """Calculate the H co-polarised beam at the provided coordinates.

        Parameters
        ----------
        ll_deg, mm_deg : arrays of floats of matching shape
            Coordinates where beam must be sampled, in degrees (SIN projection)
        freqMHz : float
            Frequency, in MHz

        Returns
        -------
        HH : array of floats with same shape as the input arrays
            The H co-polarised beam (H feed response due to H polarised signal)
        """
        ffreqMHz = freqMHz+np.zeros(ll_deg.shape)#broadcasts to shape of freqMHz to that of input array
        pts=np.array(list(zip(ffreqMHz.reshape(-1),mm_deg.reshape(-1),ll_deg.reshape(-1))))
        return (self.reHHinterp(pts)+1j*self.imHHinterp(pts)).reshape(ll_deg.shape)

    def HV(self, ll_deg, mm_deg, freqMHz):
        """Calculate the H cross-polarised beam at the provided coordinates.

        Parameters
        ----------
        ll_deg, mm_deg : arrays of floats of matching shape
            Coordinates where beam must be sampled, in degrees (SIN projection)
        freqMHz : float
            Frequency, in MHz

        Returns
        -------
        HV : array of floats with same shape as the input arrays
            The H cross-polarised beam (H feed response due to V polarised signal)
        """
        ffreqMHz = freqMHz+np.zeros(ll_deg.shape)#broadcasts to shape of freqMHz to that of input array
        pts=np.array(list(zip(ffreqMHz.reshape(-1),mm_deg.reshape(-1),ll_deg.reshape(-1))))
        return (self.reHVinterp(pts)+1j*self.imHVinterp(pts)).reshape(ll_deg.shape)

    def VH(self, ll_deg, mm_deg, freqMHz):
        """Calculate the V cross-polarised beam at the provided coordinates.

        Parameters
        ----------
        ll_deg, mm_deg : arrays of floats of matching shape
            Coordinates where beam must be sampled, in degrees (SIN projection)
        freqMHz : float
            Frequency, in MHz

        Returns
        -------
        VH : array of floats with same shape as the input arrays
            The V cross-polarised beam (V feed response due to H polarised signal)
        """
        ffreqMHz = freqMHz+np.zeros(ll_deg.shape)#broadcasts to shape of freqMHz to that of input array
        pts=np.array(list(zip(ffreqMHz.reshape(-1),mm_deg.reshape(-1),ll_deg.reshape(-1))))
        return (self.reVHinterp(pts)+1j*self.imVHinterp(pts)).reshape(ll_deg.shape)

    def VV(self, ll_deg, mm_deg, freqMHz):
        """Calculate the V co-polarised beam at the provided coordinates.

        Parameters
        ----------
        ll_deg, mm_deg : arrays of floats of matching shape
            Coordinates where beam must be sampled, in degrees (SIN projection)
        freqMHz : float
            Frequency, in MHz

        Returns
        -------
        VV : array of floats with same shape as the input arrays
            The V co-polarised beam (V feed response due to V polarised signal)
        """
        ffreqMHz = freqMHz+np.zeros(ll_deg.shape)#broadcasts to shape of freqMHz to that of input array
        pts=np.array(list(zip(ffreqMHz.reshape(-1),mm_deg.reshape(-1),ll_deg.reshape(-1))))
        return (self.reVVinterp(pts)+1j*self.imVVinterp(pts)).reshape(ll_deg.shape)

    def I(self, ll_deg, mm_deg, freqMHz):
        """Calculate the Stokes I beam at the provided coordinates.

        Parameters
        ----------
        ll_deg, mm_deg : arrays of floats of matching shape
            Coordinates where beam must be sampled, in degrees (SIN projection)
        freqMHz : float
            Frequency, in MHz

        Returns
        -------
        I : array of floats with same shape as the input arrays
            The Stokes I beam (combined feed response due to unpolarised signal)
        """
        ffreqMHz = freqMHz+np.zeros(ll_deg.shape)#broadcasts to shape of freqMHz to that of input array
        pts=np.array(list(zip(ffreqMHz.reshape(-1),mm_deg.reshape(-1),ll_deg.reshape(-1))))
        return (self.absIinterp(pts)).reshape(ll_deg.shape)

    def showbeam(self, freqMHz=1500, pol='HV',component='abs', beamextent=None,beamgridsize=128,doclf=True,docolorbar=True,colmap='jet',clim=None):
        """Displays a view of the beam on the sky as seen from the antenna mount.

        Parameters
        ----------
        ll_deg, mm_deg : arrays of floats of matching shape
            Coordinates where beam must be sampled, in degrees (SIN projection)
        freqMHz : float
            Frequency, in MHz
        pol : string
            Polarisation, one of 'HH','HV','VH','VV', or 'I', 
            where pol[0] refers to feed and pol[1] refers to the beacon where applicable.
        component : string
            Component to be shown, one of 'abs','pow','phase','angle','real','imag'
        beamextent : float or None
            Total extent of beam in projected degrees
        beamgridsize : int
            Gridsize to be used for display
        doclf : bool
            Clears figure first if True
        docolorbar : bool
            Show colorbar if True
        colmap : string
            Name of colormap to use
        clim : list or None
            Limits to use to scale image colorscale, None is autoscale, or lower and upper limits in list

        """
        if beamextent is None:
            beamextent=np.max(np.abs(self.l_deg))*2
        l_deg=np.linspace(-beamextent/2.,beamextent/2.,beamgridsize)
        m_deg=np.linspace(-beamextent/2.,beamextent/2.,beamgridsize)
        ll_deg,mm_deg=np.meshgrid(l_deg, m_deg)

        if pol=='HH':
            beampixels=self.HH(ll_deg,mm_deg,freqMHz)
        elif pol=='HV':
            beampixels=self.HV(ll_deg,mm_deg,freqMHz)
        elif pol=='VH':
            beampixels=self.VH(ll_deg,mm_deg,freqMHz)
        elif pol=='VV':
            beampixels=self.VV(ll_deg,mm_deg,freqMHz)
        elif pol=='I':
            beampixels=(self.I(ll_deg,mm_deg,freqMHz))
        else:
            print('Unknown pol %s. Choose from HH, HV, VH, VV, I, '%pol)
            return
        if component in ['abs','mag']:
            values=np.abs(beampixels)
        elif component in ['pow']:
            if pol in ['I']:
                values=10*np.log10(np.abs(beampixels))
            else:
                values=20*np.log10(np.abs(beampixels))
        elif component in ['arg','phase','angle','ang']:
            values=180/np.pi*np.angle(beampixels)
        elif component in ['real','re']:
            values=np.real(beampixels)
        elif component in ['imag','im']:
            values=np.imag(beampixels)

        if doclf:
            plt.clf()
        plt.imshow(values,extent=[-beamextent/2,beamextent/2,-beamextent/2,beamextent/2],cmap=colmap,vmin=None if clim is None else clim[0],vmax=None if clim is None else clim[1],origin='lower')
        plt.title('%s\n%s %s pol beam at %dMHz'%(self.filename.split('/')[-1],component,pol,freqMHz))
        plt.xlabel('l [deg]')
        plt.ylabel('m [deg]')
        if docolorbar:
            plt.colorbar(label=component)
