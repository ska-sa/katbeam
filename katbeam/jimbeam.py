################################################################################
# Copyright (c) 2020, National Research Foundation (SARAO)
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

# --------------------------------------------------------------------------------------------------
# --- CLASS :  JimBeam
# --------------------------------------------------------------------------------------------------


class JimBeam(object):
    """MeerKAT simplified primary beam models for L and UHF bands

    A cosine aperture taper (Essential Radio Astronomy, Condon & Ransom, 2016,
    page 83, link_) is used as a simplified model of the co-polarisation primary beams.
    While the sidelobe level accuracy may be coincidental, the model attains a good fit
    to measurements for the mainlobe region. The model is parameterised by measured
    frequency dependent pointing, and frequency dependent full width half maximum
    beam widths (FWHM). The MeerKAT beams are measured using holography techniques,
    and an averaged result at 60 degrees elevation is used here to determine the
    frequency dependent parameter values. The pointing errors are determined in
    the aperture plane using standard phase fitting techniques, while the FWHM
    values are measured in the beam plane along axis-aligned cuts through the beam
    centers.

    Notes
    ------
    a) This model is a simplification.
    b) The actual beam varies per antenna, and depends on environmental factors.
    c) Since per-antenna pointing errors during an observation often exceed 1 arc
       minute, the nett 'imaging primary beam' will be slightly broader, and could
       be approximated by averaging several individual antenna beams with
       respective antenna pointing errors inserted.
    d) Depending on the usecase it may be necessary to do reference pointing (or
       use another technique) to remove the antenna pointing errors during the
       observation in order to use a beam model successfully.

    Parameters
    ----------
    name : str
        Name of model, must be either 'MKAT-AA-L-JIM-2020' or 'MKAT-AA-UHF-JIM-2020'

    Request
    -------
    As a user, please email the author (mattieu@ska.ac.za) with details about
    your usecase requirements. This may influence future releases. A general
    description, what extent of the beams are needed, pixelation, frequency
    resolution, and accuracy requirements are of interest.

    Example usage
    -------------

    .. code:: python

      import matplotlib.pylab as plt
      from katbeam import JimBeam

      def showbeam(beam,freqMHz=1000,pol='H',beamextent=10.):
          margin=np.linspace(-beamextent/2.,beamextent/2.,128)
          x,y=np.meshgrid(margin,margin)
          if pol=='H':
              beampixels=beam.HH(x,y,freqMHz)
          elif pol=='V':
              beampixels=beam.VV(x,y,freqMHz)
          else:
              beampixels=beam.I(x,y,freqMHz)
              pol='I'
          plt.clf()
          plt.imshow(beampixels,extent=[-beamextent/2,beamextent/2,-beamextent/2,beamextent/2])
          plt.title('%s pol beam\nfor %s at %dMHz'%(pol,beam.name,freqMHz))
          plt.xlabel('deg')
          plt.ylabel('deg')

      uhfbeam=JimBeam('MKAT-AA-UHF-JIM-2020')
      showbeam(uhfbeam,800,'H',10)

    .. _link: https://books.google.co.za/books?id=Jg6hCwAAQBAJ
    """

    def __init__(self, name='MKAT-AA-L-JIM-2020'):
        self.name = name
        knownmodels={'MKAT-AA-L-JIM-2020':
     '''freq, Hx squint, Hy squint, Vx squint, Vy squint, Hx fwhm, Hy fwhm, Vx fwhm, Vy fwhm
        MHz, arcmin, arcmin, arcmin, arcmin, arcmin, arcmin, arcmin, arcmin
        900, 0.00, 0.88, -0.00, 0.72, 97.98, 100.37, 96.41, 101.89
        950, -0.01, 0.50, -0.03, 0.41, 92.58, 94.70, 90.72, 96.26
        1000, 0.05, 0.20, 0.02, 0.38, 87.89, 89.30, 85.59, 91.74
        1050, -0.02, 0.31, 0.02, -0.12, 83.39, 84.55, 80.96, 86.98
        1100, -0.03, -0.03, 0.01, -0.23, 79.08, 80.06, 76.76, 82.25
        1150, -0.03, 0.09, -0.02, -0.29, 74.79, 76.14, 72.92, 78.09
        1200, -0.05, 0.00, -0.00, -0.36, 70.81, 72.78, 69.70, 73.86
        1250, -0.05, -0.03, 0.02, -0.35, 67.30, 69.69, 66.79, 70.31
        1300, 0.06, 0.02, 0.01, -0.58, 64.20, 67.18, 64.30, 67.11
        1350, 0.18, -0.07, 0.03, -0.42, 61.67, 64.80, 62.15, 64.32
        1400, -0.43, -0.07, 0.03, -0.07, 59.58, 62.70, 60.11, 62.26
        1450, -1.27, -0.12, -0.00, 1.07, 57.92, 60.78, 58.31, 60.45
        1500, -0.97, -0.23, -0.02, 1.14, 56.91, 58.82, 56.61, 59.31
        1550, -0.40, -0.21, -0.02, 0.74, 56.08, 57.00, 54.83, 58.31
        1600, -0.04, -0.29, -0.04, 0.49, 55.35, 55.24, 53.18, 57.44
        1650, 0.22, -0.18, -0.04, 1.07, 55.22, 53.52, 51.58, 56.90''',
        'MKAT-AA-UHF-JIM-2020':
     '''freq, Hx squint, Hy squint, Vx squint, Vy squint, Hx fwhm, Hy fwhm, Vx fwhm, Vy fwhm
        MHz, arcmin, arcmin, arcmin, arcmin, arcmin, arcmin, arcmin, arcmin
        550, -0.15, 2.46, -0.08, 0.40, 159.05, 165.92, 157.72, 165.00
        600, -0.00, 0.93, -0.06, 1.22, 147.75, 153.60, 146.55, 155.25
        650, -0.02, 1.18, 0.00, 0.43, 135.71, 139.61, 133.70, 141.99
        700, 0.06, 0.14, -0.01, 0.03, 124.92, 128.66, 122.75, 130.42
        750, 0.07, -0.13, -0.02, -0.16, 115.48, 118.02, 113.03, 121.01
        800, 0.08, -0.01, 0.02, -0.81, 106.78, 110.47, 105.56, 111.64
        850, -0.15, -0.61, 0.01, -0.58, 99.25, 103.60, 99.38, 103.52
        900, -1.12, -0.61, -0.01, -0.10, 93.46, 97.88, 93.96, 97.68
        950, -1.50, -0.80, -0.09, 0.15, 89.67, 93.10, 89.45, 93.52
        1000, -0.58, -0.83, -0.14, -0.47, 87.38, 88.87, 85.55, 90.83
        1050, 0.32, -0.43, -0.08, -0.72, 86.10, 85.16, 82.32, 88.15'''}

        if name in knownmodels:
            table=np.array([line.split(',') for line in knownmodels[name].split('\n')[2:]],dtype='float')
        else:
            print('Error: %s model is unknown'%name)
        self.squintlist=table[:,1:5].T/60#arcmin to degrees [4,nfreq], where 4 refers to Hx,Hy,Vx,Vy components
        self.fwhmlist=table[:,5:9].T/60#arcmin to degrees [4,nfreq], where 4 refers to Hx,Hy,Vx,Vy components
        self.freqMHzlist=table[:,0]

    #r is normalised such that the half power point occurs at r=0.5: jim(0)=1.0 and jim(0.5)=sqrt(0.5)
    def jim(self,r):
        rr=r*1.18896478#achieves jim(0.5)=sqrt(0.5)
        return np.cos(np.pi*rr)/(1.-4.*(rr**2))

    #margin=np.linspace(-beamextent/2.,beamextent/2.,128)
    #x,y=np.meshgrid(margin,margin)
    #x,y in degrees
    def _HH(self,x,y,squint,fwhm):
        return self.jim(np.sqrt(((x-squint[0])/fwhm[0])**2+((y-squint[1])/fwhm[1])**2))

    def _VV(self,x,y,squint,fwhm):
        return self.jim(np.sqrt(((x-squint[2])/fwhm[2])**2+((y-squint[3])/fwhm[3])**2))

    def _I(self,x,y,squint,fwhm):
        H=self._HH(x,y,squint,fwhm)
        V=self._VV(x,y,squint,fwhm)
        I=0.5*(np.abs(H)**2+np.abs(V)**2)
        return I

    def interp_squint_fwhm(self,freqMHz):
        squint=[np.interp(freqMHz,self.freqMHzlist,lst) for lst in self.squintlist]
        fwhm=[np.interp(freqMHz,self.freqMHzlist,lst) for lst in self.fwhmlist]
        return squint,fwhm

    def HH(self,x,y,freqMHz):
        '''
        Calculates the H co-polarised beam at coordinates provided

        Parameters
        ----------
        x,y : arrays specifying coordinates where beam is sampled, in degrees
        freqMHz : frequency, in MHz
        '''
        squint,fwhm=self.interp_squint_fwhm(freqMHz)
        return self._HH(x,y,squint,fwhm)

    def VV(self,x,y,freqMHz):
        '''
        Calculates the V co-polarised beam at coordinates provided

        Parameters
        ----------
        x,y : arrays specifying coordinates where beam is sampled, in degrees
        freqMHz : frequency, in MHz
        '''
        squint,fwhm=self.interp_squint_fwhm(freqMHz)
        return self._VV(x,y,squint,fwhm)

    def I(self,x,y,freqMHz):
        '''
        Calculates the Stokes I beam at coordinates provided

        Parameters
        ----------
        x,y : arrays specifying coordinates where beam is sampled, in degrees
        freqMHz : frequency, in MHz
        '''
        squint,fwhm=self.interp_squint_fwhm(freqMHz)
        return self._I(x,y,squint,fwhm)
