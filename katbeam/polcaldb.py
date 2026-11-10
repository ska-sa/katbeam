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
import json

# --------------------------------------------------------------------------------------------------
# --- CLASS :  FitsBeam
# --------------------------------------------------------------------------------------------------

#Note according to Ludwig astropy replaced pyfm (which did not account for UT1 leap seconds) in 
# September 2021, resulting in improved geometric delay calibration stability over long tracks.

class PolCalDB(object):
    """MeerKAT Polarization Calibration Database handler class.

    This class facilitates the reading of precalculated polarization 
    calibration solutions and prepares them for use with a particular 
    dataset. 

    Parameters
    ----------
    delaypath : str
        Path to directory that contains processed noise diode information 
        extracted from delay calibration files

    solutionfilename : str
        Filename of file that contains noise diode to sky solutions

    Raises
    ------
    ValueError
        If `filename` is of unknown format

    Example usage
    -------------

    .. code:: python

      from katbeam import PolCalDB

      polcal=PolCalDB(delaypath='/home/mattieu/polcal/ndphase',solutionfilename='/home/mattieu/polcal/ndsky/L_best_2026_nd_sky.npz')
      freqMHz,hvphase=polcal.hvphase(delay_cbid=1616735560,refant='m060')
      plot(freqMHz,hvphase)
    """

    def __init__(self, delaypath, solutionfilename):
        self.delaypath = delaypath
        self.solutionfilename = solutionfilename

    def hvphase(self, delay_cbid, refant, freqMHz=None,verbose=False,return_components=False):
        """Determines the hvphase solution that is applicable to a given dataset.

        Parameters
        ----------
        delay_cbid : int
            cbid of delaycal file
        refant : str
            antenna name used as reference antenna
        freqMHz: array of floats
            Frequency channels of where to supply hvphase
            default None provides what is available
        verbose: bool
            print out comments

        Returns
        -------
        freqMHz : array of floats
            Frequency of channels
        hvphase : array of floats
            hvphase in degrees for specified reference antenna
        """

        nd=np.load('%s/%d_nd_phase.npz'%(self.delaypath,delay_cbid), allow_pickle=True)
        soln=np.load(self.solutionfilename, allow_pickle=True)
        ndfreqMHz=nd['freqMHz']
        ndphase=nd['ndsphase'][nd['antnames'].tolist().index(refant),:]
        self.solutionreceivers=soln['receivers']
        if refant not in soln['receivers'].tolist() or refant not in nd['antnames']:
            raise ValueError("refant %s not available"%(refant))
        if soln['receivers'].tolist()[refant]!=nd['receivers'][nd['antnames'].tolist().index(refant)]:
            possible_refants=sorted([str(antname) for antname in nd['antnames'] if soln['receivers'].tolist()[antname]==nd['receivers'][nd['antnames'].tolist().index(antname)]])
            raise ValueError("refant %s receiver mismatch delaycal: %s, solution: %s, choose another refant from"%(refant,soln['receivers'].tolist()[refant],nd['receivers'][nd['antnames'].tolist().index(refant)]),possible_refants)
        if freqMHz is None:
            freqMHz=ndfreqMHz
        elif not np.allclose(freqMHz,ndfreqMHz):
            ndphase=np.interp(freqMHz,ndfreqMHz,np.unwrap(ndphase))
        if 'best_hvphase_nd_smooth' in soln:#some older obsolete format in '.npz' to be abolished
            nd_sky_phase=np.interp(freqMHz,soln['freqMHz1024'],soln['best_hvphase_nd_smooth'].tolist()[refant])
        else:#newer file format '*_nd_sky.npz'
            nd_sky_phase=np.interp(freqMHz,soln['freqMHz'],soln['nd_sky'].tolist()[refant])
             
        hvphase=ndphase+nd_sky_phase
        if verbose:
            print('%s receivername: %s'%(refant,soln['receivers'].tolist()[refant]))
        if return_components:
            return freqMHz,hvphase,ndphase,nd_sky_phase
        return freqMHz,hvphase
