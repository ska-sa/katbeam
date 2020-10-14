katbeam
=======

Primary beam model library for the MeerKAT project, providing functionality to
compute simplified beam patterns of MeerKAT antennas.


JimBeam class
=============

MeerKAT simplified primary beam models for L and UHF bands

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

Notes:

a) This model is a simplification.
b) The actual beam varies per antenna, and depends on environmental factors.
c) Since per-antenna pointing errors during an observation often exceed 1 arc
   minute, the nett 'imaging primary beam' will be slightly broader, and could
   be approximated by averaging several individual antenna beams with
   respective antenna pointing errors inserted.
d) Depending on the usecase it may be necessary to do reference pointing (or
   use another technique) to remove the antenna pointing errors during the
   observation in order to use a beam model successfully.

Request:

As a user, please email the author (mattieu@ska.ac.za) with details about
your usecase requirements. This may influence future releases. A general
description, what extent of the beams are needed, pixelation, frequency
resolution, and accuracy requirements are of interest.

Example usage:

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
