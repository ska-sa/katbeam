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
   be approximated by averaging several of these individual antenna beams with
   respective antenna pointing errors inserted.
d) Depending on the usecase it may be necessary to do reference pointing (or
   use another technique) to remove the antenna pointing errors during the
   observation in order to use a beam model successfully.

.. _link: https://books.google.co.za/books?id=Jg6hCwAAQBAJ
