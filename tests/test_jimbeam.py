import pytest
import numpy as np
import matplotlib
# Enforce a non-interactive Matplotlib backend
matplotlib.use('agg')
import matplotlib.pylab as plt  # noqa: E402

from katbeam import JimBeam  # noqa: E402


@pytest.mark.parametrize(
    'name,pol,x,y,freqMHz,value',
    [
        ('MKAT-AA-UHF-JIM-2020', 'HH', 0, 0, 800, 1.0),
        ('MKAT-AA-L-JIM-2020', 'HH', 0, 0, 1420, 0.999774),
        ('MKAT-AA-UHF-JIM-2020', 'VV', 0, 1, 800, 0.6600966),
        ('MKAT-AA-L-JIM-2020', 'VV', 0, 1, 1420, 0.2062726),
        ('MKAT-AA-UHF-JIM-2020', 'I', 1, 0, 800, 0.4077328),
        ('MKAT-AA-L-JIM-2020', 'I', 1, 0, 1420, 0.02575332),
    ]
)
def test_sample_beam_values(name, pol, x, y, freqMHz, value):
    beam = JimBeam(name)
    pattern = getattr(beam, pol)
    assert pattern(x, y, freqMHz) == pytest.approx(value)


def showbeam(beam, freqMHz=1000, pol='HH', beamextent=10.):
    margin = np.linspace(-beamextent / 2., beamextent / 2., 128)
    x, y = np.meshgrid(margin, margin)
    pattern = getattr(beam, pol)
    beampixels = pattern(x, y, freqMHz)
    fig, ax = plt.subplots()
    ax.imshow(beampixels, extent=[-beamextent / 2., beamextent / 2.,
                                  -beamextent / 2., beamextent / 2.])
    ax.set_title('{} pol beam\nfor {} at {:d}MHz'.format(pol, beam.name, freqMHz))
    return fig


@pytest.mark.mpl_image_compare(remove_text=True, filename='UHF_800_HH_10.png')
def test_UHF_beam_image():
    beam = JimBeam('MKAT-AA-UHF-JIM-2020')
    return showbeam(beam, 800, 'HH', 10.)


@pytest.mark.mpl_image_compare(remove_text=True, filename='L_1420_VV_5.png')
def test_L_beam_image():
    beam = JimBeam('MKAT-AA-L-JIM-2020')
    return showbeam(beam, 1420, 'VV', 5.)
