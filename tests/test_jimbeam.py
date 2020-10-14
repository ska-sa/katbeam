import pytest
import numpy as np
import matplotlib
# Enforce a non-interactive Matplotlib backend
matplotlib.use('agg')
import matplotlib.pylab as plt  # noqa: E402
from matplotlib.testing.decorators import image_comparison, remove_ticks_and_titles  # noqa: E402

from katbeam import JimBeam  # noqa: E402


def showbeam(beam, freqMHz=1000, pol='H', beamextent=10.):
    margin = np.linspace(-beamextent/2., beamextent/2., 128)
    x, y = np.meshgrid(margin, margin)
    if pol == 'H':
        beampixels = beam.HH(x, y, freqMHz)
    elif pol == 'V':
        beampixels = beam.VV(x, y, freqMHz)
    else:
        beampixels = beam.I(x, y, freqMHz)
        pol = 'I'
    plt.clf()
    plt.imshow(beampixels, extent=[-beamextent/2, beamextent/2,
                                   -beamextent/2, beamextent/2])
    plt.title('%s pol beam\nfor %s at %dMHz' % (pol, beam.name, freqMHz))


GENERATE_IMAGES = False


@pytest.mark.parametrize(
    'name,freqMHz,pol,beamextent,baseline_images',
    [
        ('MKAT-AA-UHF-JIM-2020', 800, 'H', 10, ['UHF_800_H_10']),
        ('MKAT-AA-UHF-JIM-2020', 800, 'V', 10, ['UHF_800_V_10']),
        ('MKAT-AA-UHF-JIM-2020', 800, 'I', 10, ['UHF_800_I_10']),
        ('MKAT-AA-L-JIM-2020', 1420, 'H', 5, ['L_1420_H_5']),
    ]
)
@image_comparison(baseline_images=None, remove_text=True, extensions=['png'])
def test_jimbeam(name, freqMHz, pol, beamextent, baseline_images):
    beam = JimBeam(name)
    showbeam(beam, freqMHz, pol, beamextent)
    if GENERATE_IMAGES:
        remove_ticks_and_titles(plt.gcf())
        plt.savefig('baseline_images/{}/{}.png'
                    .format(__name__, baseline_images[0]), dpi=100)
