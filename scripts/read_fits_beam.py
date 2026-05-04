#example script shows how to read fits or numpy MeerKAT beam files and plot 
#interpolated azimuth or elevation cut profiles through the primary beam
#mattieu@sarao.ac.za
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import katpoint
import scipy
import numpy as np

#note Ibeam is in power, HHbeam etc is voltage
def read_beam_file(filename):
    if filename.endswith('.npz'): #read npz beam file
        d=np.load(filename)
        beam=d['beam']
        #Note npz file polarisation ordering: HH,HV,VH,VV specified in d['pols']
        HHbeam,HVbeam,VHbeam,VVbeam=beam[0,-1,:,:,:],beam[1,-1,:,:,:],beam[2,-1,:,:,:],beam[3,-1,:,:,:]
        margin_l_deg,margin_m_deg=d['margin_deg'],d['margin_deg']
        freqMHz=d['freq_MHz']
    elif filename.endswith('.fits'): #read fits beam file
        d=pyfits.open(filename)[0]
        margin_l_deg=d.header['CDELT1']*(np.arange(1,d.data.shape[-1]+1)-d.header['CRPIX1'])+d.header['CRVAL1']
        margin_m_deg=d.header['CDELT2']*(np.arange(1,d.data.shape[-2]+1)-d.header['CRPIX2'])+d.header['CRVAL2']
        freq=d.header['CDELT3']*(np.arange(1,d.data.shape[-3]+1)-d.header['CRPIX3'])+d.header['CRVAL3']
        freqMHz=freq/1e6
        #Note fits file polarisation ordering: VV,HH,VH,HV
        VVbeam=d.data[0,0,:,:,:]+1j*d.data[1,0,:,:,:]
        HHbeam=d.data[0,1,:,:,:]+1j*d.data[1,1,:,:,:]
        VHbeam=d.data[0,2,:,:,:]+1j*d.data[1,2,:,:,:]
        HVbeam=d.data[0,3,:,:,:]+1j*d.data[1,3,:,:,:]
    else:
        raise Exception('Unknown file format')
    Ibeam=0.5*(np.abs(HHbeam)**2+np.abs(VVbeam)**2+np.abs(HVbeam)**2+np.abs(VHbeam)**2)
    margin_l=margin_l_deg*np.pi/180
    margin_m=margin_m_deg*np.pi/180
    ll,mm=np.meshgrid(margin_l,margin_m)
    return freqMHz,ll,mm,HHbeam,HVbeam,VHbeam,VVbeam,Ibeam

def plotbeam_az_el_cuts(filename,ichan=0,el_target_deg=60,cut='az'):
    az_target_deg=0
    freqMHz,ll,mm,HHbeam,HVbeam,VHbeam,VVbeam,Ibeam=read_beam_file(filename)

    #make grid of cuts in az or el due to az/el offsets from scanned target
    if cut=='az':#azimuth cut sample locations
        az_sample_deg=np.linspace(0,3,1024)/np.cos(el_target_deg*np.pi/180)+az_target_deg
        el_sample_deg=np.zeros(1024)+el_target_deg
    else:#elevation cut sample locations
        az_sample_deg=np.zeros(1024)/np.cos(el_target_deg*np.pi/180)+az_target_deg
        el_sample_deg=np.linspace(0,3,1024)+el_target_deg

    #calculate corresponding direction cosine l,m coordinates in beam that match az/el cuts
    l_sample, m_sample=katpoint.projection.sphere_to_plane_sin(az_target_deg*np.pi/180, el_target_deg*np.pi/180, az_sample_deg*np.pi/180, el_sample_deg*np.pi/180)
    
    #compile points to interpolate
    xi=(l_sample, m_sample)
    method='cubic'
    points=np.transpose(np.array([ll.reshape(-1),mm.reshape(-1)]))

    #interpolate beam at az/el
    cbeam=HHbeam[ichan,:,:].reshape(-1)
    iHHbeam=scipy.interpolate.griddata(points, np.real(cbeam), xi, method=method)+1j*scipy.interpolate.griddata(points, np.imag(cbeam), xi, method=method)
    cbeam=VVbeam[ichan,:,:].reshape(-1)
    iVVbeam=scipy.interpolate.griddata(points, np.real(cbeam), xi, method=method)+1j*scipy.interpolate.griddata(points, np.imag(cbeam), xi, method=method)

    #plot to current axis
    if cut=='az':
        plt.plot((az_sample_deg-az_target_deg)*np.cos(el_target_deg*np.pi/180),np.abs(iHHbeam)**2,'--',label='model HH, el %d'%(el_target_deg))
        plt.plot((az_sample_deg-az_target_deg)*np.cos(el_target_deg*np.pi/180),np.abs(iVVbeam)**2,'-',label='model VV, el %d'%(el_target_deg))
        plt.xlabel('az_offset*cos(target_el)')
    else:
        plt.plot((el_sample_deg-el_target_deg),np.abs(iHHbeam)**2,'--',label='model HH')
        plt.plot((el_sample_deg-el_target_deg),np.abs(iVVbeam)**2,'-',label='model VV')
        plt.xlabel('el_offset')
    plt.yscale('log')
    plt.grid('both')
    plt.ylabel('Power')
    plt.title(filename+'\n%d MHz'%np.round(freqMHz[ichan]))
    plt.legend()

if __name__ == "__main__":
    plt.figure()
    plotbeam_az_el_cuts('MK+S0_sym.fits',ichan=87,el_target_deg=0,cut='az')
    plotbeam_az_el_cuts('MK+S0_sym.fits',ichan=87,el_target_deg=60,cut='az')
    plt.figure()
    plotbeam_az_el_cuts('MK+S0_sym.fits',ichan=87,cut='el')


