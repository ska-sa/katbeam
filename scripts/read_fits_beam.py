#example script shows how to read fits or numpy MeerKAT beam files and plot 
#interpolated azimuth or elevation cut profiles through the primary beam
#mattieu@sarao.ac.za
import matplotlib.pyplot as plt
import katpoint
import scipy
import numpy as np

def plot_az_el_cuts(fb,fMHz=1500,az_target_deg=0,el_target_deg=60):
    
    for cut in ['az','el']:
        #make grid of cuts in az or el due to az/el offsets from scanned target
        if cut=='az':#azimuth cut sample locations
            az_sample_deg=np.linspace(-3,3,1024)/np.cos(el_target_deg*np.pi/180)+az_target_deg
            el_sample_deg=np.zeros(1024)+el_target_deg
        else:#elevation cut sample locations
            az_sample_deg=np.zeros(1024)/np.cos(el_target_deg*np.pi/180)+az_target_deg
            el_sample_deg=np.linspace(-3,3,1024)+el_target_deg

        #calculate corresponding direction cosine l,m coordinates in beam that match az/el cuts
        l_sample, m_sample=katpoint.projection.sphere_to_plane_sin(az_target_deg*np.pi/180, el_target_deg*np.pi/180, az_sample_deg*np.pi/180, el_sample_deg*np.pi/180)
        if True:
            iHHbeam=fb.HH(l_sample*180/np.pi, m_sample*180/np.pi,fMHz)
            iVVbeam=fb.VV(l_sample*180/np.pi, m_sample*180/np.pi,fMHz)
        else:#alternate method if not interpolating across frequency
            ichan=np.argmin(np.abs(fb.freqMHz-fMHz))
            #compile points to interpolate
            xi=(l_sample, m_sample)
            method='cubic'
            ll,mm=np.meshgrid(fb.l_deg*np.pi/180,fb.m_deg*np.pi/180)
            points=np.transpose(np.array([ll.reshape(-1),mm.reshape(-1)]))

            #interpolate beam at az/el
            cbeam=fb.HHbeam[ichan,:,:].reshape(-1)
            iHHbeam=scipy.interpolate.griddata(points, np.real(cbeam), xi, method=method)+1j*scipy.interpolate.griddata(points, np.imag(cbeam), xi, method=method)
            cbeam=fb.VVbeam[ichan,:,:].reshape(-1)
            iVVbeam=scipy.interpolate.griddata(points, np.real(cbeam), xi, method=method)+1j*scipy.interpolate.griddata(points, np.imag(cbeam), xi, method=method)

        #plot to current axis
        if cut=='az':
            plt.subplot(2,1,1)
            plt.plot((az_sample_deg-az_target_deg)*np.cos(el_target_deg*np.pi/180),np.abs(iHHbeam)**2,'--',label='model HH, el %d'%(el_target_deg))
            plt.plot((az_sample_deg-az_target_deg)*np.cos(el_target_deg*np.pi/180),np.abs(iVVbeam)**2,'-',label='model VV, el %d'%(el_target_deg))
            plt.xlabel('(Azimuth offset)*cos(target_el) [deg]')
        else:
            plt.subplot(2,1,2)
            plt.plot((el_sample_deg-el_target_deg),np.abs(iHHbeam)**2,'--',label='model HH')
            plt.plot((el_sample_deg-el_target_deg),np.abs(iVVbeam)**2,'-',label='model VV')
            plt.xlabel('Elevation offset [deg]')
        plt.yscale('log')
        plt.grid('both')
        plt.ylabel('Power')
        plt.legend()
    plt.suptitle(fb.filename+' %.1f'%fMHz)

if __name__ == "__main__":
    from katbeam import FitsBeam

    fb=FitsBeam('beam_mavg_L.fits')
    plt.figure()
    fb.showbeam(1500,'HH')
    plt.figure()
    plot_az_el_cuts(fb)


