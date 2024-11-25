# import numpy as np

import jax.numpy as jnp
from jax.image import scale_and_translate
from simulate import apply_psf
from xscatter import PI, get_wavelen, get_delta_beta_mix

def detect_img(x, in_px, out_Nx, out_px, det_fwhm=None, normalize=False):
    """
    create the detector resampling function
    this should be applied to both signals m and partials dm/dt before the Fisher info calc.
    """
    out_shape = [out_Nx, out_Nx]
    scale = jnp.array([in_px/out_px, in_px/out_px])
    translation = -0.5 * (scale*jnp.array(x.shape) - jnp.array(out_shape))
    total = x.sum()
    x = scale_and_translate(x, out_shape, (0,1), scale, translation, method='linear')
    if normalize:
        x = x * (total / x.sum())
    if det_fwhm is not None:
        x = apply_psf(x, x.shape[0]*out_px, out_px, fwhm=det_fwhm, psf='lorentzian', kernel_width=0.1)
    return x
    

def crlb_2matdecomp_sq(matinfo1, matinfo2, energies, propdists, w, Nx, dx, det_Nx, det_dx, det_fwhm):
    
    kx = np.fft.fftfreq(Nx, d=dx)
    KX, KY = np.meshgrid(kx, kx)
    sinc2D = w**2 * np.sinc(w*KX) * np.sinc(w*KY) / dx**2    # normalize rect amplitude = 1

    def get_Aij(E_i, R_i, mat_j, p_j):
        delta_ij, beta_ij = get_delta_beta_mix(mat_j, E_i, p_j)
        mu_ij = 2 * (2 * PI / get_wavelen(E_i)) * beta_ij
        return mu_ij - (R_i * (KX**2 + KY**2) * delta_ij) 
    
    def get_mi(E_i, R_i, matinfo1, matinfo2, I0):
        mat1, p1, t1 = matinfo1
        mat2, p2, t2 = matinfo2
        Ai1 = get_Aij(E_i, R_i, mat1, p1)
        Ai2 = get_Aij(E_i, R_i, mat2, p2)  
        iFT = jnp.fft.fftshift(jnp.fft.ifft2(sinc2D * (Ai1*t1 + Ai2*t2))).real
        return I0 * jnp.exp(-detect_img(iFT, dx, det_Nx, det_dx, det_fwhm))
        
    def get_dmi_dAij(mi, Aij):
        iFT = np.fft.fftshift(np.fft.ifft2(sinc2D * Aij)).real
        return mi * detect_img(iFT, dx, det_Nx, det_dx, det_fwhm)
    
    F = np.zeros([2,2])
    for i in range(len(energies)):
        E_i = energies[i]
        R_i = R
        mi = get_mi(E_i, R, matinfo1, matinfo2, I0)
        dmi_dA1 = get_dmi_dAij(mi, get_Aij(E_i, R_i, matinfo1[0], matinfo1[1]))
        dmi_dA2 = get_dmi_dAij(mi, get_Aij(E_i, R_i, matinfo2[0], matinfo2[1]))
        F[0,0] += jnp.sum(dmi_dA1**2 / mi)
        F[0,1] += jnp.sum(dmi_dA1 * dmi_dA2 / mi)
        F[1,0] += jnp.sum(dmi_dA2 * dmi_dA1 / mi)
        F[1,1] += jnp.sum(dmi_dA2**2 / mi)
    crlbs = np.linalg.diagonal(np.linalg.inv(F))

    return crlbs




