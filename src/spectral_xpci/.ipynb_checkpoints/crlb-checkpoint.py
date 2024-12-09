# import numpy as np

import jax.numpy as jnp
from jax.image import scale_and_translate
from src.spectral_xpci.simulate import PI, apply_psf
from src.spectral_xpci.xscatter import get_wavelen, get_delta_beta_mix


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


def crlb_2matdecomp_sq(deltas, mus, Rs, Ts, w, Nx, dx, det_Nx, det_dx, det_fwhm, I0=1.0):
    """
    mus, deltas, Rs:
        axis=0: material j
        axis=1: acquisition i (energy OR propdist)
    Ts: basis material thicknesses
    w: rect width
    Nx, dx: object dimensions (probably upsampled for simulation purposes)
    det_Nx, det_dx, det_fwhm: detector parameters
    """

    kx = jnp.fft.fftfreq(Nx, d=dx)
    KX, KY = jnp.meshgrid(kx, kx)
    sinc2D = w**2 * jnp.sinc(w*KX) * jnp.sinc(w*KY) / dx**2    # normalize rect amplitude = 1

    def get_A(i, j):
        return mus[j,i] - (Rs[j,i] * (KX**2 + KY**2) * deltas[j,i]) 
        
    def get_m(i):
        iFT = jnp.fft.fftshift(jnp.fft.ifft2(sinc2D * (get_A(i,0)*Ts[0] + get_A(i,1)*Ts[1]))).real
        return I0 * jnp.exp(-detect_img(iFT, dx, det_Nx, det_dx, det_fwhm))

    def get_dm_dA(mi, Aij):
        iFT = jnp.fft.fftshift(jnp.fft.ifft2(sinc2D * Aij)).real
        return mi * detect_img(iFT, dx, det_Nx, det_dx, det_fwhm)
    
    F = jnp.zeros([2,2])
    for i in range(2):
        mi = get_m(i)
        dmi_dA1 = get_dm_dA(mi, get_A(i,0))
        dmi_dA2 = get_dm_dA(mi, get_A(i,1))
        F = F.at[0,0].set(F[0,0] + jnp.sum(dmi_dA1**2 / mi))
        F = F.at[0,1].set(F[0,1] + jnp.sum(dmi_dA1 * dmi_dA2 / mi))
        F = F.at[1,0].set(F[1,0] + jnp.sum(dmi_dA2 * dmi_dA1 / mi))
        F = F.at[1,1].set(F[1,1] + jnp.sum(dmi_dA2**2 / mi))
    crlbs = jnp.linalg.diagonal(jnp.linalg.inv(F))

    return crlbs


def get_acquisition_info(Es, Rs, matcomp1, density1, t1, matcomp2, density2, t2):

    energies = jnp.atleast_1d(Es)
    propdists = jnp.atleast_1d(Rs)
    thicknesses = jnp.atleast_1d([t1, t2])
    
    Ni = max(energies.size, propdists.size)
    Nj = 2  
    assert Ni == 2  # only 2x2 currently supported!
    
    if energies.size < Ni:
        energies = jnp.tile(energies, Ni)
    elif propdists.size < Ni: 
        propdists = jnp.tile(propdists, Ni)
    
    mat1_dn, mat1_beta = get_delta_beta_mix(matcomp1, energies, density1)
    mat2_dn, mat2_beta = get_delta_beta_mix(matcomp2, energies, density2)

    deltas = jnp.array([mat1_dn, mat2_dn])
    mus = 2 * (2 * PI / get_wavelen(energies)) * jnp.array([mat1_beta, mat2_beta])
    propdists = jnp.tile(propdists, [Nj, 1])

    return deltas, mus, propdists, thicknesses



