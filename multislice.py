#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 08:59:10 2025

@author: gjadick

Helpful things for organizing the AAPM 2025 multislice abstract:
    - `Material` class 
    - phantom-making functions
    - multislice simulation function (like `simulate_projection` in src)

TODO: docstrings

"""

import numpy as np
import jax.numpy as jnp
import jax

from src.spectral_xpci.xscatter import get_delta_beta_mix
from src.spectral_xpci.simulate import get_wavelen, apply_psf
from chromatix.ops import init_plane_resample
import chromatix.functional as cx



class Material:
    def __init__(self, name, matcomp, density):
        self.name = name
        self.matcomp = matcomp
        self.density = density
    def delta_beta(self, energy):
        delta, beta = get_delta_beta_mix(self.matcomp, np.atleast_1d(energy), self.density)
        return delta.squeeze(), beta.squeeze()


def make_fiber_phantom(N, Nz, dx, fiber_width, energy, fiber_material, background_material, height_fraction=0.8):

    fov = N*dx
    fiber_height = height_fraction*fov
    coords = np.abs(np.linspace(-fov/2, fov/2, N))
    X, Y = np.meshgrid(coords, coords)

    obj = np.zeros([Nz, N, N], dtype=np.uint8)
    for z in range(Nz):
        mask = np.where((X <= fiber_width/2) & (Y <= fiber_height/2))  
        obj[z][mask] = 1

    matdict = {0: background_material, 1: fiber_material}
    obj_delta = jnp.zeros(obj.shape)
    obj_beta = jnp.zeros(obj.shape)
    for mat_id in matdict:
        d, b = matdict[mat_id].delta_beta(energy)
        obj_delta = obj_delta.at[obj == mat_id].set(d)
        obj_beta = obj_beta.at[obj == mat_id].set(b)

    return obj_delta, obj_beta


    
def simulate_multislice(obj_beta, obj_delta, phantom_dx, phantom_dz, phantom_N, z_slices, det_shape, det_dx, det_fwhm, energy, propdist, I0,
                       det_psf='lorentzian', N_pad=100, n_medium=1, n_avg=1, key=jax.random.PRNGKey(3),
                        dzlist=None
):

    phantom_shape = jnp.array([z_slices, phantom_N, phantom_N])

    field = cx.plane_wave(
        shape = phantom_shape[1:], 
        dx = phantom_dx,
        spectrum = get_wavelen(energy),
        spectral_density = 1.0,
    )

    field = field / field.intensity.max()**0.5 
    cval = field.intensity.max()

    # modulate field thru sample
    # exit_field = cx.multislice_thick_sample(field, obj_beta, obj_delta, n_avg, phantom_dz, N_pad)
    field_k = field
    if dzlist is not None:
        all_dz = dzlist
    else:
        all_dz = np.ones(z_slices) * phantom_dz
        
    for k in range(z_slices):
        # field_k = cx.thin_sample(field_k, obj_beta[k][None, ..., None, None], obj_delta[k][None, ..., None, None], phantom_dz)
        # field_k = cx.transfer_propagate(field_k, phantom_dz, n_medium, N_pad, cval=cval, mode='same')
        field_k = cx.thin_sample(field_k, obj_beta[k][None, ..., None, None], obj_delta[k][None, ..., None, None], all_dz[k])
        field_k = cx.transfer_propagate(field_k, all_dz[k], n_medium, N_pad, cval=cval, mode='same')
    exit_field = field_k
    
    # propagate thru free space to the detector
    det_field = cx.transfer_propagate(exit_field, propdist, n_medium, N_pad, cval=cval, mode='same')
    
    # resample from phantom resolution -> detector resolution (pixel sizes don't necessarily match)
    det_resample_func = init_plane_resample(det_shape, (det_dx, det_dx), resampling_method='linear')
    img_ms = det_resample_func(det_field.intensity.squeeze()[...,None,None], field.dx.ravel()[:1])[...,0,0]
    img_ms = img_ms / (det_dx/phantom_dx)**2  # normalize to new pixel size

    # add noise and PSF blur
    if I0 is not None:
        img_ms = jax.random.poisson(key, I0*img_ms, img_ms.shape) / I0 # noise

    if det_psf is not None:
        det_fov = det_shape[0]*det_dx
        img_ms = apply_psf(img_ms, det_fov, det_dx, psf=det_psf, fwhm=det_fwhm, kernel_width=0.1)

    return img_ms
