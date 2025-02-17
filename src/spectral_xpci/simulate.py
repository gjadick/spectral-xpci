import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
import chromatix.functional as cx
from chromatix.ops import init_plane_resample


h  = 6.62607015e-34           # Planck constant, J/Hz
c = 299792458.0               # speed of light, m/s
J_eV = 1.602176565e-19        # J per eV conversion
PI = jnp.pi


def get_wavelen(energy):  
    """energy in keV -> wavelength in m"""
    return 1e-3*h*c/(energy*J_eV)


def simulate_projection(beta_proj, dn_proj, phantom_px, phantom_N,
                        det_shape, det_px, det_fwhm, 
                        energy, R, I0,
                        det_psf='lorentzian', n_medium=1, N_pad=100, key=jax.random.PRNGKey(42)):
    """
    cailey added new input parameter phantom_N 01/18/25, delete this note after merge
    beta_proj :  ∫ beta(x,y,z) dz
    dn_proj :  ∫ delta(x,y,z) dz
    """

    assert (beta_proj.shape == dn_proj.shape)
    phantom_fov = phantom_px*phantom_N
    det_fov = det_px*det_shape[0]
    
    field = cx.plane_wave(
        shape = beta_proj.shape, 
        dx = phantom_px,
        spectrum = get_wavelen(energy),
        spectral_density = 1.0,
    )    
    field = field / field.intensity.max()**0.5 
    cval = field.intensity.max()

    exit_field = cx.thin_sample(field, beta_proj[None, ..., None, None], dn_proj[None, ..., None, None], 1.0)
    
    det_field = cx.transfer_propagate(exit_field, R, n_medium, N_pad, cval=cval, mode='same')

    det_resample_func = init_plane_resample(det_shape, (det_px, det_px), resampling_method='linear')
    img = det_resample_func(det_field.intensity.squeeze()[...,None,None], field.dx.ravel()[:1])[...,0,0]
    img = img / (det_px/phantom_px)**2  # normalize to new pixel size

    if I0 is not None:
        img = jax.random.poisson(key, I0*img, img.shape) / I0

    if det_psf is not None:
        img = apply_psf(img, det_fov, det_px, psf=det_psf, fwhm=det_fwhm, kernel_width=0.1)

    return img

    

def lorentzian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Lorentzian kernel.
    x, y : 1D arrays
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-max of the Lorentzian (units must match x,y)
    """
    gamma = fwhm/2
    X, Y = jnp.meshgrid(x, y)
    kernel = gamma / (2 * PI * (X**2 + Y**2 + gamma**2)**1.5)
    if normalize:
        kernel = kernel / jnp.sum(kernel)
    return kernel


def apply_psf(img, FOV, dx, fwhm=None, kernel_width=1, psf='lorentzian'):
    """ Apply a Lorentzian PSF to an image."""
    if psf.lower() == 'lorentzian':
        assert fwhm is not None
        
        small_FOV = kernel_width * FOV   # reduce kernel size to improve convolution time
        x = jnp.arange(-small_FOV, small_FOV, dx) + dx/2
        psf = lorentzian2D(x, x, fwhm)
        
        img_pad = jnp.pad(img, psf.shape, constant_values=img[0,0])    # pad img to account for fillvalue = 0
        img_nonideal_pad = convolve2d(img_pad, psf, mode='same')
        img_nonideal = img_nonideal_pad[psf.shape[0]:-psf.shape[0], psf.shape[1]:-psf.shape[1]]
        
    else:
        print("only psf='lorentzian' supported")
        
    return img_nonideal



def xpci_2matdecomp(imgs, s, R, mus, deltas):
    img1, img2 = imgs

    # Compute 2D frequency components k^2 = kx^2 + ky^2, shape ~ [Nx, Ny].
    # Then, ravel the 2D array so that we can parallelize on each (kx, ky) coordinate.
    kx = jnp.fft.fftfreq(img1.shape[0], s)
    ky = jnp.fft.fftfreq(img1.shape[1], s)
    KX, KY = jnp.meshgrid(kx, ky)
    K2 = 4 * PI**2 * (KX**2 + KY**2).ravel()

    # Define the material-dependent matrices, raveled for parallelization.
    A11 = mus[0,0] - (R * K2 * deltas[0,0])
    A12 = mus[1,0] - (R * K2 * deltas[1,0])
    A21 = mus[0,1] - (R * K2 * deltas[0,1])
    A22 = mus[1,1] - (R * K2 * deltas[1,1])
    
    # Define the frequency-domain image vectors, raveled for parallelization.
    G1 = jnp.fft.fft2(-jnp.log(img1)).ravel()
    G2 = jnp.fft.fft2(-jnp.log(img2)).ravel()

    # Solve for the frequency-domain thickness images.
    @jax.jit
    def solve_T_1coord(a11, a12, a21, a22, g1, g2):
        A = jnp.array([[a11, a12],
                       [a21, a22]])
        G = jnp.array([g1, g2]).T
        T, _, _, _ = jnp.linalg.lstsq(A, G)
        return T
    solve_T = jax.jit(jax.vmap(solve_T_1coord))
    T1, T2 = solve_T(A11, A12, A21, A22, G1, G2).transpose()
    
    # Inverse Fourier Transform to recover the material decomposition images.
    t1 = jnp.real(jnp.fft.ifft2((T1.reshape(img1.shape))))
    t2 = jnp.real(jnp.fft.ifft2((T2.reshape(img1.shape))))

    return jnp.array([t1, t2])

