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


def simulate_projection(beta_proj, dn_proj, phantom_px, 
                        det_shape, det_px, det_fwhm, 
                        energy, R, I0,
                        det_psf='lorentzian', n_medium=1, N_pad=100, key=jax.random.PRNGKey(42)):
    """
    beta_proj :  ∫ beta(x,y,z) dz
    dn_proj :  ∫ delta(x,y,z) dz
    """

    assert (beta_proj.shape == dn_proj.shape)
    phantom_fov = beta_proj.shape[0] * phantom_px
    
    field = cx.plane_wave(
        shape = beta_proj.shape, 
        dx = phantom_px,
        spectrum = get_wavelen(energy),
        spectral_density = 1.0,
    )
    field = field / field.intensity.max()**0.5  # normalize
    cval = field.intensity.max()

    exit_field = cx.thin_sample(field, beta_proj[None, ..., None, None], dn_proj[None, ..., None, None], 1.0)
    det_field = cx.transfer_propagate(exit_field, R, n_medium, N_pad, cval=cval, mode='same')

    det_resample_func = init_plane_resample(det_shape, (det_px, det_px), resampling_method='linear')
    img = det_resample_func(det_field.intensity.squeeze()[...,None,None], field.dx.ravel()[:1])[...,0,0]
    img /= img.ravel()[0] 

    if I0 is not None:
        img = jax.random.poisson(key, I0*img, img.shape) / I0

    if det_psf is not None:
        img = apply_psf(img, det_px, psf=det_psf, fwhm=det_fwhm, kernel_width=0.1)

    return img

def gaussian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Gaussian kernel.
    x, y : 1D arrays
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-maximum of the Gaussian (units must match x, y)
    normalize : bool
        If True, normalize the kernel to sum to 1
    """
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    X, Y = jnp.meshgrid(x, y)
    kernel = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    if normalize:
        kernel = kernel / jnp.sum(kernel)
    return kernel


def lorentzian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Lorentzian kernel.
    x, y : 1D arrays
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-max of the Lorentzian (units must match x,y)
    normalize : bool
        If True, normalize the kernel to sum to 1
    """
    gamma = fwhm/2
    X, Y = jnp.meshgrid(x, y)
    kernel = gamma / (2 * PI * (X**2 + Y**2 + gamma**2)**1.5)
    if normalize:
        kernel = kernel / jnp.sum(kernel)
    return kernel


def apply_psf(img, dx, psf='lorentzian', fwhm='pixel', kernel_width=0.2):
    """ 
    Apply a point spread function (PSF) to a 2D image via convolution.

    Parameters
    ----------
    img : 2D array (jnp.ndarray)
        The input image to which the PSF will be applied.
    dx : float
        Pixel size in physical units (e.g., mm or µm).
    psf : {'lorentzian', 'gaussian'}, optional
        The type of PSF to apply. Default is 'lorentzian'.
    fwhm : float or {'pixel', None}, optional
        Full width at half maximum of the PSF, in the same units as dx.
        - If 'pixel', sets FWHM to dx (i.e., 1 pixel wide).
        - If None, no PSF is applied (function returns `img` unchanged).
    kernel_width : float, optional
        Fraction of the image field-of-view to use as the PSF kernel width.
        A smaller value reduces computational cost. Default is 0.2.

    Returns
    -------
    img_nonideal : 2D array (jnp.ndarray)
        The image convolved with the PSF kernel, simulating the effect 
        of limited resolution due to the imaging system.

    Notes
    -----
    - Assumes a square image (`img.shape[0] == img.shape[1]`).
    - The kernel is computed over a reduced field-of-view (`kernel_width * FOV`)
      for computational efficiency.
    - Pads the input image with constant edge values before convolution to 
      avoid edge artifacts.
    """

    # Handle spetial FWHM options
    if fwhm is None:
        return img
    elif fwhm == 'pixel':
        fwhm = dx   

    # Check if PSF format is supported
    psf = psf.lower()
    assert psf in ('lorentzian', 'gaussian')

    # Compute reduced FOV for kernel grid for efficiency
    small_FOV = kernel_width * max(img.shape) * dx
    x = jnp.arange(-small_FOV, small_FOV, dx) + dx

    # Generate the kernel (normalized by default)
    if psf == 'lorentzian':
        kernel = lorentzian2D(x, x, fwhm)
    elif psf == 'gaussian':
        kernel = gaussian2D(x, x, fwhm)

    # Compute padding (half kernel size on each size to account for fillvalue = 0)
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    img_pad = jnp.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode='edge')

    # Apply convolution
    img_nonideal = convolve2d(img_pad, kernel, mode='valid')

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

