import numpy as np

def crlb_2mat_sq(Nx, dx, w, ts, R, mus, deltas):
    t1, t2 = ts
    kx = np.fft.fftfreq(Nx, dx)
    KX, KY = jnp.meshgrid(kx, ky)
    F = (w**2) * np.sinc(w*KX) * np.sinc(w*KY)

    # Define the material-dependent matrices
    # TODO : reshape Rs so you can vary those instead of mus, deltas
    K2 = 4 * PI**2 * (KX**2 + KY**2)
    A11 = mus[0,0] - (R * K2 * deltas[0,0])
    A12 = mus[1,0] - (R * K2 * deltas[1,0])
    A21 = mus[0,1] - (R * K2 * deltas[0,1])
    A22 = mus[1,1] - (R * K2 * deltas[1,1])

    # Signals
    m1 = np.exp(np.fft.ifft2(t1*F*A11 + t2*F*A12)).real    
    m2 = np.exp(np.fft.ifft2(t1*F*A21 + t2*F*A22)).real

    # Partial derivatives
    dm11 = m1 * np.fft.ifft2(F*A11).real
    dm12 = m1 * np.fft.ifft2(F*A12).real
    dm21 = m2 * np.fft.ifft2(F*A21).real
    dm22 = m2 * np.fft.ifft2(F*A22).real    

    # Fisher information
    I11 = np.sum((dm11**2/m1)   + (dm21**2/m2))
    I12 = np.sum((dm11*dm12/m1) + (dm21*dm22/m2))
    I21 = np.sum((dm12*dm11/m1) + (dm22*dm21/m2))
    I22 = np.sum((dm12**2/m1)   + (dm22**2/m2))
    I = np.array([[I11, I12], [I21, I22]])

    CRLBs = np.linalg.inv(I).diagonal()
    
    return CRLBs