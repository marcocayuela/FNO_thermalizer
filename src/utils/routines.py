import numpy as np
import torch


def vorticity(velocity_field, dx=None, dy=None):

    nx = velocity_field.shape[-3]
    ny = velocity_field.shape[-2]
    u = velocity_field[...,0]
    v = velocity_field[...,1]

    if dx is None:
        dx = 2*np.pi/nx
    if dy is None:
        dy = 2*np.pi/ny

    du_dy = np.gradient(u, dy, axis=-1)
    dv_dx = np.gradient(v, dx, axis=-2)
    return dv_dx - du_dy




def energy_spectrum(velocity_field):

    U = velocity_field[...,0]
    V = velocity_field[...,1]

    T, nx, ny = U.shape

    U_fluct = U - U.mean(axis=(1,2), keepdims=True)
    V_fluct = V - V.mean(axis=(1,2), keepdims=True)

    fft_U = np.fft.fft2(U_fluct, axes=(1,2))
    fft_V = np.fft.fft2(V_fluct, axes=(1,2))

    # énergie spectrale 2D
    E_hat = 0.5 * (np.abs(fft_U)**2 + np.abs(fft_V)**2) / (nx*ny)**2

    # moyenne temporelle
    E_hat = np.mean(E_hat, axis=0)

    kx = np.fft.fftfreq(nx)*nx
    ky = np.fft.fftfreq(ny)*ny
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k = np.sqrt(kx**2 + ky**2)

    k_max = int(np.max(k))
    E_k = np.zeros(k_max+1)

    for i in range(k_max+1):
        mask = (k >= i) & (k < i+1)
        E_k[i] = np.sum(E_hat[mask])

    return E_k, np.arange(k_max+1)


def energy_spectrum_torch(velocity_field):


    U = velocity_field[..., 0]  # (T, nx, ny)
    V = velocity_field[..., 1]

    T, nx, ny = U.shape

    # Fluctuations (retrait de la moyenne spatiale)
    U_fluct = U - U.mean(dim=(1, 2), keepdim=True)
    V_fluct = V - V.mean(dim=(1, 2), keepdim=True)

    # FFT 2D (sur axes spatiaux)
    fft_U = torch.fft.fft2(U_fluct, dim=(1, 2))
    fft_V = torch.fft.fft2(V_fluct, dim=(1, 2))

    # Énergie spectrale 2D, normalisée
    E_hat = 0.5 * (fft_U.abs()**2 + fft_V.abs()**2) / (nx * ny)**2

    # Moyenne temporelle
    E_hat = E_hat.mean(dim=0)  # (nx, ny)

    # Grille des nombres d'onde
    kx = torch.fft.fftfreq(nx, device=velocity_field.device) * nx
    ky = torch.fft.fftfreq(ny, device=velocity_field.device) * ny
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    k = torch.sqrt(kx**2 + ky**2)

    k_max = int(k.max().item())
    E_k = torch.zeros(k_max + 1, device=velocity_field.device)

    # Binning par anneau (intégration radiale)
    for i in range(k_max + 1):
        mask = (k >= i) & (k < i + 1)
        E_k[i] = E_hat[mask].sum()

    k_bins = torch.arange(k_max + 1, device=velocity_field.device)
    return E_k, k_bins