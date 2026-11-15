"""
CT reconstruction by (filtered) backprojection -- interactive matplotlib widget.

Companion to forward.py, which goes from the object to the sinogram. This script
goes back: from the sinogram to an image of the object.

Four panels side by side:
  1. the true object mu(x, y), for reference;
  2. the sinogram p(theta, rho); the angles used so far are highlighted;
  3. plain backprojection of the first K angles (blurry);
  4. filtered backprojection of the first K angles (sharp).

The slider chooses K, the number of angles used.

Conventions are the same as in forward.py: the ray at angle theta and detector
position rho is the line x*cos(theta) + y*sin(theta) = rho.

Requires: numpy, scipy, scikit-image, matplotlib
Run:      python inverse.py
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

from forward import N, N_ANGLES, SEL, phantom, radon


def ramp_filter(sino):
    """Filter every projection (row) with the ramp filter |f|, in the Fourier domain."""
    n = sino.shape[1]
    n_pad = 2 ** int(np.ceil(np.log2(2 * n)))       # zero-pad to limit wrap-around
    freqs = fftfreq(n_pad, d=2 / n)          # detector spacing is 2/n
    spectrum = fft(sino, n_pad, axis=1) * np.abs(freqs)
    return ifft(spectrum, axis=1).real[:, :n]


def backproject(sino):
    """
    Backproject every projection and return the running sum over angles.

    Output has shape (n_angles, n, n): entry k is the image built from the
    first k + 1 angles. Each projection p(theta, .) is smeared across the image:
    pixel (x, y) receives the value p(theta, x*cos(theta) + y*sin(theta)), i.e. the
    value of the one ray at this angle that passes through that pixel.
    """
    n_angles, n = sino.shape
    rho = (np.arange(n) + 0.5) / n * 2 - 1          # detector positions
    c = rho                                         # pixel centers, same grid as phantom()
    x, y = np.meshgrid(c, -c)
    d_theta = np.pi / n_angles
    layers = np.empty((n_angles, n, n))
    for i, th in enumerate(np.deg2rad(np.arange(n_angles))):
        t = x * np.cos(th) + y * np.sin(th)         # which ray passes through each pixel
        layers[i] = np.interp(t, rho, sino[i], left=0.0, right=0.0) * d_theta
    return np.cumsum(layers, axis=0)


def main():
    mu = phantom()
    sino = radon(mu)
    bp = backproject(sino)                          # plain backprojection
    fbp = backproject(ramp_filter(sino))            # filtered backprojection

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.24, wspace=0.12)
    ax_mu, ax_sino, ax_bp, ax_fbp = axes
    for ax in (ax_mu, ax_bp, ax_fbp):
        ax.set_xticks([])
        ax.set_yticks([])

    ax_mu.imshow(mu, cmap="gray", vmin=0, vmax=1)
    ax_mu.set_title("True object μ(x, y)", loc="left", fontsize=11)

    ax_sino.imshow(sino.T, cmap="gray", aspect="auto", origin="lower",
                   extent=[-0.5, N_ANGLES - 0.5, -1, 1])
    shade = ax_sino.add_patch(Rectangle((0.5, -1), N_ANGLES - 1, 2, color="black", alpha=0.65))  # angles not used yet
    edge = ax_sino.axvline(0.5, color=SEL, lw=1.5)
    ax_sino.set_xlabel("θ (degrees)")
    ax_sino.set_ylabel("ρ (detector position)")
    ax_sino.set_title("Sinogram: the angles used so far", loc="left", fontsize=11)

    # Each image is the average over the K angles used so far (running sum * N_ANGLES / K),
    # so brightness stays comparable as K grows. Display ranges are fixed.
    bp_max = bp[-1].max()
    im_bp = ax_bp.imshow(bp[0], cmap="gray", vmin=0, vmax=bp_max)
    im_fbp = ax_fbp.imshow(fbp[0], cmap="gray", vmin=0, vmax=1)
    ax_bp.set_title("Backprojection (blurry)", loc="left", fontsize=11)
    ax_fbp.set_title("Filtered backprojection (sharp)", loc="left", fontsize=11)

    s_k = Slider(fig.add_axes([0.25, 0.10, 0.50, 0.035]), "Angles used K", 1, N_ANGLES,
                 valinit=1, valstep=1, color=SEL)

    def update(_=None):
        k = int(s_k.val)
        im_bp.set_data(bp[k - 1] * N_ANGLES / k)
        im_fbp.set_data(fbp[k - 1] * N_ANGLES / k)
        shade.set_x(k - 0.5)
        shade.set_width(N_ANGLES - k)
        edge.set_xdata([k - 0.5, k - 0.5])
        ax_sino.set_title(f"Sinogram: angles 0 to {k - 1} used", loc="left", fontsize=11)
        fig.canvas.draw_idle()

    s_k.on_changed(update)

    btn_exit = Button(fig.add_axes([0.46, 0.02, 0.08, 0.05]), "Exit", color="#ddd", hovercolor=SEL)
    btn_exit.on_clicked(lambda _: plt.close(fig))

    update()
    fig._widgets = (s_k, btn_exit)   # keep references alive
    return fig, update


if __name__ == "__main__":
    main()
    plt.show()
