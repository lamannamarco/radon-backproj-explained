"""
CT and the Radon transform -- interactive matplotlib widget.

Three panels side by side:
  1. the object mu(x, y) with a family of parallel X-rays at angle theta
     (the selected ray, detector rho, is highlighted);
  2. the projection p(theta, .) at that angle: one number per detector;
  3. the sinogram p(theta, rho): theta on the x-axis, rho on the y-axis, one column per angle.

Sliders choose the angle theta and the detector rho.

Note: a real scanner measures intensities I, and p = -ln(I / I0) is computed
from them (Beer-Lambert). Here we skip that step and compute p directly as the
line integral of mu, which is what -ln(I / I0) equals.

Conventions: a ray at angle theta and detector position rho is the line
    x*cos(theta) + y*sin(theta) = rho,
so at theta = 0 the rays are vertical and p(0, rho) is a column sum.

Requires: numpy, scipy, scikit-image, matplotlib
Run:      python forward.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy.ndimage import map_coordinates
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

N = 128            # image is N x N pixels, and there are N detectors
N_ANGLES = 180     # angles 0, 1, ..., 179 degrees
RAY = "#1D9E75"    # color of the ray family
SEL = "#D85A30"    # color of the selected ray / detector

def phantom(n=N):
    """Shepp-Logan phantom from scikit-image, resampled to n x n on [-1, 1]^2.

    Row 0 is the top (y = +1), matching the pixel convention used in radon().
    """
    return resize(shepp_logan_phantom(), (n, n), anti_aliasing=True)


def radon(mu, n_angles=N_ANGLES):
    """
    Sinogram p[theta, rho]: the integral of mu along every ray.

    For each angle the ray through detector rho is sampled at N points
    s in [-1, 1] along its direction and the samples are summed (times ds).
    """
    n = mu.shape[0]
    rho = (np.arange(n) + 0.5) / n * 2 - 1          # detector positions
    s = (np.arange(n) + 0.5) / n * 2 - 1            # position along each ray
    R, S = np.meshgrid(rho, s, indexing="ij")       # (rho, s)
    sino = np.zeros((n_angles, n))
    for i, th in enumerate(np.deg2rad(np.arange(n_angles))):
        c, sn = np.cos(th), np.sin(th)
        x = R * c - S * sn                          # points on the rays
        y = R * sn + S * c
        col = (x + 1) / 2 * n - 0.5                 # world -> pixel coordinates
        row = (1 - y) / 2 * n - 0.5
        vals = map_coordinates(mu, [row, col], order=1, mode="constant", cval=0.0)
        sino[i] = vals.sum(axis=1) * (2 / n)        # sum along s, times ds
    return sino


def main():
    mu = phantom()
    sino = radon(mu)
    rho_axis = (np.arange(N) + 0.5) / N * 2 - 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5.4))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.24, wspace=0.28)

    # 1 -- object with rays
    L = 1.3
    ax1.set_facecolor("#111")
    ax1.imshow(mu, cmap="gray", vmin=0, vmax=1, extent=[-1, 1, -1, 1])
    ax1.set_xlim(-L, L)
    ax1.set_ylim(-L, L)
    ax1.set_aspect("equal")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Input: object μ(x, y), rays at angle θ", loc="left", fontsize=11)
    ray_lines = [ax1.plot([], [], color=RAY, lw=1, alpha=0.7)[0] for _ in range(12)]
    sel_line, = ax1.plot([], [], color=SEL, lw=3)
    det_line, = ax1.plot([], [], color="#bbb", lw=6, solid_capstyle="round")
    p_text = ax1.text(0.97, 0.96, "", transform=ax1.transAxes, ha="right", va="top",
                      color="#ddd", fontsize=10)
    ax1.text(0.03, 0.03, "detector row", transform=ax1.transAxes, color="#ddd", fontsize=9)

    # 2 -- one projection
    prof_line, = ax2.plot(rho_axis, sino[0], color=RAY, lw=2)
    prof_dot, = ax2.plot([], [], "o", color=SEL, ms=8)
    prof_stem, = ax2.plot([], [], color=SEL, lw=1, alpha=0.6)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(0, sino.max() * 1.08)
    ax2.set_xlabel("ρ (detector position)")
    ax2.set_ylabel("p = ∫ μ dl  (ray sum)")
    ax2.set_title("One angle: p(θ, ρ) across all detectors", loc="left", fontsize=11)
    ax2.grid(alpha=0.3)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    # 3 -- sinogram
    ax3.imshow(sino.T, cmap="gray", aspect="auto", origin="lower",
               extent=[-0.5, N_ANGLES - 0.5, -1, 1])
    sin_col = ax3.axvline(0, color=SEL, lw=1, alpha=0.7)
    sin_dot, = ax3.plot([], [], "o", mfc="none", mec=SEL, mew=2, ms=11)
    ax3.set_xlabel("θ (degrees)")
    ax3.set_ylabel("ρ (detector position)")
    ax3.set_title("Output: sinogram p(θ, ρ), one column per angle", loc="left", fontsize=11)

    # sliders
    s_th = Slider(fig.add_axes([0.10, 0.10, 0.35, 0.035]), "Angle θ (°)", 0, N_ANGLES - 1,
                  valinit=0, valstep=1, color=SEL)
    s_rh = Slider(fig.add_axes([0.58, 0.10, 0.35, 0.035]), "Detector index", 0, N - 1,
                  valinit=N // 2, valstep=1, color=SEL)

    def update(_=None):
        t = int(s_th.val)
        j = int(s_rh.val)
        th = np.deg2rad(t)
        c, sn = np.cos(th), np.sin(th)

        def ray(rho, ext=1.15):
            return [rho * c + ext * sn, rho * c - ext * sn], [rho * sn - ext * c, rho * sn + ext * c]

        for k, ln in enumerate(ray_lines):
            ln.set_data(*ray(-0.92 + k * 0.167))
        sel_line.set_data(*ray(rho_axis[j]))
        d = -1.22                                   # detector sits at s = d
        det_line.set_data([-c - d * sn, c - d * sn], [-sn + d * c, sn + d * c])
        p_text.set_text(f"p = {sino[t, j]:.2f}")

        prof_line.set_ydata(sino[t])
        prof_dot.set_data([rho_axis[j]], [sino[t, j]])
        prof_stem.set_data([rho_axis[j]] * 2, [0, sino[t, j]])

        sin_col.set_xdata([t, t])
        sin_dot.set_data([t], [rho_axis[j]])
        fig.canvas.draw_idle()

    s_th.on_changed(update)
    s_rh.on_changed(update)

    # exit button (bottom center)
    btn_exit = Button(fig.add_axes([0.46, 0.02, 0.08, 0.05]), "Exit", color="#ddd", hovercolor=SEL)
    btn_exit.on_clicked(lambda _: plt.close(fig))
    update()
    fig._widgets = (s_th, s_rh, btn_exit)      # keep references alive
    return fig, update


if __name__ == "__main__":
    main()
    plt.show()
