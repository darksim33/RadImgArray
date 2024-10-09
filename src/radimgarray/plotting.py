from __future__ import annotations
import numpy as np
import PyQt6
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def show_image(image: np.ndarray, title: str = ""):
    """
    Display a 2D image.

    Args:
        image: np.ndarray
            2D image to display.
        title: str
            Title of the plot.
    """
    if len(image.shape) < 3:
        plt.imshow(image, cmap="gray")
        plt.show()
    elif len(image.shape) == 3:
        plot_3d_image(image)
    elif len(image.shape) == 4:
        plot_4d_image(image)


def plot_3d_image(image: np.ndarray):
    """
    Display a 3D image.

    Args:
        image: np.ndarray
            3D image to display.
    """

    # Set the backend to Qt5Agg
    matplotlib.use("Qt5Agg")

    # Prepare plot
    init_idx = image.shape[-1] // 2
    init_img = image[..., init_idx]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # slider spacing
    current_axis = ax.imshow(init_img, cmap="gray")
    ax.axis("off")
    # Add slider
    ax_slider = plt.axes((0.8, 0.275, 0.03, 0.55))  # (0.25, 0.1, 0.65, 0.03)
    slider = Slider(
        ax=ax_slider,
        label="Slice",  # "Image Index",
        valmin=0,
        valmax=image.shape[-1] - 1,
        valinit=init_idx,
        valstep=1,
        orientation="vertical",
    )

    def update(val):
        idx = int(slider.val)
        current_axis.set_data(image[..., idx])
        # plt.draw()
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def plot_4d_image(image: np.ndarray):

    # Set the backend to Qt5Agg
    matplotlib.use("Qt5Agg")

    # Prepare plot
    init_slice_idx = image.shape[-2] // 2
    init_4d_idx = 0
    init_img = image[..., init_slice_idx, init_4d_idx]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # slider spacing
    current_axis = ax.imshow(init_img, cmap="gray")
    ax.axis("off")
    # Add slider
    ax_slider_slice = plt.axes((0.8, 0.275, 0.03, 0.55))  # (0.25, 0.1, 0.65, 0.03)
    slider_slice = Slider(
        ax=ax_slider_slice,
        label="Slice",  # "Image Index",
        valmin=0,
        valmax=image.shape[-2] - 1,
        valinit=init_slice_idx,
        valstep=1,
        orientation="vertical",
    )
    ax_slider_4d = plt.axes((0.3, 0.1, 0.45, 0.03))
    slider_4d = Slider(
        ax=ax_slider_4d,
        label="4.D",  # "Image Index",
        valmin=0,
        valmax=image.shape[-1] - 1,
        valinit=init_4d_idx,
        valstep=1,
        orientation="horizontal",
    )

    def update_slider_slice(val):
        idx = int(slider_slice.val)
        current_axis.set_data(image[..., idx, int(slider_4d.val)])
        # plt.draw()
        fig.canvas.draw_idle()

    slider_slice.on_changed(update_slider_slice)

    def update_slider_4d(val):
        idx = int(slider_4d.val)
        current_axis.set_data(image[..., int(slider_slice.val), idx])
        plt.draw()

    slider_4d.on_changed(update_slider_4d)

    plt.show()
