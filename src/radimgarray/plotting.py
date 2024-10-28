from __future__ import annotations
import imantics
import numpy as np

# import PyQt6
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider


def show_image(image: np.ndarray, title: str = ""):
    """Display a 2D image.

    Args:
        image (np.ndarray): 2D image to display.
        title (str, optional): Title of the plot.
    """
    if len(image.shape) < 3:
        plt.imshow(image, cmap="gray")
        plt.show()
    elif len(image.shape) == 3:
        plot_3d_image(image)
    elif len(image.shape) == 4:
        plot_4d_image(image)


def plot_3d_image(image: np.ndarray):
    """Display a 3D image.

    Args:
        image (np.ndarray): 3D image to display.
    """

    # Set the backend to Qt5Agg
    # matplotlib.use("Qt5Agg")

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
    """Display a 4D image.

    Args:
        image (np.ndarray): 4D image to display.
    """
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


def calculate_polygons(array: np.ndarray, seg_values: list):
    """Lists polygons/segmentations for all slices.

    Creates a list containing one list for each slice (size = number of slices).
    Each of these lists contains the lists for each segmentation (number of segmentations).
    Each of these lists contains the polygons(/segmentation obj?) found in that slice (this length might be
    varying)

    Args:
        array (np.ndarray): 3D image array
        seg_values (list): list of segmentation values

    """
    segmentations = dict()
    for seg_index in seg_values:
        segmentations[seg_index] = Segmentation(array, seg_index)
    return segmentations


class Segmentation:
    """Segmentation class for a single segmentation.

    The main purpose of the Segmentation class is to store the image array containing
    only the selected segmentation, as well as the polygons for each slice. The class
    uses the imantics library to convert the image array into a list of Polygon objects.

    Attributes:
        seg_index (int): segmentation index
        img (np.ndarray): image array containing only the selected segmentation
        polygons (dict): dictionary containing the polygons for each slice
        polygon_patches (dict): dictionary containing the patches for each slice
    """

    def __init__(self, seg_img: np.ndarray, seg_index: int):
        self.seg_index = seg_index
        self.img = seg_img.copy()
        # check if image contains only the selected seg_index else change
        self.img[self.img != seg_index] = 0
        self.polygons = dict()
        self.polygon_patches = dict()
        self.__get_polygons()
        # self.number_polygons = len(self.polygons)

    def __get_polygons(self):
        """Create imantics Polygon list of image array.

        The __get_polygons function is a helper function that uses the imantics library
        to convert the image array into a list of Polygon objects. The polygons are
        stored in self.polygons, and a list of patches for each slice is stored in
        self.polygon_patches.
        """

        # Set dictionaries for polygons
        polygons = dict()
        polygon_patches = dict()

        for slice_number in range(self.img.shape[2]):
            polygons[slice_number] = (
                imantics.Mask(np.rot90(self.img[:, :, slice_number])).polygons()
                if not None
                else None
            )
            # Transpose Points to fit patchify
            points = list()
            for poly in polygons[slice_number].points:
                points.append(poly.T)
            if len(points):
                if len(points) > 1:
                    polygon_patches_list = list()
                    for point_set in points:
                        if point_set.size > 2:
                            polygon_patches_list.append(self.patchify([point_set]))
                    polygon_patches[slice_number] = polygon_patches_list
                else:
                    polygon_patches[slice_number] = [
                        patches.Polygon(polygons[slice_number].points[0])
                    ]
        self.polygons = polygons
        self.polygon_patches = polygon_patches

    # https://gist.github.com/yohai/81c5854eaa4f8eb5ad2256acd17433c8
    @staticmethod
    def patchify(polys):
        """Returns a matplotlib patch representing the polygon with holes.

        polys is an iterable (i.e. list) of polygons, each polygon is a numpy array
        of shape (2, N), where N is the number of points in each polygon. The first
        polygon is assumed to be the exterior polygon and the rest are holes. The
        first and last points of each polygon may or may not be the same.
        This is inspired by
        https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
        Example usage:
        ext = np.array([[-4, 4, 4, -4, -4], [-4, -4, 4, 4, -4]])
        t = -np.linspace(0, 2 * np.pi)
        hole1 = np.array([2 + 0.4 * np.cos(t), 2 + np.sin(t)])
        hole2 = np.array([np.cos(t) * (1 + 0.2 * np.cos(4 * t + 1)),
                          np.sin(t) * (1 + 0.2 * np.cos(4 * t))])
        hole2 = np.array([-2 + np.cos(t) * (1 + 0.2 * np.cos(4 * t)),
                          1 + np.sin(t) * (1 + 0.2 * np.cos(4 * t))])
        hole3 = np.array([np.cos(t) * (1 + 0.5 * np.cos(4 * t)),
                          -2 + np.sin(t)])
        holes = [ext, hole1, hole2, hole3]
        patch = patchify([ext, hole1, hole2, hole3])
        ax = plt.gca()
        ax.add_patch(patch)
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        """

        # TODO: this only works as desired if the first is the exterior and none of the
        # other regions is outside the first one therefor the segmentation needs to be
        # treated accordingly

        def reorder(poly, cw=True):
            """Reorders the polygon to run clockwise or counter-clockwise according to
            the value of cw.

            It calculates whether a polygon is cw or ccw by summing (x2-x1)*(y2+y1) for
            all edges of the polygon, see https://stackoverflow.com/a/1165943/898213.
            """
            # Close polygon if not closed
            if not np.allclose(poly[:, 0], poly[:, -1]):
                poly = np.c_[poly, poly[:, 0]]
            direction = (
                (poly[0] - np.roll(poly[0], 1)) * (poly[1] + np.roll(poly[1], 1))
            ).sum() < 0
            if direction == cw:
                return poly
            else:
                return np.array([p[::-1] for p in poly])

        def ring_coding(n):
            """Returns a list of len(n).

            Of this format:
            [MOVETO, LINETO, LINETO, ..., LINETO, LINETO CLOSEPOLY]
            """

            codes = [matplotlib.path.Path.LINETO] * n
            codes[0] = matplotlib.path.Path.MOVETO
            codes[-1] = matplotlib.path.Path.CLOSEPOLY
            return codes

        ccw = [True] + ([False] * (len(polys) - 1))
        polys = [reorder(poly, c) for poly, c in zip(polys, ccw)]
        path_codes = np.concatenate([ring_coding(p.shape[1]) for p in polys])
        vertices = np.concatenate(polys, axis=1)
        return patches.PathPatch(matplotlib.path.Path(vertices.T, path_codes))
