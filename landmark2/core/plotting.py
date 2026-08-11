# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from landmark2.core import IS_COLAB, IS_KAGGLE, LOGGER, TryExcept, ops, plt_settings, threaded
from landmark2.core.checks import check_font, check_version, is_ascii
from landmark2.core.files import increment_path


def _gaussian_filter1d(y, sigma: int = 3, truncate: float = 4.0) -> np.ndarray:
    """Smooth a 1D array with a Gaussian kernel (NumPy replacement for scipy.ndimage.gaussian_filter1d).

    Args:
        y (np.ndarray): Input 1D array to smooth.
        sigma (int): Standard deviation of the Gaussian kernel.
        truncate (float): Truncate the kernel at this many standard deviations.

    Returns:
        (np.ndarray): Smoothed 1D array with the same length as the input.
    """
    y = np.asarray(y, dtype=float)
    radius = int(truncate * sigma + 0.5)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    # scipy 'reflect' boundary mode is equivalent to NumPy 'symmetric'
    return np.convolve(np.pad(y, radius, mode="symmetric"), kernel, mode="valid")


class Colors:
    """Ultralytics color palette for visualization and plotting.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to RGB
    values and accessing predefined color schemes for object detection and pose estimation.

    ## Ultralytics Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## Pose Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |

    !!! note "Ultralytics Brand Colors"

        For Ultralytics brand colors see [https://www.ultralytics.com/brand](https://www.ultralytics.com/brand).
        Please use the official Ultralytics colors for all marketing materials.

    Attributes:
        palette (list[tuple]): List of RGB color tuples for general use.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array for pose estimation with dtype np.uint8.

    Examples:
        >>> from landmark2.core.plotting import Colors
        >>> colors = Colors()
        >>> colors(5, True)  # Returns BGR format: (221, 111, 255)
        >>> colors(5, False)  # Returns RGB format: (255, 111, 221)
    """

    def __init__(self):
        """Initialize the Ultralytics color palette from a fixed list of hex color codes."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i: int | torch.Tensor, bgr: bool = False) -> tuple:
        """Return a color from the palette by index.

        Args:
            i (int | torch.Tensor): Color index.
            bgr (bool, optional): Whether to return BGR format instead of RGB.

        Returns:
            (tuple): RGB or BGR color tuple.
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> tuple:
        """Convert hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class Annotator:
    """Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image | np.ndarray): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype | ImageFont.load_default): Font used for text annotations.
        lw (int): Line width for drawing.
        skeleton (list[list[int]]): Skeleton structure for keypoints.
        limb_color (np.ndarray): Color palette for limbs.
        kpt_color (np.ndarray): Color palette for keypoints.
        dark_colors (set): Set of colors considered dark for text contrast.
        light_colors (set): Set of colors considered light for text contrast.

    Examples:
        >>> from landmark2.core.plotting import Annotator
        >>> im0 = cv2.imread("test.png")
        >>> annotator = Annotator(im0, line_width=10)
        >>> annotator.box_label([10, 10, 100, 100], "person", (255, 0, 0))
    """

    def __init__(
        self,
        im,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if not input_is_pil:
            if im.shape[2] == 1:  # handle grayscale
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 2:  # handle 2-channel images
                im = np.ascontiguousarray(np.dstack((im, np.zeros_like(im[..., :1]))))
            elif im.shape[2] > 3:  # multispectral
                im = np.ascontiguousarray(im[..., :3])
        if self.pil:  # use PIL
            self.im = im if input_is_pil else Image.fromarray(im)  # stay in BGR since color palette is in BGR
            if self.im.mode not in {"RGB", "RGBA"}:  # multispectral
                self.im = self.im.convert("RGB")
            self.draw = ImageDraw.Draw(self.im, "RGBA")
            try:
                font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            # Deprecation fix for w, h = getsize(string) -> _, _, w, h = getbox(string)
            if check_version(pil_version, "9.2.0"):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # text width, height
        else:  # use cv2
            assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)  # font thickness
            self.sf = self.lw / 3  # font scale
        # Pose
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(self, color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)) -> tuple:
        """Assign text color based on background color.

        Args:
            color (tuple, optional): The background color of the rectangle for text.
            txt_color (tuple, optional): The fallback color of the text.

        Returns:
            (tuple): Text color for label.

        Examples:
            >>> from landmark2.core.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_txt_color(color=(104, 31, 17))  # return (255, 255, 255)
        """
        if color in self.dark_colors:
            return 104, 31, 17
        elif color in self.light_colors:
            return 255, 255, 255
        else:
            return txt_color

    def box_label(self, box, label: str = "", color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)):
        """Draw a bounding box on an image with a given label.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str, optional): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle.
            txt_color (tuple, optional): The color of the text.

        Examples:
            >>> from landmark2.core.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.box_label(box=[10, 20, 30, 40], label="person")
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, torch.Tensor):
            box = box.tolist()

        multi_points = isinstance(box[0], list)  # multiple points with shape (n, 2)
        p1 = [int(b) for b in box[0]] if multi_points else (int(box[0]), int(box[1]))
        if self.pil:
            self.draw.polygon(
                [tuple(b) for b in box], width=self.lw, outline=color
            ) if multi_points else self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = p1[1] >= h  # label fits outside box
                if p1[0] > self.im.size[0] - w:  # size is (w, h), check if label extend beyond right side of image
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # self.draw.text([box[0], box[1]], label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            cv2.polylines(
                self.im, [np.asarray(box, dtype=int)], True, color, self.lw
            ) if multi_points else cv2.rectangle(
                self.im, p1, (int(box[2]), int(box[3])), color, thickness=self.lw, lineType=cv2.LINE_AA
            )
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                h += 3  # add pixels to pad text
                outside = p1[1] >= h  # label fits outside box
                if p1[0] > self.im.shape[1] - w:  # shape is (h, w), check if label extend beyond right side of image
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA,
                )

    def masks(self, masks, colors, im_gpu: torch.Tensor = None, alpha: float = 0.5, retina_masks: bool = False):
        """Plot masks on image.

        Args:
            masks (torch.Tensor | np.ndarray): Predicted masks with shape [n, h, w].
            colors (list[list[int]]): Colors for predicted masks, [[r, g, b] * n].
            im_gpu (torch.Tensor | None): Image on GPU with shape [3, h, w], range [0, 1].
            alpha (float, optional): Mask transparency: 0.0 fully transparent, 1.0 opaque.
            retina_masks (bool, optional): Whether to use high resolution masks or not.
        """
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        if im_gpu is None:
            assert isinstance(masks, np.ndarray), "`masks` must be a np.ndarray if `im_gpu` is not provided."
            overlay = self.im.copy()
            for i, mask in enumerate(masks):
                overlay[mask.astype(bool)] = colors[i]
            self.im = cv2.addWeighted(self.im, 1 - alpha, overlay, alpha, 0)
        else:
            assert isinstance(masks, torch.Tensor), "'masks' must be a torch.Tensor if 'im_gpu' is provided."
            if len(masks) == 0:
                self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
                return
            if im_gpu.device != masks.device:
                im_gpu = im_gpu.to(masks.device)

            ih, iw = self.im.shape[:2]
            if not retina_masks:
                # Use scale_masks to properly remove padding and upsample, convert bool to float first
                masks = ops.scale_masks(masks[None].float(), (ih, iw))[0] > 0.5
                # Convert original BGR image to RGB tensor
                im_gpu = (
                    torch.from_numpy(self.im).to(masks.device).permute(2, 0, 1).flip(0).contiguous().float() / 255.0
                )

            colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
            colors = colors[:, None, None]  # shape(n,1,1,3)
            masks = masks.unsqueeze(3)  # shape(n,h,w,1)
            masks_color = masks * (colors * alpha)  # shape(n,h,w,3)
            inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
            mcs = masks_color.max(dim=0).values  # shape(h,w,3)

            im_gpu = im_gpu.flip(dims=[0]).permute(1, 2, 0).contiguous()  # shape(h,w,3)
            im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
            self.im[:] = (im_gpu * 255).byte().cpu().numpy()
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def semantic_mask(self, mask, alpha: float = 0.5, ignore_index: int = 255):
        """Plot a semantic segmentation mask on the image.

        Args:
            mask (np.ndarray): Semantic mask with shape [h, w] containing integer class indices.
            alpha (float, optional): Mask transparency: 0.0 fully transparent, 1.0 opaque.
            ignore_index (int, optional): Class index to ignore (e.g., 255 for void/ignore).
        """
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        overlay = np.zeros_like(self.im)
        for cls_id in np.unique(mask):
            if cls_id == ignore_index:
                continue
            overlay[mask == cls_id] = colors(int(cls_id), True)
        self.im = cv2.addWeighted(self.im, 1 - alpha, overlay, alpha, 0)
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def kpts(
        self,
        kpts,
        shape: tuple = (640, 640),
        radius: int | None = None,
        kpt_line: bool = True,
        conf_thres: float = 0.25,
        kpt_color: tuple | None = None,
    ):
        """Plot keypoints on the image.

        Args:
            kpts (torch.Tensor): Keypoints, shape [17, 3] (x, y, confidence).
            shape (tuple, optional): Image shape (h, w).
            radius (int, optional): Keypoint radius.
            kpt_line (bool, optional): Draw lines between keypoints.
            conf_thres (float, optional): Confidence threshold.
            kpt_color (tuple, optional): Keypoint color.

        Notes:
            - `kpt_line=True` currently only supports human pose plotting.
            - Modifies self.im in-place.
            - If self.pil is True, converts image to numpy array and back to PIL.
        """
        radius = radius if radius is not None else self.lw
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):
            color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < conf_thres or conf2 < conf_thres:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(
                    self.im,
                    pos1,
                    pos2,
                    kpt_color or self.limb_color[i].tolist(),
                    thickness=int(np.ceil(self.lw / 2)),
                    lineType=cv2.LINE_AA,
                )
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width: int = 1):
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text: str, txt_color: tuple = (255, 255, 255), anchor: str = "top", box_color: tuple = ()):
        """Add text to an image using PIL or cv2.

        Args:
            xy (list[int]): Top-left coordinates for text placement.
            text (str): Text to be drawn.
            txt_color (tuple, optional): Text color.
            anchor (str, optional): Text anchor position ('top' or 'bottom').
            box_color (tuple, optional): Box background color with optional alpha.
        """
        if self.pil:
            w, h = self.font.getsize(text)
            if anchor == "bottom":  # start y from font bottom
                xy[1] += 1 - h
            for line in text.split("\n"):
                if box_color:
                    # Draw rectangle for each line
                    w, h = self.font.getsize(line)
                    self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=box_color)
                self.draw.text(xy, line, fill=txt_color, font=self.font)
                xy[1] += h
        else:
            if box_color:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3  # add pixels to pad text
                outside = xy[1] >= h  # label fits outside box
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, box_color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """Update `self.im` from a NumPy array or PIL image."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self, pil=False):
        """Return annotated image as array or PIL image."""
        im = np.asarray(self.im)  # self.im is in BGR
        return Image.fromarray(im[..., ::-1]) if pil else im

    def show(self, title: str | None = None):
        """Show the annotated image."""
        im = Image.fromarray(np.asarray(self.im)[..., ::-1])  # Convert BGR NumPy array to RGB PIL Image
        if IS_COLAB or IS_KAGGLE:  # cannot use IS_JUPYTER as it runs for all IPython environments
            try:
                display(im)  # noqa - display() function only available in ipython environments
            except ImportError as e:
                LOGGER.warning(f"Unable to display image in Jupyter notebooks: {e}")
        else:
            im.show(title=title)

    def save(self, filename: str = "image.jpg"):
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.im))

    @staticmethod
    def get_bbox_dimension(bbox: tuple | list):
        """Calculate the dimensions and area of a bounding box.

        Args:
            bbox (tuple | list): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).

        Returns:
            width (float): Width of the bounding box.
            height (float): Height of the bounding box.
            area (float): Area enclosed by the bounding box.

        Examples:
            >>> from landmark2.core.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_bbox_dimension(bbox=[10, 20, 30, 40])
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height


@TryExcept()
@plt_settings()
def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """Plot training labels including class histograms and box statistics.

    Args:
        boxes (np.ndarray): Bounding box coordinates in format [x, y, width, height].
        cls (np.ndarray): Class indices.
        names (dict, optional): Dictionary mapping class indices to class names.
        save_dir (Path, optional): Directory to save the plot.
        on_plot (Callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
    import polars
    from matplotlib.colors import LinearSegmentedColormap

    # Plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    nc = int(cls.max() + 1)  # number of classes
    boxes = boxes[:1000000]  # limit to 1M boxes
    x = polars.DataFrame(boxes, schema=["x", "y", "width", "height"])

    # Matplotlib labels
    subplot_3_4_color = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([x / 255 for x in colors(i)])
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
        ax[0].bar_label(y[2])
    else:
        ax[0].set_xlabel("classes")
    boxes = np.column_stack([0.5 - boxes[:, 2:4] / 2, 0.5 + boxes[:, 2:4] / 2]) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for class_id, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box.tolist(), width=1, outline=colors(class_id))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    ax[2].hist2d(x["x"], x["y"], bins=50, cmap=subplot_3_4_color)
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[3].hist2d(x["width"], x["height"], bins=50, cmap=subplot_3_4_color)
    ax[3].set_xlabel("width")
    ax[3].set_ylabel("height")
    for a in {0, 1, 2, 3}:
        for s in {"top", "right", "left", "bottom"}:
            ax[a].spines[s].set_visible(False)

    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def save_one_box(
    xyxy,
    im,
    file: Path = Path("im.jpg"),
    gain: float = 1.02,
    pad: int = 10,
    square: bool = False,
    BGR: bool = False,
    save: bool = True,
):
    """Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop.

    This function takes a bounding box and an image, and then saves a cropped portion of the image according to the
    bounding box. Optionally, the crop can be squared, and the function allows for gain and padding adjustments to the
    bounding box.

    Args:
        xyxy (torch.Tensor | list): A tensor or list representing the bounding box in xyxy format.
        im (np.ndarray): The input image.
        file (Path, optional): The path where the cropped image will be saved.
        gain (float, optional): A multiplicative factor to increase the size of the bounding box.
        pad (int, optional): The number of pixels to add to the width and height of the bounding box.
        square (bool, optional): If True, the bounding box will be transformed into a square.
        BGR (bool, optional): If True, the image will be returned in BGR format, otherwise in RGB.
        save (bool, optional): If True, the cropped image will be saved to disk.

    Returns:
        (np.ndarray): The cropped image.

    Examples:
        >>> from landmark2.core.plotting import save_one_box
        >>> xyxy = [50, 50, 150, 150]
        >>> im = cv2.imread("image.jpg")
        >>> cropped_im = save_one_box(xyxy, im, file="cropped.jpg", square=True)
    """
    if not isinstance(xyxy, torch.Tensor):  # may be list
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    grayscale = im.shape[2] == 1  # grayscale image
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR or grayscale else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        crop = crop.squeeze(-1) if grayscale else crop[..., ::-1] if BGR else crop
        Image.fromarray(crop).save(f, quality=95, subsampling=0)  # save RGB
    return crop


@threaded
def plot_images(
    labels: dict[str, Any],
    images: torch.Tensor | np.ndarray = np.zeros((0, 3, 640, 640), dtype=np.float32),
    paths: list[str] | None = None,
    fname: str = "images.jpg",
    names: dict[int, str] | None = None,
    on_plot: Callable | None = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.25,
    show_labels: bool = True,
    show_conf: bool = True,
) -> np.ndarray | None:
    """Plot image grid with labels, bounding boxes, masks, and keypoints.

    Args:
        labels (dict[str, Any]): Dictionary containing detection data with keys like 'cls', 'bboxes', 'conf', 'masks',
            'keypoints', 'batch_idx', 'img'.
        images (torch.Tensor | np.ndarray): Batch of images to plot. Shape: (batch_size, channels, height, width).
        paths (list[str] | None): List of file paths for each image in the batch.
        fname (str): Output filename for the plotted image grid.
        names (dict[int, str] | None): Dictionary mapping class indices to class names.
        on_plot (Callable | None): Callback function to be called after saving the plot.
        max_size (int): Maximum size of the output image grid.
        max_subplots (int): Maximum number of subplots in the image grid.
        save (bool): Whether to save the plotted image grid to a file.
        conf_thres (float): Confidence threshold for displaying detections.
        show_labels (bool): Whether to display class labels.
        show_conf (bool): Whether to display confidence values.

    Returns:
        (np.ndarray | None): Plotted image grid as a numpy array if save is False, None otherwise.

    Notes:
        This function supports both tensor and numpy array inputs. It will automatically
        convert tensor inputs to numpy arrays for processing.

        Channel Support:
        - 1 channel: Grayscale
        - 2 channels: Third channel added as zeros
        - 3 channels: Used as-is (standard RGB)
        - 4+ channels: Cropped to first 3 channels
    """
    for k in {"cls", "bboxes", "conf", "masks", "keypoints", "batch_idx", "images", "semantic_mask"}:
        if k not in labels:
            continue
        if k == "cls" and labels[k].ndim == 2:
            labels[k] = labels[k].squeeze(1)  # squeeze if shape is (n, 1)
        if isinstance(labels[k], torch.Tensor):
            labels[k] = labels[k].cpu().numpy()

    cls = labels.get("cls", np.zeros(0, dtype=np.int64))
    batch_idx = labels.get("batch_idx", np.zeros(cls.shape, dtype=np.int64))
    bboxes = labels.get("bboxes", np.zeros(0, dtype=np.float32))
    confs = labels.get("conf", None)
    masks = labels.get("masks", np.zeros(0, dtype=np.uint8))
    kpts = labels.get("keypoints", np.zeros(0, dtype=np.float32))
    semantic_masks = labels.get("semantic_mask", np.zeros(0, dtype=np.int64))
    images = labels.get("img", images)  # default to input images

    if len(images) and isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    # Handle 2-ch and n-ch images
    c = images.shape[1]
    if c == 2:
        zero = np.zeros_like(images[:, :1])
        images = np.concatenate((images, zero), axis=1)  # pad 2-ch with a black channel
    elif c > 3:
        images = images[:, :3]  # crop multispectral images to first 3 channels

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    fs = max(fs, 18)  # ensure that the font size is large enough to be easily readable.
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=str(names))
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text([x + 5, y + 5], text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None
            conf = confs[idx] if confs is not None else None  # check for confidence presence (label vs pred)

            if len(bboxes):
                boxes = bboxes[idx]
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                        boxes[..., [0, 2]] *= w  # scale to pixels
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5  # xywhr
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        conf_text = f"{conf[j]:.1f}" if conf is not None else ""
                        label = f"{c}" if show_labels else ""
                        label += f" {conf_text}".strip() if show_conf else ""
                        annotator.box_label(box, label, color=color)

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    label = f"{c}" if labels else f"{c} {conf[0]:.1f}"
                    annotator.text([x, y], label, txt_color=color, box_color=(64, 64, 64, 128))

            # Plot keypoints
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01:  # if normalized with tolerance .01
                        kpts_[..., 0] *= w  # scale to pixels
                        kpts_[..., 1] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)

            # Plot masks
            if len(masks):
                if idx.shape[0] == masks.shape[0] and masks.max() <= 1:  # overlap_mask=False
                    image_masks = masks[idx]
                else:  # overlap_mask=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(1, nl + 1).reshape((nl, 1, 1))
                    image_masks = (image_masks == index).astype(np.float32)

                im = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        try:
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                        except Exception:
                            pass
                annotator.fromarray(im)

        # Plot semantic masks
        if len(semantic_masks) and i < len(semantic_masks):
            mask = semantic_masks[i]
            mh, mw = mask.shape
            if mh != h or mw != w:
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            im = np.asarray(annotator.im).copy()
            sub_annotator = Annotator(np.ascontiguousarray(im[y : y + h, x : x + w]), line_width=1, pil=False)
            sub_annotator.semantic_mask(mask, alpha=0.4)
            im[y : y + h, x : x + w] = sub_annotator.im
            annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)  # save
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results(file: str = "path/to/results.csv", dir: str = "", on_plot: Callable | None = None):
    """Plot training results from a results CSV file. The function supports various types of data including instance
    segmentation, semantic segmentation, pose estimation, and classification. Plots are saved as 'results.png' in
    the directory where the CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided.
        on_plot (Callable, optional): Callback function to be executed after plotting. Takes filename as an argument.

    Examples:
        >>> from landmark2.core.plotting import plot_results
        >>> plot_results("path/to/results.csv")
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
    import polars as pl

    save_dir = Path(file).parent if file else Path(dir)
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."

    loss_keys, metric_keys = [], []
    fig, ax = None, None
    for i, f in enumerate(files):
        try:
            data = pl.read_csv(f, infer_schema_length=None)
            if i == 0:
                for c in data.columns:
                    if "loss" in c:
                        loss_keys.append(c)
                    elif "metric" in c:
                        metric_keys.append(c)
                loss_mid, metric_mid = len(loss_keys) // 2, len(metric_keys) // 2
                columns = (
                    loss_keys[:loss_mid] + metric_keys[:metric_mid] + loss_keys[loss_mid:] + metric_keys[metric_mid:]
                )
                fig, ax = plt.subplots(2, len(columns) // 2, figsize=(len(columns) + 2, 6), tight_layout=True)
                ax = ax.ravel()
            x = data.select(data.columns[0]).to_numpy().flatten()
            for i, j in enumerate(columns):
                y = data.select(j).to_numpy().flatten().astype("float")
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, _gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[i].set_title(j, fontsize=12)
        except Exception as e:
            LOGGER.error(f"Plotting error for {f}: {e}")
    if ax is not None:
        ax[1].legend()
        fname = save_dir / "results.png"
        fig.savefig(fname, dpi=200)
        plt.close()
        if on_plot:
            on_plot(fname)


@plt_settings()
def plot_multitrain_results(scores: dict, key: str = "fitness", save_dir=Path()):
    """Plot per-dataset metrics from a multi-dataset training run as a bar chart with the cross-dataset mean.

    Args:
        scores (dict): Mapping of dataset name to its scalar metric value.
        key (str): Name of the plotted metric, used as the y-axis label.
        save_dir (str | Path): Directory to save the 'multitrain_results.png' figure.

    Returns:
        (Path): Path to the saved figure.

    Examples:
        >>> from landmark2.core.plotting import plot_multitrain_results
        >>> plot_multitrain_results({"coco8": 0.61, "dota8": 0.48}, key="metrics/mAP50-95(B)")
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    mean = sum(scores.values()) / len(scores)
    fig, ax = plt.subplots(figsize=(max(6.0, len(scores) * 0.45), 5), tight_layout=True)
    ax.bar(range(len(scores)), list(scores.values()), color="#042AFF")
    ax.axhline(mean, color="orange", linestyle="--", label=f"mean = {mean:.3f}")
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(list(scores), rotation=90)
    ax.set_ylabel(key)
    ax.set_title(f"MultiTrainer results across {len(scores)} datasets")
    ax.legend()
    fname = Path(save_dir) / "multitrain_results.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def plt_color_scatter(v, f, bins: int = 20, cmap: str = "viridis", alpha: float = 0.8, edgecolors: str = "none"):
    """Plot a scatter plot with points colored based on a 2D histogram.

    Args:
        v (array-like): Values for the x-axis.
        f (array-like): Values for the y-axis.
        bins (int, optional): Number of bins for the histogram.
        cmap (str, optional): Colormap for the scatter plot.
        alpha (float, optional): Alpha for the scatter plot.
        edgecolors (str, optional): Edge colors for the scatter plot.

    Examples:
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    # Calculate 2D histogram and corresponding colors
    hist, xedges, yedges = np.histogram2d(v, f, bins=bins)
    colors = [
        hist[
            min(np.digitize(v[i], xedges, right=True) - 1, hist.shape[0] - 1),
            min(np.digitize(f[i], yedges, right=True) - 1, hist.shape[1] - 1),
        ]
        for i in range(len(v))
    ]

    # Scatter plot
    plt.scatter(v, f, c=colors, cmap=cmap, alpha=alpha, edgecolors=edgecolors)


@plt_settings()
def plot_tune_results(results_file: str = "tune_results.ndjson", exclude_zero_fitness_points: bool = True):
    """Plot the evolution results stored in a tuning NDJSON file.

    Args:
        results_file (str, optional): Path to the NDJSON file containing the tuning results.
        exclude_zero_fitness_points (bool, optional): Don't include points with zero fitness in tuning plots.

    Examples:
        >>> plot_tune_results("path/to/tune_results.ndjson")
    """
    import json

    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    def _save_one_file(file):
        """Save one matplotlib plot to 'file'."""
        plt.savefig(file, dpi=200)
        plt.close()
        LOGGER.info(f"Saved {file}")

    results_file = Path(results_file)
    with open(results_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        return

    keys = list(records[0].get("hyperparameters", {}))
    x = np.array(
        [[r.get("fitness", 0.0)] + [r.get("hyperparameters", {}).get(k, np.nan) for k in keys] for r in records],
        dtype=float,
    )
    len(x)
    all_fitness = x[:, 0]  # fitness
    zero_mask = slice(None)
    if exclude_zero_fitness_points:
        zero_mask = all_fitness > 0  # exclude zero-fitness points
        x, all_fitness = x[zero_mask], all_fitness[zero_mask]
    if len(all_fitness) == 0:
        LOGGER.warning("No valid fitness values to plot (all iterations may have failed)")
        return
    fitness = all_fitness.copy()
    # Iterative sigma rejection on lower bound only
    for _ in range(3):  # max 3 iterations
        mean, std = fitness.mean(), fitness.std()
        lower_bound = mean - 3 * std
        mask = fitness >= lower_bound
        if mask.all():  # no more outliers
            break
        x, fitness = x[mask], fitness[mask]
    j = np.argmax(fitness)  # max fitness index
    n = math.ceil(len(keys) ** 0.5)  # columns and rows in plot
    plt.figure(figsize=(10, 10), tight_layout=True)
    for i, k in enumerate(keys):
        v = x[:, i + 1]
        mu = v[j]  # best single result
        plt.subplot(n, n, i + 1)
        plt_color_scatter(v, fitness, cmap="viridis", alpha=0.8, edgecolors="none")
        plt.plot(mu, fitness.max(), "k+", markersize=15)
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # limit to 40 characters
        plt.tick_params(axis="both", labelsize=8)  # Set axis label size to 8
        if i % n != 0:
            plt.yticks([])
    _save_one_file(results_file.with_name("tune_scatter_plots.png"))

    # Fitness vs iteration
    x = range(1, len(all_fitness) + 1)
    plt.figure(figsize=(10, 6), tight_layout=True)
    for dataset in sorted({k for r in records for k in r.get("datasets", {})}):
        y = np.array([r.get("datasets", {}).get(dataset, {}).get("fitness", np.nan) for r in records], dtype=float)
        if exclude_zero_fitness_points and not isinstance(zero_mask, slice):
            y = y[zero_mask]
        plt.plot(x, y, "o", markersize=5, alpha=0.8, label=dataset)
    plt.plot(x, _gaussian_filter1d(all_fitness, sigma=3), ":", color="0.35", label="smoothed mean", linewidth=2)
    plt.title("Fitness vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()
    _save_one_file(results_file.with_name("tune_fitness.png"))


@plt_settings()
def feature_visualization(x, module_type: str, stage: int, n: int = 32, save_dir: Path = Path("runs/detect/exp")):
    """Visualize feature maps of a given model module during inference.

    Args:
        x (torch.Tensor): Features to be visualized.
        module_type (str): Module type.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot.
        save_dir (Path, optional): Directory to save results.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    for m in {"Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"}:  # all model heads
        if m in module_type:
            return
    if isinstance(x, torch.Tensor):
        _, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.rsplit('.', 1)[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            _, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # n/8 rows x 8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis("off")

            LOGGER.info(f"Saving {f}... ({n}/{channels})")
            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # npy save
"""Only the three standardized visual artifacts emitted by landmark training."""


import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_CACHE = Path(tempfile.gettempdir()) / "uknee-matplotlib"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from landmark2.data.schema import LANDMARK_PATH_RANGES, REGION_NAMES


COLORS = ("#00b4ff", "#ea580c", "#ff78dc", "#16a34a")


def _style(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", length=5, width=1.0, colors="black")
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.set_axisbelow(True)


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = rows[0].keys() if rows else ()
    return {
        key: np.asarray([float(row[key]) if row.get(key, "") else np.nan for row in rows], dtype=float)
        for key in keys
    }


def plot_dashboard_pose(
    csv_file: str | Path,
    output_png: str | Path | None = None,
    *,
    pixel_spacing: float = 0.10,
    model_name: str | None = None,
    elapsed_seconds: float | None = None,
) -> Path | None:
    csv_path = Path(csv_file)
    if not csv_path.exists():
        return None
    values = _read_csv(csv_path)
    if not values:
        return None
    epochs = values.get("epoch", np.arange(1, len(next(iter(values.values()))) + 1))
    destination = Path(output_png) if output_png else csv_path.parent / "landmark2_dashboard.png"

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)

    # Core vibrant color palette
    c_blue = "#2563eb"
    c_red = "#dc2626"
    c_green = "#16a34a"
    c_purple = "#9333ea"

    # Balanced uniform marker sizes
    marker_styles = [
        ("#facc15", "*", 12, "#b45309"),  # Top 1: Gold Star
        ("#94a3b8", "D", 9, "#475569"),   # Top 2: Slate Diamond
        ("#b45309", "o", 8, "#78350f"),   # Top 3: Bronze Circle
    ]

    # Title formatting
    m_name = model_name or csv_path.parent.name or "yolo26-pose-v1"
    time_info = ""
    if elapsed_seconds and elapsed_seconds > 0:
        mins = int(elapsed_seconds // 60)
        secs = int(elapsed_seconds % 60)
        avg_ep = elapsed_seconds / max(1, len(epochs))
        time_info = f" | Train Time: {mins}m {secs}s ({avg_ep:.1f}s/ep)"

    fig.suptitle(f"Landmark: {m_name}{time_info}", color="#1e293b", fontsize=14, fontweight="bold", y=0.98)

    # Subplot 1: Train & Val Loss + Top 1, 2, 3 Val Loss Markers
    ax1 = axes[0, 0]
    _style(ax1)
    if "train/loss" in values:
        ax1.plot(epochs, values["train/loss"], label="Train Loss", color=c_blue, lw=2.2)
    if "val/loss" in values:
        val_losses = values["val/loss"]
        ax1.plot(epochs, val_losses, label="Val Loss", color=c_red, lw=2.2, linestyle="--")
        finite = np.flatnonzero(np.isfinite(val_losses))
        if finite.size:
            top_indices = finite[np.argsort(val_losses[finite])[:3]]
            for rank, idx in enumerate(top_indices):
                c, m, ms, ec = marker_styles[rank % len(marker_styles)]
                ep = epochs[idx]
                val = val_losses[idx]
                ax1.plot(ep, val, marker=m, markersize=ms, color=c, markeredgecolor=ec,
                         markeredgewidth=1.2, linestyle="None", label=f"Top{rank+1} Val Loss: {val:.4f} (E{ep})", zorder=5)

    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold", color="#1e293b")
    ax1.set_xlabel("Epochs", fontsize=10, color="black")
    ax1.set_ylabel("Loss", fontsize=10, color="black", fontweight="semibold")
    ax1.set_ylim(bottom=-0.005)
    leg1 = ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.0)
    if leg1:
        leg1.get_frame().set_alpha(0.96)
        leg1.set_zorder(100)

    # Subplot 2: Overall MRE & Box mAP50-95
    ax2 = axes[0, 1]
    _style(ax2)
    best_mre_idx = 0
    if "metrics/MRE" in values:
        mre_mm = values["metrics/MRE"] * pixel_spacing
        best_mre_idx = int(np.nanargmin(mre_mm))
        best_mre_ep = epochs[best_mre_idx]
        best_mre_val = mre_mm[best_mre_idx]

        ax2.plot(epochs, mre_mm, label="Overall MRE (mm)", color=c_red, lw=2.5)
        ax2.plot(best_mre_ep, best_mre_val, marker="*", markersize=12, color="#facc15",
                 markeredgecolor="#991b1b", markeredgewidth=1.2,
                 label=f"Best MRE: {best_mre_val:.4f} mm (E{best_mre_ep})", zorder=5)

    ax2.plot([], [], " ", label=f"Pixel Spacing: {pixel_spacing:.2f} mm/px")

    has_bbox_map = "metrics/mAP50-95(B)" in values and np.isfinite(values["metrics/mAP50-95(B)"]).any()
    if has_bbox_map:
        bbox_vals = values["metrics/mAP50-95(B)"]
        ax2_right = ax2.twinx()
        _style(ax2_right)
        ax2_right.grid(False)
        ax2_right.plot(epochs, bbox_vals, label="Box mAP50-95", color=c_blue, lw=2.0, linestyle="--")
        best_map_idx = int(np.nanargmax(bbox_vals))
        best_map_ep = epochs[best_map_idx]
        best_map_val = bbox_vals[best_map_idx]
        ax2_right.plot(best_map_ep, best_map_val, marker="*", markersize=12,
                       color="#60a5fa", markeredgecolor="#1e3a8a", markeredgewidth=1.2,
                       label=f"Best Box mAP: {best_map_val:.4f} (E{best_map_ep})", zorder=5)
        ax2_right.set_ylabel("Box mAP50-95", fontsize=10, color=c_blue, fontweight="semibold")
        ax2_right.set_ylim(bottom=-0.04, top=1.05)

    ax2.set_title("Mean Radial Error (MRE)" + (" & Box mAP50-95" if has_bbox_map else ""),
                  fontsize=12, fontweight="bold", color="#1e293b")
    ax2.set_xlabel("Epochs", fontsize=10, color="black")
    ax2.set_ylabel("MRE Error (mm)", fontsize=10, color=c_red, fontweight="semibold")
    ax2.set_ylim(bottom=-0.05)

    handles_l2, labels_l2 = ax2.get_legend_handles_labels()
    if has_bbox_map:
        handles_r2, labels_r2 = ax2_right.get_legend_handles_labels()
        leg2 = ax2_right.legend(handles_l2 + handles_r2, labels_l2 + labels_r2,
                                loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.8)
    else:
        leg2 = ax2.legend(handles_l2, labels_l2, loc="upper left", frameon=True,
                          facecolor="white", edgecolor="#cbd5e1", fontsize=9.0)
    if leg2:
        leg2.get_frame().set_alpha(0.96)
        leg2.set_zorder(100)

    # Subplot 3: Per-Region MRE
    ax3 = axes[1, 0]
    _style(ax3)
    region_cols = [("metrics/MRE_femur", "Femur", c_blue),
                   ("metrics/MRE_tibia", "Tibia", c_red),
                   ("metrics/MRE_fibula", "Fibula", c_green),
                   ("metrics/MRE_patella", "Patella", c_purple)]
    for col, name, color in region_cols:
        if col in values:
            r_mm = values[col] * pixel_spacing
            best_val = r_mm[best_mre_idx] if len(r_mm) > best_mre_idx else r_mm[-1]
            ax3.plot(epochs, r_mm, label=f"{name}: {best_val:.4f} mm", color=color, lw=2.0)

    if "metrics/MRE" in values and len(mre_mm) > best_mre_idx:
        ax3.plot(epochs[best_mre_idx], mre_mm[best_mre_idx], marker="*", markersize=12, color="#facc15",
                 markeredgecolor="#b45309", markeredgewidth=1.2, label=f"Best MRE Epoch (E{best_mre_ep})", zorder=5)

    ax3.set_title("Per-Region MRE (Femur, Tibia, Fibula, Patella in mm)", fontsize=12, fontweight="bold", color="#1e293b")
    ax3.set_xlabel("Epochs", fontsize=10, color="black")
    ax3.set_ylabel("Error (mm)", fontsize=10, color="black", fontweight="semibold")
    ax3.set_ylim(bottom=-0.05)
    leg3 = ax3.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.8)
    if leg3:
        leg3.get_frame().set_alpha(0.96)
        leg3.set_zorder(100)

    # Subplot 4: PCK Accuracy & HD95
    ax4 = axes[1, 1]
    _style(ax4)
    has_hd95 = "metrics/HD95" in values and np.isfinite(values["metrics/HD95"]).any()
    pck_cols = [("metrics/PCK2", "PCK@0.2mm", c_blue),
                ("metrics/PCK4", "PCK@0.4mm", c_green),
                ("metrics/PCK8", "PCK@0.8mm", c_purple)]
    for col, name, color in pck_cols:
        if col in values:
            p_vals = values[col] * 100.0
            best_pck = p_vals[best_mre_idx] if len(p_vals) > best_mre_idx else p_vals[-1]
            ax4.plot(epochs, p_vals, label=f"{name}: {best_pck:.2f}%", color=color, lw=2.0)

    if "metrics/PCK2" in values:
        p2_vals = values["metrics/PCK2"] * 100.0
        best_p2_idx = int(np.nanargmax(p2_vals))
        best_p2_ep = epochs[best_p2_idx]
        best_p2_val = p2_vals[best_p2_idx]
        ax4.plot(best_p2_ep, best_p2_val, marker="*", markersize=12,
                 color="#facc15", markeredgecolor="#1e3a8a", markeredgewidth=1.2,
                 label=f"Best PCK@0.2mm: {best_p2_val:.2f}% (E{best_p2_ep})", zorder=5)

    if has_hd95:
        hd95_vals = values["metrics/HD95"] * pixel_spacing
        ax4_right = ax4.twinx()
        _style(ax4_right)
        ax4_right.grid(False)
        ax4_right.plot(epochs, hd95_vals, label="Val HD95 (mm)", color=c_red, lw=2.0, linestyle="-.")
        best_hd95_idx = int(np.nanargmin(hd95_vals))
        best_hd95_ep = epochs[best_hd95_idx]
        best_hd95_val = hd95_vals[best_hd95_idx]
        ax4_right.plot(best_hd95_ep, best_hd95_val, marker="D", markersize=9,
                       color="#f87171", markeredgecolor="#991b1b", markeredgewidth=1.2,
                       label=f"Best HD95: {best_hd95_val:.4f} mm (E{best_hd95_ep})", zorder=5)
        ax4_right.set_ylabel("HD95 (mm)", fontsize=10, color=c_red, fontweight="semibold")
        ax4_right.set_ylim(bottom=-0.05)

    ax4.set_title("PCK Accuracy" + (" & Hausdorff Distance (HD95)" if has_hd95 else ""),
                  fontsize=12, fontweight="bold", color="#1e293b")
    ax4.set_xlabel("Epochs", fontsize=10, color="black")
    ax4.set_ylabel("Accuracy (%)", fontsize=10, color="black", fontweight="semibold")
    ax4.set_ylim(bottom=-5, top=108)

    handles_l4, labels_l4 = ax4.get_legend_handles_labels()
    if has_hd95:
        handles_r4, labels_r4 = ax4_right.get_legend_handles_labels()
        leg4 = ax4_right.legend(handles_l4 + handles_r4, labels_l4 + labels_r4,
                                loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.8)
    else:
        leg4 = ax4.legend(handles_l4, labels_l4, loc="upper left", frameon=True,
                          facecolor="white", edgecolor="#cbd5e1", fontsize=9.0)
    if leg4:
        leg4.get_frame().set_alpha(0.96)
        leg4.set_zorder(100)

    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


def _mean_curve(curve: Any) -> np.ndarray:
    array = np.asarray(curve, dtype=float)
    return np.nanmean(array, axis=0) if array.ndim > 1 else array


def plot_pose_metrics(metrics: Any, output_png: str | Path) -> Path:
    """Write normalized confusion, pose/box PR, and F1-confidence in one figure."""
    destination = Path(output_png)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    raw_matrix = np.asarray(metrics.confusion_matrix.matrix, dtype=float)[:4, :4]
    norm_matrix = raw_matrix / np.maximum(raw_matrix.sum(axis=0, keepdims=True), 1e-9)
    gt_counts = raw_matrix.sum(axis=0).astype(int)

    ax = axes[0, 0]
    image = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)
    _style(ax)
    ax.grid(False)  # Remove grid lines slicing through heatmap text

    for row in range(4):
        for column in range(4):
            val = norm_matrix[row, column]
            raw_cnt = int(raw_matrix[row, column])
            # Subtle adaptive frosted glass pill: harmonious, elegant, high legibility
            if val > 0.45:
                text_color = "#ffffff"
                bg_box = "#000000"
                bg_alpha = 0.25
            else:
                text_color = "#0f172a"
                bg_box = "#ffffff"
                bg_alpha = 0.55

            cell_text = f"{val * 100:.1f}%\n(n={raw_cnt})"
            ax.text(
                column,
                row,
                cell_text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25,rounding_size=0.4",
                    facecolor=bg_box,
                    alpha=bg_alpha,
                    edgecolor="none",
                ),
            )

    x_labels = [f"{name.title()}\n(N={gt_counts[i]})" for i, name in enumerate(REGION_NAMES)]
    ax.set_xticks(range(4), x_labels, rotation=30, ha="right")
    ax.set_yticks(range(4), [name.title() for name in REGION_NAMES])
    ax.set(title="Normalized Confusion Matrix (Counts & %)", xlabel="True", ylabel="Predicted")
    fig.colorbar(image, ax=ax, fraction=0.046)

    for ax, metric, title in (
        (axes[0, 1], metrics.pose, "Pose PR Curve"),
        (axes[1, 0], metrics.box, "Box PR Curve"),
    ):
        x = np.asarray(getattr(metric, "px", np.linspace(0, 1, 1000)))
        precision = _mean_curve(getattr(metric, "prec_values", np.zeros_like(x)))
        ax.plot(x, precision, color=COLORS[0], linewidth=2)
        ax.set(title=title, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
        _style(ax)

    ax = axes[1, 1]
    for metric, label, color in ((metrics.pose, "Pose F1", COLORS[1]), (metrics.box, "Box F1", COLORS[0])):
        x = np.asarray(getattr(metric, "px", np.linspace(0, 1, 1000)))
        f1 = _mean_curve(getattr(metric, "f1_curve", np.zeros_like(x)))
        ax.plot(x, f1, color=color, linewidth=2, label=label)
    ax.set(title="F1-Score / Confidence", xlabel="Confidence", ylabel="F1", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    _style(ax)

    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_validation_samples(records: list[dict[str, Any]], output_png: str | Path, epoch: int | float = 150) -> Path | None:
    """Render four validation images in a 2x2 grid (2 top, 2 bottom) preserving aspect ratio."""
    if not records:
        return None
    destination = Path(output_png)
    fig, axes = plt.subplots(2, 2, figsize=(11, 11), dpi=150)
    axes_list = list(axes.flat)
    neon_green = "#39ff14"  # Vibrant neon green for keypoint scatter markers

    for ax, record in zip(axes_list, records[:4]):
        image = record["image"]
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None, aspect="equal")
        pred, valid = record["pred"], record["valid"]
        offset = 0
        for color, name, count in zip(COLORS, REGION_NAMES, (45, 51, 24, 9)):
            local = pred[offset : offset + count]
            local_valid = valid[offset : offset + count] & np.isfinite(local).all(axis=1)
            # Solid neon green points without outer black border
            ax.scatter(local[local_valid, 0], local[local_valid, 1], s=10, color=neon_green, edgecolors="none", zorder=5)
            offset += count
        for start, stop in LANDMARK_PATH_RANGES:
            path = pred[start:stop]
            mask = valid[start:stop] & np.isfinite(path).all(axis=1)
            if mask.sum() >= 2:
                path_color = COLORS[0] if stop <= 45 else (COLORS[1] if stop <= 96 else (COLORS[2] if stop <= 120 else COLORS[3]))
                # Clean colored region path line without black stroke outline underneath
                ax.plot(path[mask, 0], path[mask, 1], color=path_color, linewidth=1.1, zorder=4)
        banner = (
            f"MRE {record['mre_px'] * 0.10:.3f} mm | PCK {record['pck2'] * 100:.1f}%\n"
            f"HD95 {record['hd95_px'] * 0.10:.3f} mm | IoU {record['box_iou'] * 100:.1f}%"
        )
        ax.text(0.01, 0.99, banner, transform=ax.transAxes, va="top", ha="left", color="white", fontsize=7.5,
                bbox={"facecolor": "black", "alpha": 0.75, "pad": 3.5, "edgecolor": "#334155", "linewidth": 0.8})
        ax.set_title(Path(record["path"]).name, fontsize=9.5, fontweight="bold", color="#1e293b", pad=6)
        ax.axis("off")
    for ax in axes_list[len(records[:4]) :]:
        ax.axis("off")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=COLORS[0], lw=2.0, label="Femur Path"),
        Line2D([0], [0], color=COLORS[1], lw=2.0, label="Tibia Path"),
        Line2D([0], [0], color=COLORS[2], lw=2.0, label="Fibula Path"),
        Line2D([0], [0], color=COLORS[3], lw=2.0, label="Patella Path"),
        Line2D([0], [0], color=neon_green, marker="o", markersize=6, linestyle="None", label="Keypoints"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=9.5, frameon=True, facecolor="white", edgecolor="#94a3b8")
    fig.suptitle(f"Validation Landmark Predictions — Epoch {int(epoch)}", fontsize=13.5, fontweight="bold", color="#1e293b", ha="center", y=0.955)
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


__all__ = ["plot_dashboard_pose", "plot_pose_metrics", "plot_validation_samples"]
