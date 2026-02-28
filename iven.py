import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from itertools import combinations

import scipy
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_hex, to_rgba
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Path3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QComboBox,
    QLabel,
    QFileDialog,
    QSlider,
    QLineEdit,
    QMessageBox,
    QSizePolicy,
    QDialog,
    QRadioButton,
    QButtonGroup,
    QDialogButtonBox,
    QToolBar,
    QToolButton,
    QGridLayout,
    QColorDialog,
    QScrollArea,
    QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QIcon, QAction, QColor

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS # type: ignore
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

def abspath(path):
    return os.path.join(base_path, path)


class IconToggle(QToolButton):
    """A checkable icon-only QToolButton that swaps SVG icons when toggled.

    Drop-in replacement for QCheckBox: same .isChecked(), .setChecked(),
    and .toggled signal interface.
    """

    def __init__(
        self, svg_on_path: str, svg_off_path: str, checked: bool = False, parent=None
    ):
        super().__init__(parent)
        self._icon_on = QIcon(svg_on_path)
        self._icon_off = QIcon(svg_off_path)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setIconSize(QSize(16, 16))
        self.setFixedSize(24, 24)
        self.setStyleSheet(
            "QToolButton { border: none; background: transparent; border-radius: 3px; }"
            "QToolButton:hover { background: #e0e0e0; }"
        )
        self._refresh_icon()
        self.toggled.connect(self._refresh_icon)

    def _refresh_icon(self):
        self.setIcon(self._icon_on if self.isChecked() else self._icon_off)


class ClickableSwatch(QLabel):
    """A small coloured square that emits *clicked* when pressed."""

    clicked = pyqtSignal()

    def __init__(self, colour: str = "#888888", size: int = 10, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._colour = colour
        self._size = size
        self._update_style()
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def get_luminance(self):
        bg_hex = self._colour.lstrip("#")
        r = int(bg_hex[0:2], 16) / 255
        g = int(bg_hex[2:4], 16) / 255
        b = int(bg_hex[4:6], 16) / 255

        def correct(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        r = correct(r)
        g = correct(g)
        b = correct(b)

        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return luminance

    def text_color(self) -> str:
        luminance = self.get_luminance()
        return "#000000" if luminance > 0.179 else "#ffffff"
    
    def border_color(self) -> str:
        luminance = self.get_luminance()
        return "#000000" if luminance > 0.8 else self._colour

    def _update_style(self):
        text_colour = self.text_color()
        border_colour = self.border_color()
        self.setStyleSheet(
            f"color: {text_colour};"
            f"background: {self._colour}; border-radius: {self._size // 2}px;"
            f"border: 0.5px solid {border_colour};"
        )

    def set_colour(self, colour: str):
        self._colour = colour
        self._update_style()

    def get_colour(self) -> str:
        return self._colour

    def mousePressEvent(self, ev):
        self.clicked.emit()
        super().mousePressEvent(ev)


class ChannelColourDialog(QDialog):
    """Dialog for choosing continuous vs categorical colouring and picking colours."""

    def __init__(
        self,
        column_name: str,
        unique_values,
        current_mode: str,
        gradient_colours: Tuple,
        category_colours: Dict,
        edge_outside_colour: str,
        edge_inside_colour: str,
        bg_colour: str,
        force_categorical: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Colour Settings")
        self.setMinimumWidth(380)
        self._column_name = column_name
        self._unique_values = unique_values or []
        self._mode = current_mode
        self._gradient_colours = (
            list(gradient_colours) if gradient_colours else ["#ff66cc", "#ffcc00"]
        )
        self._category_colours = dict(category_colours) if category_colours else {}
        self._cat_swatches = {}

        self.edge_outside_colour = edge_outside_colour
        self.edge_inside_colour = edge_inside_colour
        self.bg_colour = bg_colour

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        lbl = QLabel(f"General Colour Settings")
        lbl.setStyleSheet("font-weight: 600; font-size: 12px; color: #1a1a1a;")
        layout.addWidget(lbl)

        # Outside
        outside_row = QHBoxLayout()
        outside_row.setSpacing(6)
        self._edge_outside_swatch = ClickableSwatch(edge_outside_colour, 14)
        self._edge_outside_swatch.setToolTip("Edge colour for outside cells")
        self._edge_outside_swatch.clicked.connect(self._pick_edge_outside_colour)
        edge_outside_lbl = QLabel("Outside")
        edge_outside_lbl.setStyleSheet("font-size: 10px; color: #777; background: transparent;")
        outside_row.addWidget(self._edge_outside_swatch)
        outside_row.addWidget(edge_outside_lbl)
        outside_row.addStretch()
        outside_row.addSpacing(8)
        layout.addLayout(outside_row)

        # Inside
        inside_row = QHBoxLayout()
        inside_row.setSpacing(6)
        self._edge_inside_swatch = ClickableSwatch(edge_inside_colour, 14)
        self._edge_inside_swatch.setToolTip("Edge colour for inside cells")
        self._edge_inside_swatch.clicked.connect(self._pick_edge_inside_colour)
        edge_inside_lbl = QLabel("Inside")
        edge_inside_lbl.setStyleSheet("font-size: 10px; color: #777; background: transparent;")
        inside_row.addWidget(self._edge_inside_swatch)
        inside_row.addWidget(edge_inside_lbl)
        inside_row.addStretch()
        inside_row.addSpacing(8)
        layout.addLayout(inside_row)

        # Background
        bg_row = QHBoxLayout()
        bg_row.setSpacing(6)
        self._bg_swatch = ClickableSwatch(bg_colour, 14)
        self._bg_swatch.setToolTip("Canvas background colour")
        self._bg_swatch.clicked.connect(self._pick_bg_colour)
        bg_lbl = QLabel("Background")
        bg_lbl.setStyleSheet("font-size: 10px; color: #777; background: transparent;")
        bg_row.addWidget(self._bg_swatch)
        bg_row.addWidget(bg_lbl)
        bg_row.addStretch()
        layout.addLayout(bg_row)

        lbl = QLabel(f"Channel Colour Settings ({column_name})")
        lbl.setStyleSheet("font-weight: 600; font-size: 12px; color: #1a1a1a;")
        layout.addWidget(lbl)

        mode_lbl = QLabel("Colour type")
        mode_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #555;")
        layout.addWidget(mode_lbl)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)
        self._rb_continuous = QRadioButton("Continuous")
        self._rb_categorical = QRadioButton("Categorical")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._rb_continuous)
        self._mode_group.addButton(self._rb_categorical)

        if force_categorical:
            self._rb_continuous.setEnabled(False)

        if current_mode == "categorical":
            self._rb_categorical.setChecked(True)
        else:
            self._rb_continuous.setChecked(True)
        mode_row.addWidget(self._rb_continuous)
        mode_row.addWidget(self._rb_categorical)
        mode_row.addStretch()
        layout.addLayout(mode_row)
        self._rb_continuous.toggled.connect(self._on_mode_radio_changed)

        self._grad_lbl = QLabel("Gradient colours")
        self._grad_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #555;")
        layout.addWidget(self._grad_lbl)
        self._gradient_widget = QWidget()
        gl = QHBoxLayout(self._gradient_widget)
        gl.setContentsMargins(8, 4, 8, 4)
        gl.addWidget(QLabel("Low:"))
        self._swatch_low = ClickableSwatch(self._gradient_colours[0], 16)
        self._swatch_low.clicked.connect(lambda: self._pick_gradient_colour(0))
        gl.addWidget(self._swatch_low)
        gl.addSpacing(12)
        gl.addWidget(QLabel("High:"))
        self._swatch_high = ClickableSwatch(self._gradient_colours[1], 16)
        self._swatch_high.clicked.connect(lambda: self._pick_gradient_colour(1))
        gl.addWidget(self._swatch_high)
        gl.addStretch()
        layout.addWidget(self._gradient_widget)

        self._cat_lbl = QLabel("Category colours")
        self._cat_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #555;")
        layout.addWidget(self._cat_lbl)
        self._cat_widget = QWidget()
        cat_outer = QVBoxLayout(self._cat_widget)
        cat_outer.setContentsMargins(8, 4, 8, 4)
        cat_outer.setSpacing(4)

        n_cats = len(self._unique_values)
        default_palette = self._default_palette(n_cats)
        for i, val in enumerate(self._unique_values):
            key = str(val)
            if key not in self._category_colours:
                self._category_colours[key] = default_palette[i % len(default_palette)]
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl_val = QLabel(str(val))
            lbl_val.setFixedWidth(120)
            lbl_val.setStyleSheet("font-size: 11px; color: #333;")
            sw = ClickableSwatch(self._category_colours[key], 14)
            sw.clicked.connect(
                lambda checked=False, k=key, s=sw: self._pick_cat_colour(k, s)
            )
            self._cat_swatches[key] = sw
            row.addWidget(sw)
            row.addWidget(lbl_val)
            row.addStretch()
            cat_outer.addLayout(row)

        if n_cats > 8:
            scroll = QScrollArea()
            scroll.setWidget(self._cat_widget)
            scroll.setWidgetResizable(True)
            scroll.setMaximumHeight(250)
            self._cat_scroll_or_widget = scroll
            layout.addWidget(scroll)
        else:
            self._cat_scroll_or_widget = self._cat_widget
            layout.addWidget(self._cat_widget)

        self._set_section_visibility(current_mode)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_edge_outside_colour(self):
        """Open a colour picker for the outside cell edge colour."""
        current = self.edge_outside_colour
        c = QColorDialog.getColor(QColor(current), self, "Edge colour for outside cells")
        if c.isValid():
            self.edge_outside_colour = c.name()
            self._edge_outside_swatch.set_colour(c.name())

    def _pick_edge_inside_colour(self):
        """Open a colour picker for the inside cell edge colour."""
        current = self.edge_inside_colour
        c = QColorDialog.getColor(QColor(current), self, "Edge colour for inside cells")
        if c.isValid():
            self.edge_inside_colour = c.name()
            self._edge_inside_swatch.set_colour(c.name())

    def _pick_bg_colour(self):
        """Open a colour picker for the canvas background colour."""
        current = self.bg_colour
        c = QColorDialog.getColor(QColor(current), self, "Canvas background colour")
        if c.isValid():
            self.bg_colour = c.name()
            self._bg_swatch.set_colour(c.name())

    def _on_mode_radio_changed(self, continuous_checked: bool):
        mode = "continuous" if continuous_checked else "categorical"
        self._mode = mode
        self._set_section_visibility(mode)

    def get_mode(self) -> str:
        return self._mode

    def _set_section_visibility(self, mode: str):
        """Show only the section for the given mode ('continuous' or 'categorical')."""
        is_cat = mode == "categorical"
        self._grad_lbl.setVisible(not is_cat)
        self._gradient_widget.setVisible(not is_cat)
        self._cat_lbl.setVisible(is_cat)
        self._cat_scroll_or_widget.setVisible(is_cat)
        self.adjustSize()

    @staticmethod
    def _default_palette(n):
        base = [
            "#e6194b",
            "#3cb44b",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabed4",
            "#469990",
            "#dcbeff",
            "#9a6324",
            "#800000",
            "#aaffc3",
            "#808000",
            "#000075",
            "#a9a9a9",
        ]
        if n <= len(base):
            return base[:n]
        return (base * ((n // len(base)) + 1))[:n]

    def _pick_gradient_colour(self, idx):
        initial = QColor(self._gradient_colours[idx])
        c = QColorDialog.getColor(initial, self, "Choose gradient colour")
        if c.isValid():
            self._gradient_colours[idx] = c.name()
            if idx == 0:
                self._swatch_low.set_colour(c.name())
            else:
                self._swatch_high.set_colour(c.name())

    def _pick_cat_colour(self, key, swatch):
        initial = QColor(self._category_colours.get(key, "#888888"))
        c = QColorDialog.getColor(initial, self, f"Colour for '{key}'")
        if c.isValid():
            self._category_colours[key] = c.name()
            swatch.set_colour(c.name())

    def get_gradient_colours(self) -> tuple:
        return tuple(self._gradient_colours)

    def get_category_colours(self) -> dict:
        return dict(self._category_colours)


@dataclass
class Session:
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    headings: List = field(default_factory=list)
    ids: pd.Series = field(default_factory=pd.Series)
    xyz: np.ndarray = field(default_factory=lambda: np.array([]))
    properties: np.ndarray = field(default_factory=lambda: np.array([]))
    num_cells: int = 0

    results_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    dist: pd.DataFrame = field(default_factory=pd.DataFrame)
    info: pd.DataFrame = field(default_factory=pd.DataFrame)

    outside_bool1: np.ndarray = field(default_factory=lambda: np.array([]))
    outside_ids1: np.ndarray = field(default_factory=lambda: np.array([]))
    outside_bool2: np.ndarray = field(default_factory=lambda: np.array([]))
    outside_ids2: np.ndarray = field(default_factory=lambda: np.array([]))
    inside_ids2: np.ndarray = field(default_factory=lambda: np.array([]))

    nbr_matrix1: np.ndarray = field(default_factory=lambda: np.array([]))
    nbr_matrix2: np.ndarray = field(default_factory=lambda: np.array([]))

    dist_matrix1: np.ndarray = field(default_factory=lambda: np.array([]))
    dist_matrix2: np.ndarray = field(default_factory=lambda: np.array([]))

    cavity_pts: list = field(default_factory=list)
    migration: pd.DataFrame = field(default_factory=pd.DataFrame)
    migration_lines: list = field(default_factory=list)
    cav_adj_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    cav_adj_bool: np.ndarray = field(default_factory=lambda: np.array([]))

    # ICM outlier exclusion
    icm_outlier_ids: list = field(default_factory=list)
    icm_outlier_bool: np.ndarray = field(default_factory=lambda: np.array([]))
    icm_outlier_std: float = 1.7  # std threshold for auto outlier detection
    icm_outlier_faces: List = field(default_factory=list)

    mean_cav_icm_distance: float = 0.0
    mean_te_distance: float = 0.0
    mean_cav_icm_distance_norm: float = 0.0

    # Angle to exclude points not on ICM surface
    angle_threshold: float = 30
    use_pe_centroid: bool = False

    # Threshold params
    thresh_method: str = "Automatic (cell position dependent)"
    thresh_k: float = 0.5
    thresh_vals: List[float] = field(default_factory=list)

    is_checkpoint: bool = False
    outside_loaded: bool = False
    cavity_loaded: bool = False

    extra_sheets: dict = field(default_factory=dict)

    def load(self, filepath: Path) -> None:
        fp = str(filepath).lower()
        if fp.endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(filepath)
            sheet_names = xl.sheet_names

            self.extra_sheets = {
                name: xl.parse(name, header=0)
                for name in sheet_names
                if name not in {"Data", "Results", "Distances", "Migration"}
            }

            if "Results" in sheet_names and "Distances" in sheet_names:
                results_sheet = xl.parse("Results", header=0)
                dist_sheet = xl.parse("Distances", header=0)
                data_sheet = xl.parse("Data", header=0)
                self._populate_from_data_sheet(data_sheet)
                self.results_df = results_sheet.copy()
                self.dist = dist_sheet.copy()
                self.is_checkpoint = True
                self._restore_checkpoint()
            else:
                raw = xl.parse(sheet_names[0], header=0)
                self._populate_from_data_sheet(raw)
                self.results_df = self.df.copy()

        elif fp.endswith(".csv"):
            raw = pd.read_csv(filepath)
            self._populate_from_data_sheet(raw)
            self.results_df = self.df.copy()
        else:
            raise ValueError(f"Unsupported file type: {filepath}")

    def _populate_from_data_sheet(self, df: pd.DataFrame) -> None:
        original_cols = [
            c
            for c in df.columns
            if c not in {"outside_bool", "cell_lineage"}
            and not str(c).startswith(
                (
                    "num_nbrs",
                    "nbr_",
                    "threshold",
                    "cavity_adj",
                    "cell_lineage",
                    "outside_bool",
                )
            )
        ]
        if len(original_cols) < 4:
            original_cols = list(df.columns)
        self.df = df[original_cols].reset_index(drop=True)
        self.headings = list(self.df.columns)
        self.ids = self.df[self.headings[0]]
        self.xyz = np.asarray(self.df[self.headings[1:4]])
        self.properties = np.asarray(self.df[self.headings[4:-1]])
        self.num_cells = len(self.ids)

    def _restore_checkpoint(self) -> None:
        if "outside_bool" in self.results_df.columns:
            ob = np.asarray(self.results_df["outside_bool"])
            self.outside_bool1 = ob.copy()
            self.outside_ids1 = np.where(ob == 1)[0]
            self.outside_bool2 = ob.copy()
            self.outside_ids2 = np.where(ob == 1)[0]
            self.inside_ids2 = np.where(ob == 0)[0]
            self.outside_loaded = True
        if "cavity_adj_bool" in self.results_df.columns:
            cab = np.asarray(self.results_df["cavity_adj_bool"])
            self.cav_adj_bool = cab.copy()
            self.cav_adj_ids = np.where(cab == 1)[0]
            if len(self.cav_adj_ids) > 0:
                centroid = self.xyz[self.cav_adj_ids].mean(axis=0)
                self.cavity_pts = [centroid]
                self.results_df["cavity_adj_bool"] = cab
            self.cavity_loaded = True

    def save_checkpoint(self, output_path: Path) -> None:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            self.df.to_excel(writer, sheet_name="Data", index=False)
            self.results_df.to_excel(writer, sheet_name="Results", index=False)
            self.dist.to_excel(writer, sheet_name="Distances", index=False)
            if hasattr(self, "migration") and not self.migration.empty:
                self.migration.to_excel(writer, sheet_name="Migration", index=False)

            self.info.to_excel(writer, sheet_name="Info", index=False)

def classify_outside(pts, n):
    hull = ConvexHull(pts)
    outside_ids = np.unique(hull.simplices)
    outside_bool = np.zeros(n)
    outside_bool[outside_ids] = 1
    return outside_bool, outside_ids


def detect_icm_outliers_numneighbours(
    inside_ids: np.ndarray, nbr_matrix,
):
    if len(inside_ids) < 4:
        return []
    
    num_neighbours = {}

    cell1_ids, cell2_ids = np.where(np.triu(nbr_matrix, k=1) == 1)

    for id1, id2 in zip(cell1_ids, cell2_ids):
        if id1 in inside_ids and id2 in inside_ids:
            if id1 in num_neighbours:
                num_neighbours[id1] += 1
            else:
                num_neighbours[id1] = 1

            if id2 in num_neighbours:
                num_neighbours[id2] += 1
            else:
                num_neighbours[id2] = 1

    more_than_two = [int(ins_id) for ins_id, num_nbr in num_neighbours.items() if num_nbr >= 2]
    outliers = [ins_id for ins_id in inside_ids if ins_id not in more_than_two]
    return outliers


def detect_icm_outliers(
    data_xyz: np.ndarray, inside_ids: np.ndarray, std_threshold
):
    """Detect ICM cells that are migrating away from the main ICM cluster.

    Uses distance from the ICM centroid: cells further than
    ``std_threshold`` standard-deviations from the mean distance are
    flagged as outliers.

    Returns
    -------
    outlier_global_ids : list[int]
        Global indices of outlier ICM cells.
    """
    if len(inside_ids) < 4:
        return []
    icm_pts = data_xyz[inside_ids]

    n_icm = len(icm_pts)

    dists = cdist(icm_pts, icm_pts)  # (n_icm, n_icm)
    dists += np.eye(n_icm, n_icm) * 1e8
    nearest_dist = dists.min(axis=1)
    nearest_ids = np.argmin(dists, axis=1)

    mean_dist = np.mean(nearest_dist)
    median_dist = np.median(nearest_dist)

    dists[nearest_ids] = 1e8

    for i in nearest_ids:
        dists[i] = 1e8
    second_nearest = dists.min(axis=1)

    threshold = median_dist * std_threshold

    outliers_bool = (nearest_dist > threshold) * (second_nearest > threshold)

    return [int(inside_ids[i]) for i in np.where(outliers_bool)[0]]


def detect_cavity_adjacent_threshold(
    data_xyz,
    inside_ids: List[int],
    outside_ids: List[int],
    outlier_ids: List[int],
    threshold: float = 0.5,
) -> List[int]:
    filtered_inside_ids = [i for i in inside_ids if i not in outlier_ids]

    if len(filtered_inside_ids) == 0 or len(outside_ids) == 0:
        return []

    icm_pts = data_xyz[filtered_inside_ids]
    te_pts = data_xyz[outside_ids]

    te_centroid = te_pts.mean(axis=0)
    icm_centroid = icm_pts.mean(axis=0)

    # Compute smallest principal component of icm_pts
    icm_centered = icm_pts - icm_centroid
    cov = np.cov(icm_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # smallest principal component
    vhat = eigvecs[:, np.argmin(eigvals)]

    # ensure vhat points roughly toward te_centroid (optional)
    if np.dot(vhat, te_centroid - icm_centroid) < 0:
        vhat = -vhat

    # project points onto direction vector
    icm_proj = (icm_pts - icm_centroid) @ vhat
    te_proj = (te_pts - icm_centroid) @ vhat

    # span of ICM projections
    proj_min = icm_proj.min()
    proj_max = icm_proj.max()
    span = proj_max - proj_min
    if span == 0:
        return []

    tlim = proj_min + threshold * span

    # select cavity-adjacent points
    icm_cavity_ids = [idx for idx, p in zip(filtered_inside_ids, icm_proj) if p > tlim]
    pe_cavity_ids = [idx for idx, p in zip(outside_ids, te_proj) if p > tlim]

    cavity_ids = icm_cavity_ids + pe_cavity_ids
    return cavity_ids


def tangent_frame(w):
    w = w / np.linalg.norm(w)

    # Pick vector not parallel to n
    if w[0] <= w[1] and w[0] <= w[2]:
        a = np.array([1, 0, 0])
    elif w[1] <= w[2]:
        a = np.array([0, 1, 0])
    else:
        a = np.array([0, 0, 1])

    u = np.cross(w, a)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    return u, v, w


def flat_frame(X):
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]  # descending – largest first
    u = eigvecs[:, order[0]]  # first  PC (largest variance)
    v = eigvecs[:, order[1]]  # second PC
    w = eigvecs[:, order[2]]  # smallest PC = plane normal

    return u, v, w


def detect_cavity_adjacent_angle(
    data_xyz,
    inside_ids: np.ndarray,
    outside_ids: np.ndarray,
    outlier_ids: List[int],
    angle_threshold_deg: float,
    use_pe_centroid: bool
) -> List[int]:
    filtered_inside_ids = [i for i in inside_ids if i not in outlier_ids]

    if len(filtered_inside_ids) == 0 or len(outside_ids) == 0:
        return []

    icm_pts = data_xyz[filtered_inside_ids]  # (N, 3)
    te_pts = data_xyz[outside_ids]

    te_centroid = te_pts.mean(axis=0)
    icm_centroid = icm_pts.mean(axis=0)
    icm_to_te = te_centroid - icm_centroid
    icm_centered = icm_pts - icm_centroid

    if use_pe_centroid:
        u,v,w = tangent_frame(icm_to_te)
    else:
        u,v,w = flat_frame(icm_centered)

    # orient normal toward the TE centroid
    if np.dot(w, icm_to_te) < 0:
        w = -w

    # 2-D projection for Delaunay
    pts_2d = np.column_stack([icm_centered @ u, icm_centered @ v])

    try:
        tri = Delaunay(pts_2d)
    except Exception:
        return []

    n_icm = len(icm_pts)
    neighbours: Dict[int, set] = {i: set() for i in range(n_icm)}
    for simplex in tri.simplices:
        for a, b in combinations(simplex, 2):
            neighbours[a].add(b)
            neighbours[b].add(a)

    angle_threshold_rad = np.radians(angle_threshold_deg)

    keep_mask = np.ones(n_icm, dtype=bool)
    for i in range(n_icm):
        for j in neighbours[i]:
            edge = icm_pts[j] - icm_pts[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            # signed component along the cavity-facing normal
            comp_normal = np.dot(edge, w)
            slope_angle = np.arcsin(np.clip(abs(comp_normal) / edge_len, -1.0, 1.0))
            if comp_normal > 0 and slope_angle > angle_threshold_rad:
                keep_mask[i] = False
                break

    cavity_local = np.where(keep_mask)[0]
    if len(cavity_local) == 0:
        return []

    icm_cavity_ids = [filtered_inside_ids[i] for i in cavity_local]
    cavity_icm_pts = icm_pts[cavity_local]

    cav_centroid = cavity_icm_pts.mean(axis=0)
    cav_centered = cavity_icm_pts - cav_centroid
    if len(cavity_icm_pts) >= 3:
        cov2 = np.cov(cav_centered, rowvar=False)
        eigvals2, eigvecs2 = np.linalg.eigh(cov2)
        vhat = eigvecs2[:, np.argmin(eigvals2)]
    else:
        vhat = w

    if np.dot(vhat, te_centroid - icm_centroid) < 0:
        vhat = -vhat

    te_proj = (te_pts - cav_centroid) @ vhat
    icm_cav_proj = (cavity_icm_pts - cav_centroid) @ vhat
    plane_threshold = np.median(icm_cav_proj)

    pe_cavity_ids = [idx for idx, p in zip(outside_ids, te_proj) if p > plane_threshold]

    return icm_cavity_ids + pe_cavity_ids + outlier_ids


def nbr_matrix(session):
    dt = Delaunay(session.xyz)
    n = session.num_cells
    mat = np.zeros((n, n))
    for simplex in dt.simplices:
        for i in simplex:
            for j in simplex:
                if i != j:
                    mat[i, j] = 1
    return mat


def eval_threshold(session):
    session.dist_matrix1 = np.zeros((session.num_cells, session.num_cells))
    session.dist_matrix1[:] = np.nan
    for c1 in range(session.num_cells):
        for c2 in range(session.num_cells):
            if session.nbr_matrix1[c1, c2] == 1:
                session.dist_matrix1[c1, c2] = np.linalg.norm(
                    session.xyz[c1] - session.xyz[c2]
                )

    ob_sum = np.sum(session.outside_bool2)
    if ob_sum == 0 and session.thresh_method == "Automatic (cell position dependent)":
        session.thresh_method = "Automatic (cell position independent)"
    if (
        ob_sum == session.num_cells
        and session.thresh_method == "Automatic (cell position dependent)"
    ):
        session.thresh_method = "Automatic (cell position independent)"

    if session.thresh_method == "None":
        val = float(np.nanmax(session.dist_matrix1) * 100)
        session.thresh_vals = [val, val]
        session.results_df["threshold"] = ["-"] * session.num_cells
    elif session.thresh_method == "Manual":
        val = session.thresh_k
        session.thresh_vals = [val, val]
        session.results_df["threshold"] = (
            np.ones(session.num_cells) * val
        )
    elif session.thresh_method == "Automatic (cell position dependent)":
        k = session.thresh_k
        dists_out = session.dist_matrix1[session.outside_ids2, :]
        dists_in = session.dist_matrix1[session.inside_ids2, :]
        p75_out = np.nanpercentile(dists_out, 75)
        iqr_out = scipy.stats.iqr(dists_out, nan_policy="omit")
        p75_in = np.nanpercentile(dists_in, 75)
        iqr_in = scipy.stats.iqr(dists_in, nan_policy="omit")
        session.thresh_vals = [p75_out + (k * iqr_out), p75_in + (k * iqr_in)]
    elif session.thresh_method == "Automatic (cell position independent)":
        k = k = session.thresh_k
        p75 = np.nanpercentile(session.dist_matrix1, 75)
        iqr = scipy.stats.iqr(session.dist_matrix1, nan_policy="omit")
        val = p75 + (k * iqr)
        session.thresh_vals = [val, val]
    return session


def check_nbrs(session):
    nbr1 = session.nbr_matrix1
    dist1 = session.dist_matrix1
    session.nbr_matrix2 = nbr2 = np.copy(nbr1)
    session.dist_matrix2 = dist2 = np.copy(dist1)
    
    for i in range(session.num_cells):
        for j in range(session.num_cells):
            has_outside = session.outside_bool2[i] or session.outside_bool2[j]
            has_inside = (not session.outside_bool2[i]) or (not session.outside_bool2[j])
            thresh = session.thresh_vals[0] if has_outside else session.thresh_vals[1]
            if nbr1[i, j] == 1 and dist1[i, j] > thresh:
                nbr2[i, j] = 0
                dist2[i, j] = np.nan
    return session


def point_to_triangle_distance(p, a, b, c):
    """Return (distance, closest_point) from point p to triangle (a, b, c)."""
    # Vectors
    ab = b - a
    ac = c - a
    ap = p - a
    
    # Project p onto plane of triangle using a unit normal
    normal = np.cross(ab, ac)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-15:
        # Degenerate triangle – fall back to edge distances
        d1, cp1 = point_to_segment_distance(p, a, b)
        d2, cp2 = point_to_segment_distance(p, b, c)
        d3, cp3 = point_to_segment_distance(p, c, a)
        results = [(d1, cp1), (d2, cp2), (d3, cp3)]
        return min(results, key=lambda x: x[0])

    n_hat = normal / norm_len
    signed_dist = np.dot(ap, n_hat)
    closest_on_plane = p - signed_dist * n_hat
    
    # Check if projection lies inside triangle using barycentric coordinates
    v0 = ac
    v1 = ab
    v2 = closest_on_plane - a
    
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-15:
        d1, cp1 = point_to_segment_distance(p, a, b)
        d2, cp2 = point_to_segment_distance(p, b, c)
        d3, cp3 = point_to_segment_distance(p, c, a)
        results = [(d1, cp1), (d2, cp2), (d3, cp3)]
        return min(results, key=lambda x: x[0])

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    if u >= 0 and v >= 0 and u + v <= 1:
        # Projection is inside triangle
        return abs(signed_dist), closest_on_plane

    # Otherwise find closest point on edges
    d1, cp1 = point_to_segment_distance(p, a, b)
    d2, cp2 = point_to_segment_distance(p, b, c)
    d3, cp3 = point_to_segment_distance(p, c, a)
    results = [(d1, cp1), (d2, cp2), (d3, cp3)]
    return min(results, key=lambda x: x[0])

def point_to_segment_distance(p, a, b):
    """Return (distance, closest_point) from point p to segment (a, b)."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest), closest


def compile_results(session):
    session.results_df["num_nbrs"] = np.sum(session.nbr_matrix2, axis=1)
    om = session.outside_bool2.astype(float)
    session.results_df["num_nbrs_outside"] = session.nbr_matrix2 @ om
    session.results_df["num_nbrs_inside"] = session.nbr_matrix2 @ (1 - om)
    session.results_df["nbr_ids"] = [""] * session.num_cells
    for c1 in range(session.num_cells):
        nbrs = np.where(session.nbr_matrix2[c1, :] == 1)[0]
        session.results_df.loc[c1, "nbr_ids"] = str(
            session.results_df["ID"].values[nbrs]
        )
    session.results_df["nbr_dist_mean"] = np.zeros(session.num_cells)
    session.results_df["nbr_dist_range"] = np.zeros(session.num_cells)
    for c1 in range(session.num_cells):
        row = session.dist_matrix2[c1, :]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            session.results_df.loc[c1, "nbr_dist_mean"] = np.mean(valid)
            session.results_df.loc[c1, "nbr_dist_range"] = np.max(valid) - np.min(valid)
        else:
            session.results_df.loc[c1, "nbr_dist_mean"] = np.nan
            session.results_df.loc[c1, "nbr_dist_range"] = np.nan
    return session


def compile_distances(session):
    cell1_ids, cell2_ids = np.where(np.triu(session.nbr_matrix2, k=1) == 1)
    ids = session.results_df["ID"].values
    ob = session.results_df["outside_bool"].values
    dist_data = {
        "cell_id1": ids[cell1_ids],
        "cell_id2": ids[cell2_ids],
        "outside_bool1": ob[cell1_ids],
        "outside_bool2": ob[cell2_ids],
        "nbr_dist": session.dist_matrix2[cell1_ids, cell2_ids],
    }
    if len(session.cavity_pts) > 0:
        cab = session.results_df["cavity_adj_bool"].values
        dist_data["cavity_adj_bool1"] = cab[cell1_ids]
        dist_data["cavity_adj_bool2"] = cab[cell2_ids]
    session.dist = pd.DataFrame(dist_data)
    return session


def get_neighbour_distances(xyz, ids_list):
    nbr_distances = []
    for te_id in ids_list:
        other_te_ids = [id for id in ids_list if id != te_id]
        point_xyz = xyz[[te_id]]
        other_xyz = xyz[other_te_ids]

        te_distances = cdist(point_xyz, other_xyz)
        dist = te_distances.min()
        nbr_distances.append(dist)

    return nbr_distances

def compile_migration(session):

    # will normalize by the mean distance of trophectoderm cells (te) to their nearest te neighbour

    te_ids = session.outside_ids2
    faces = session.icm_outlier_faces
    outlier_ids = session.icm_outlier_ids
    distances = {"ID": [], "distance_to_icm": [], "distance_to_icm_norm": []}
    # Store line endpoints: list of (outlier_xyz, closest_surface_xyz)
    migration_lines = []
    

    if len(outlier_ids) == 0 or len(te_ids) <= 1:
        session.migration = pd.DataFrame()
        session.migration_lines = migration_lines
        return session

    te_nbr_distances = get_neighbour_distances(session.xyz, te_ids)
    mean_te_spacing = np.mean(te_nbr_distances)

    for pid in outlier_ids:
        point_xyz = session.xyz[pid]
        
        best_dist = float('inf')
        best_closest_pt = None
        for face in faces:
            a, b, c = face
            dist, closest_pt = point_to_triangle_distance(point_xyz, a, b, c)
            if dist < best_dist:
                best_dist = dist
                best_closest_pt = closest_pt

        shortest_dist_norm = best_dist / mean_te_spacing

        distances["ID"].append(pid)
        distances["distance_to_icm"].append(best_dist)
        distances["distance_to_icm_norm"].append(shortest_dist_norm)

        if best_closest_pt is not None:
            migration_lines.append((point_xyz.copy(), best_closest_pt.copy()))

    session.migration = pd.DataFrame(distances)
    session.migration_lines = migration_lines
    return session


def assign_cell_lineage(session):
    lineage = []
    for _, row in session.results_df.iterrows():
        o = row["outside_bool"]
        c = row.get("cavity_adj_bool", 0)
        if o == 1 and c == 0:
            lineage.append("mural trophectoderm")
        elif o == 1 and c == 1:
            lineage.append("polar trophectoderm")
        elif o == 0:
            lineage.append("inner cell mass")
        else:
            lineage.append("unknown")
    session.results_df["cell_lineage"] = lineage
    return session


_CATEGORIES = ("inside", "outside", "cavity", "not_cavity")


class EmbryoCanvas(FigureCanvasQTAgg):
    cell_picked = pyqtSignal(int)

    COL_MIG = np.array([134 / 256, 66 / 256, 201 / 256, 1.0]) # purple
    COL_ALL = np.array([255 / 256, 102 / 256, 204 / 256, 1.0])  # pink
    COL_CAV = np.array([255 / 256, 204 / 256, 0.0, 1.0])  # yellow
    COL_IN = np.array([0.3, 0.7, 1.0, 1.0])  # blue

    _WIRE_STYLE = {
        "inside": ("dodgerblue", 0.5, 0.8),
        "outside": ("hotpink", 0.5, 0.8),
        "cavity": ("goldenrod", 0.5, 0.8),
        "not_cavity": ("mediumpurple", 0.5, 0.8),
    }
    _VOL_STYLE = {
        "inside": ("dodgerblue", "dodgerblue", 0.08, 0.3),
        "outside": ("hotpink", "hotpink", 0.08, 0.3),
        "cavity": ("goldenrod", "gold", 0.10, 0.5),
        "not_cavity": ("mediumpurple", "mediumpurple", 0.08, 0.3),
    }

    session: Session
    face_col: np.ndarray
    edge_col: np.ndarray

    ax: Axes3D
    sct: Path3DCollection

    _channel_settings: dict
    _active_channel: str
    _legend_visible: bool
    _legend_artists: list

    def __init__(self, parent=None, dpi=100):
        self.fig = Figure(dpi=dpi)
        self._bg_colour = "#f0f0f0"
        self.fig.patch.set_facecolor(self._bg_colour)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        self._outer_hull_lines = []
        self._outer_hull_visible = True
        self._nbr_lines = []
        self._migration_line_artists = []

        self._wire_artists = {k: [] for k in _CATEGORIES}
        self._vol_artists = {k: [] for k in _CATEGORIES}
        self._wire_visible = {k: False for k in _CATEGORIES}
        self._vol_visible = {k: False for k in _CATEGORIES}
        self._pts_hidden = {k: False for k in _CATEGORIES}

        self._icm_surface_artists = []
        self._icm_surface_visible = False

        self._channel_settings = {}
        self._active_channel = ""
        self._legend_visible = True
        self._legend_artists = []

        # Colour settings for the special modes
        self._outside_colours = {
            "outside": to_hex(self.COL_ALL[:3]),  # type: ignore
            "inside": to_hex(self.COL_IN[:3]),  # type: ignore
        }
        self._cavity_colours = {
            "cavity": to_hex(self.COL_CAV[:3]),  # type: ignore
            "inside": to_hex(self.COL_IN[:3]),  # type: ignore
            "outside": to_hex(self.COL_ALL[:3]),  # type: ignore
        }
        self._migration_colours = {
            "outside": to_hex(self.COL_ALL[:3]),  # type: ignore
            "inside": to_hex(self.COL_IN[:3]),  # type: ignore
            "migration": to_hex(self.COL_MIG[:3]),  # type: ignore
        }

        # Edge colours for scatter main points (outside = black ring, inside = white ring)
        self._edge_outside_colour = "#000000"
        self._edge_inside_colour = "#ffffff"

        # Instance-level copies of style dicts so colours can be customised
        self._WIRE_STYLE = dict(self.__class__._WIRE_STYLE)
        self._VOL_STYLE = dict(self.__class__._VOL_STYLE)

        self.fig.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def get_cell_canvas_positions(self):
        """Returns an N x 2 array of pixel coordinates for all cells."""
        if self.ax is None or self.session.xyz.size == 0:
            return np.array([])

        points_3d = self.session.xyz
        x2, y2, _ = proj3d.proj_transform(
            points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], self.ax.get_proj()
        )
        points_projected_2d = np.column_stack((x2, y2))
        pixel_coords = self.ax.transData.transform(points_projected_2d)

        return pixel_coords

    def _on_canvas_click(self, event):
        if event.x is None or event.y is None or self.session.xyz.size == 0:
            return

        hidden_mask = self._compute_hidden_mask()

        cell_positions_2d = self.get_cell_canvas_positions()
        click_point = np.array([event.x, event.y])
        distances = np.linalg.norm(cell_positions_2d - click_point, axis=1)
        closest_idx = np.argmin(distances)

        if distances[closest_idx] < 15 and not hidden_mask[closest_idx]:
            self.cell_picked.emit(int(closest_idx))

    def load_session(self, session):
        self.session = session
        n = session.num_cells
        self.face_col = np.tile(self.COL_ALL, (n, 1))
        self.edge_col = np.ones((n, 4))
        self._size = 120 * np.ones(n)
        self.redraw()

    def redraw(self):
        """Full redraw of scatter + all enabled overlays."""
        self.fig.clf()
        s = self.session
        if s is None or s.num_cells == 0:
            self.draw()
            return

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.grid(False)
        self.ax.axis("off")
        self.ax.set_facecolor(self._bg_colour)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self._set_equal_3d()

        df, de, ds = self._apply_hidden(
            self.face_col.copy(), self.edge_col.copy(), self._size.copy()
        )

        self.sct = self.ax.scatter(
            xs=s.xyz[:, 0],
            ys=s.xyz[:, 1],
            zs=s.xyz[:, 2],  # type: ignore
            facecolors=df,
            edgecolors=de,
            picker=True,
            s=ds,
            linewidth=2,
            alpha=1,
        )

        self._outer_hull_lines = []
        if self._outer_hull_visible:
            hull = ConvexHull(s.xyz)
            for simplex in hull.simplices:
                tri = np.append(simplex, simplex[0])
                (ln,) = self.ax.plot(
                    s.xyz[tri, 0],
                    s.xyz[tri, 1],
                    s.xyz[tri, 2],
                    "-",
                    color="#8888aa",
                    linewidth=0.8,
                    alpha=0.4,
                )
                self._outer_hull_lines.append(ln)

        self._migration_line_artists = []
        for cat in _CATEGORIES:
            self._wire_artists[cat] = []
            self._vol_artists[cat] = []
        self._icm_surface_artists = []
        self._rebuild_all_overlays()
        self._rebuild_icm_surface()
        self._legend_artists = []
        if self._active_channel:
            self._draw_legend()
        self.draw()

    def update_colours(self):
        """Lightweight colour/size refresh (no full redraw)."""
        if self.sct is None:
            return
        df, de, ds = self._apply_hidden(
            self.face_col.copy(), self.edge_col.copy(), self._size.copy()
        )
        self.sct.set_facecolors(df)  # type: ignore
        self.sct.set_edgecolors(de)  # type: ignore
        self.sct.set_sizes(ds)
        self.draw_idle()

    def update_edge_for_outside(self):
        s = self.session
        if s is None or len(s.outside_bool2) == 0:
            return
        outside_rgba = np.array(to_rgba(self._edge_outside_colour))
        inside_rgba = np.array(to_rgba(self._edge_inside_colour))
        self.edge_col = np.tile(inside_rgba, (s.num_cells, 1))
        self.edge_col[np.where(s.outside_bool2 == 1)[0]] = outside_rgba
        self.update_colours()

    def colour_by_channel(self, column_name):
        s = self.session
        self._active_channel = column_name
        settings = self._channel_settings.get(column_name, {})
        mode = settings.get("mode", "continuous")

        vals = np.asarray(s.df[column_name])

        if mode == "categorical":
            cat_colours = settings.get("categories", {})
            for i, v in enumerate(vals):
                key = str(v)
                hex_c = cat_colours.get(key, "#888888")
                self.face_col[i] = to_rgba(hex_c)
        else:
            fvals = vals.astype(float)
            vmin, vmax = np.nanmin(fvals), np.nanmax(fvals)
            norm = (
                np.zeros_like(fvals)
                if vmax - vmin < 1e-12
                else (fvals - vmin) / (vmax - vmin)
            )
            gradient = settings.get("gradient", ("#ff66cc", "#ffcc00"))
            cmap = LinearSegmentedColormap.from_list(
                "custom", [gradient[0], gradient[1]], N=256
            )
            self.face_col = cmap(norm)

        self.update_colours()
        self._draw_legend()

    def colour_by_outside(self):
        s = self.session
        self._active_channel = "__outside__"
        if len(s.outside_bool2) == 0:
            return
        col_out = to_rgba(self._outside_colours["outside"])
        col_in = to_rgba(self._outside_colours["inside"])
        for i in range(s.num_cells):
            self.face_col[i] = col_out if s.outside_bool2[i] == 1 else col_in
        self.update_colours()
        self._draw_legend()

    def colour_by_cavity(self):
        s = self.session
        self._active_channel = "__cavity__"
        if len(s.cav_adj_bool) == 0:
            return
        col_cav = to_rgba(self._cavity_colours["cavity"])
        col_in = to_rgba(self._cavity_colours["inside"])
        col_out = to_rgba(self._cavity_colours["outside"])
        for i in range(s.num_cells):
            if s.cav_adj_bool[i] == 1:
                self.face_col[i] = col_cav
            elif len(s.outside_bool2) > 0 and s.outside_bool2[i] == 0:
                self.face_col[i] = col_in
            else:
                self.face_col[i] = col_out
        self.update_colours()
        self._draw_legend()

    def colour_by_migration(self):
        s = self.session
        self._active_channel = "__migration__"
        if len(s.cav_adj_bool) == 0:
            return
        col_in = to_rgba(self._migration_colours["inside"])
        col_out = to_rgba(self._migration_colours["outside"])
        col_mig = to_rgba(self._migration_colours["migration"])
        for i in range(s.num_cells):
            if len(s.icm_outlier_bool) > 0 and s.icm_outlier_bool[i] == 1:
                self.face_col[i] = col_mig
            elif len(s.outside_bool2) > 0 and s.outside_bool2[i] == 0:
                self.face_col[i] = col_in
            else:
                self.face_col[i] = col_out
        self.update_colours()
        self._draw_legend()

    def draw_neighbour_lines(self):
        self._clear_list(self._nbr_lines)
        self._nbr_lines = []
        s = self.session
        if s is None or len(s.nbr_matrix2) == 0 or self.ax is None:
            return
        for i in range(s.num_cells):
            for j in range(i + 1, s.num_cells):
                if s.nbr_matrix2[i, j] == 1:
                    (ln,) = self.ax.plot(
                        [s.xyz[i, 0], s.xyz[j, 0]],
                        [s.xyz[i, 1], s.xyz[j, 1]],
                        [s.xyz[i, 2], s.xyz[j, 2]],
                        "-",
                        color="#9999bb",
                        linewidth=0.75,
                        alpha=0.5,
                    )
                    self._nbr_lines.append(ln)
        self.draw_idle()

    def clear_neighbour_lines(self):
        self._clear_list(self._nbr_lines)
        self._nbr_lines = []
        self.draw_idle()

    def draw_migration_lines(self):
        """Draw lines from each migrating cell to its nearest point on the ICM surface."""
        self._clear_list(self._migration_line_artists)
        self._migration_line_artists = []
        s = self.session
        if s is None or self.ax is None:
            return
        if not hasattr(s, 'migration_lines') or len(s.migration_lines) == 0:
            return
        for outlier_pt, surface_pt in s.migration_lines:
            (ln,) = self.ax.plot(
                [outlier_pt[0], surface_pt[0]],
                [outlier_pt[1], surface_pt[1]],
                [outlier_pt[2], surface_pt[2]],
                "-",
                color=self.COL_MIG,
                linewidth=2.0,
                alpha=0.8,
            )
            self._migration_line_artists.append(ln)
            # Small marker at the surface contact point
            sct = self.ax.scatter(
                [surface_pt[0]], [surface_pt[1]], [surface_pt[2]], # type: ignore
                color=self.COL_MIG, s=30, marker="x", linewidths=1.5, zorder=10,
            )
            self._migration_line_artists.append(sct)
        self.draw_idle()

    def clear_migration_lines(self):
        self._clear_list(self._migration_line_artists)
        self._migration_line_artists = []
        self.draw_idle()

    def set_outer_hull_visible(self, visible):
        self._outer_hull_visible = visible
        for ln in self._outer_hull_lines:
            ln.set_visible(visible)
        self.draw_idle()

    def set_background_colour(self, colour_hex: str):
        """Change the background colour of the figure and 3D axes."""
        self._bg_colour = colour_hex
        self.fig.patch.set_facecolor(colour_hex)
        if self.ax is not None:
            self.ax.set_facecolor(colour_hex)
        self.draw_idle()

    def set_points_hidden(self, category, hidden):
        self._pts_hidden[category] = hidden
        self.update_colours()

    def set_wireframe_visible(self, category, visible):
        self._wire_visible[category] = visible
        self._rebuild_wireframe(category)
        self.draw_idle()

    def set_volume_visible(self, category, visible):
        self._vol_visible[category] = visible
        self._rebuild_volume(category)
        self.draw_idle()

    def refresh_overlays(self):
        """Rebuild every currently-visible overlay from latest session data."""
        self._rebuild_all_overlays()
        self._rebuild_icm_surface()
        self.draw_idle()

    def _apply_hidden(self, fc, ec, sz):
        """Zero-out alpha/size for hidden points."""
        h = self._compute_hidden_mask()
        fc[h, 3] = 0.0
        ec[h, 3] = 0.0
        sz[h] = 0.0
        return fc, ec, sz

    def _compute_hidden_mask(self):
        s = self.session
        n = s.num_cells
        mask = np.zeros(n, dtype=bool)
        ho = len(s.outside_bool2) > 0
        hc = len(s.cav_adj_bool) > 0
        if self._pts_hidden["inside"] and ho:
            mask |= s.outside_bool2 == 0
        if self._pts_hidden["outside"] and ho:
            mask |= s.outside_bool2 == 1
        if self._pts_hidden["cavity"] and hc:
            mask |= s.cav_adj_bool == 1
        if self._pts_hidden["not_cavity"] and hc:
            mask |= s.cav_adj_bool == 0
        return mask

    def _set_equal_3d(self):
        # https://github.com/matplotlib/matplotlib/blob/v3.10.8/lib/mpl_toolkits/mplot3d/axes3d.py#L395-L410
        def apply_aspect_nonsquare(position=None):
            if position is None:
                position = self.ax.get_position(original=True)
            self.ax._set_position(position, 'active') # type: ignore

        self.ax.apply_aspect = apply_aspect_nonsquare
        self.ax.set_box_aspect([1, 1, 1])

        _original_get_proj = self.ax.get_proj.__func__ # type: ignore
        ax_ref = self.ax

        def _corrected_get_proj():
            M = _original_get_proj(ax_ref)
            fig = ax_ref.get_figure()
            pos = ax_ref.get_position(original=False)
            fig_w, fig_h = fig.get_size_inches() * fig.dpi # type: ignore
            ax_w = pos.width * fig_w
            ax_h = pos.height * fig_h
            if ax_w == 0 or ax_h == 0:
                return M
            aspect = ax_w / ax_h
            correction = np.eye(4)
            if aspect > 1:
                correction[0, 0] = 1.0 / aspect
            else:
                correction[1, 1] = aspect
            return correction @ M

        self.ax.get_proj = _corrected_get_proj

        xyz = self.session.xyz
        mins, maxs = xyz.min(0), xyz.max(0)
        c = (mins + maxs) / 2
        r = (maxs - mins).max() / 2
        self.ax.set_xlim(c[0] - r, c[0] + r)
        self.ax.set_ylim(c[1] - r, c[1] + r)
        self.ax.set_zlim(c[2] - r, c[2] + r)

    @staticmethod
    def _clear_list(artists):
        for a in artists:
            try:
                a.remove()
            except Exception:
                pass

    def _points_for_category(self, category):
        """xyz array for category, or None if < 4 points."""
        s = self.session
        if s is None or s.num_cells == 0:
            return None
        ho = len(s.outside_bool2) > 0
        hc = len(s.cav_adj_bool) > 0
        if category == "inside" and ho:
            idx = np.where(s.outside_bool2 == 0)[0]
        elif category == "outside" and ho:
            idx = np.where(s.outside_bool2 == 1)[0]
        elif category == "cavity" and hc:
            idx = np.where(s.cav_adj_bool == 1)[0]
        elif category == "not_cavity" and hc:
            idx = np.where(s.cav_adj_bool == 0)[0]
        else:
            return None
        return s.xyz[idx] if len(idx) >= 4 else None

    def _rebuild_wireframe(self, cat):
        self._clear_list(self._wire_artists.get(cat, []))
        self._wire_artists[cat] = []
        if not self._wire_visible.get(cat) or self.ax is None:
            return
        pts = self._points_for_category(cat)
        if pts is None:
            return
        colour, alpha, lw = self._WIRE_STYLE[cat]
        try:
            hull = ConvexHull(pts)
        except Exception:
            return
        for simplex in hull.simplices:
            tri = np.append(simplex, simplex[0])
            (ln,) = self.ax.plot(
                pts[tri, 0],
                pts[tri, 1],
                pts[tri, 2],
                "-",
                color=colour,
                linewidth=lw,
                alpha=alpha,
                picker=False,
            )
            self._wire_artists[cat].append(ln)

    def _rebuild_volume(self, cat):
        self._clear_list(self._vol_artists.get(cat, []))
        self._vol_artists[cat] = []
        if not self._vol_visible.get(cat) or self.ax is None:
            return
        pts = self._points_for_category(cat)
        if pts is None:
            return
        ecol, fcol, alpha, lw = self._VOL_STYLE[cat]
        try:
            tri = Delaunay(pts)
        except Exception:
            return
        face_count = {}
        for tet in tri.simplices:
            for face in combinations(sorted(tet), 3):
                face = tuple(face)
                face_count[face] = face_count.get(face, 0) + 1
        boundary = [f for f, c in face_count.items() if c == 1]
        if not boundary:
            return
        polys = [pts[list(f)] for f in boundary]
        mesh = Poly3DCollection(
            polys,
            alpha=alpha,
            facecolor=fcol,
            edgecolor=ecol,
            linewidth=lw,
        )
        self.ax.add_collection3d(mesh)
        self._vol_artists[cat].append(mesh)

    def set_legend_visible(self, visible):
        self._legend_visible = visible
        if visible and self._active_channel:
            self._draw_legend()
        else:
            self._clear_legend()

    def set_wire_colour(self, category, colour_hex):
        """Update the wireframe colour for a category."""
        _, old_alpha, old_lw = self._WIRE_STYLE[category]
        self._WIRE_STYLE[category] = (colour_hex, old_alpha, old_lw)
        self._rebuild_wireframe(category)
        self.draw_idle()

    def set_vol_colour(self, category, colour_hex):
        """Update the volume colour for a category."""
        _, _, old_alpha, old_lw = self._VOL_STYLE[category]
        self._VOL_STYLE[category] = (colour_hex, colour_hex, old_alpha, old_lw)
        self._rebuild_volume(category)
        self.draw_idle()

    def _clear_legend(self):
        for a in self._legend_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._legend_artists = []
        self.draw_idle()

    def _draw_legend(self):
        self._clear_legend()
        if not self._legend_visible or not self._active_channel:
            return
        if self.ax is None:
            return

        if self._active_channel == "__outside__":
            self._draw_fixed_legend(
                [
                    ("Outside", self._outside_colours["outside"]),
                    ("Inside", self._outside_colours["inside"]),
                ],
            )
        elif self._active_channel == "__cavity__":
            self._draw_fixed_legend(
                [
                    ("Cavity-adjacent", self._cavity_colours["cavity"]),
                    ("Inside", self._cavity_colours["inside"]),
                    ("Outside", self._cavity_colours["outside"]),
                ],
            )
        elif self._active_channel == "__migration__":
            self._draw_fixed_legend(
                [
                    ("Inside", self._migration_colours["inside"]),
                    ("Outside", self._migration_colours["outside"]),
                    ("Migration", self._migration_colours["migration"]),
                ],
            )
        else:
            settings = self._channel_settings.get(self._active_channel, {})
            mode = settings.get("mode", "continuous")
            if mode == "continuous":
                self._draw_continuous_legend(settings)
            else:
                self._draw_categorical_legend(settings)
        self.draw_idle()

    def _draw_fixed_legend(self, items):
        """Draw a simple categorical legend with a title and list of (label, hex_colour) items."""
        start_y = 0.90
        line_h = 0.035
        x_swatch = 0.87
        x_text = 0.90

        for i, (label, hex_c) in enumerate(items):
            y = start_y - i * line_h
            if y < 0.05:
                break
            patch_ax = self.fig.add_axes((x_swatch, y - 0.008, 0.018, 0.018))
            patch_ax.set_xlim(0, 1)
            patch_ax.set_ylim(0, 1)
            patch_ax.set_axis_off()
            rect = Rectangle((0, 0), 1, 1, fc=hex_c, ec="none")
            patch_ax.add_patch(rect)
            self._legend_artists.append(patch_ax)

            txt = self.fig.text(
                x_text,
                y,
                label,
                fontsize=7,
                color="#444",
                va="center",
                transform=self.fig.transFigure,
            )
            self._legend_artists.append(txt)

    def _draw_continuous_legend(self, settings):
        gradient = settings.get("gradient", ("#ff66cc", "#ffcc00"))
        cmap = LinearSegmentedColormap.from_list(
            "custom", [gradient[0], gradient[1]], N=256
        )
        vals = np.asarray(self.session.df[self._active_channel], dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)

        # Inset axes for colourbar
        cbar_ax = self.fig.add_axes((0.87, 0.25, 0.03, 0.5))
        cbar_ax.set_facecolor("none")
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = self.fig.colorbar(sm, cax=cbar_ax)
        cb.ax.tick_params(labelsize=7, colors="#444")
        self._legend_artists.append(cbar_ax)

    def _draw_categorical_legend(self, settings):
        cat_colours = settings.get("categories", {})
        if not cat_colours:
            return
        items = list(cat_colours.items())
        n = len(items)
        start_y = 0.90
        line_h = 0.035
        x_swatch = 0.87
        x_text = 0.90

        for i, (val, hex_c) in enumerate(items):
            y = start_y - i * line_h
            if y < 0.05:
                break
            patch_ax = self.fig.add_axes((x_swatch, y - 0.008, 0.018, 0.018))
            patch_ax.set_xlim(0, 1)
            patch_ax.set_ylim(0, 1)
            patch_ax.set_axis_off()
            rect = Rectangle((0, 0), 1, 1, fc=hex_c, ec="none")
            patch_ax.add_patch(rect)
            self._legend_artists.append(patch_ax)

            txt = self.fig.text(
                x_text,
                y,
                str(val),
                fontsize=7,
                color="#444",
                va="center",
                transform=self.fig.transFigure,
            )
            self._legend_artists.append(txt)

    def set_icm_surface_visible(self, visible):
        """Toggle visibility of the ICM cavity-separating surface."""
        self._icm_surface_visible = visible
        self._rebuild_icm_surface()
        self.draw_idle()

    def _rebuild_icm_surface(self):
        self._clear_list(self._icm_surface_artists)
        self._icm_surface_artists = []
        if not self._icm_surface_visible or self.ax is None:
            return

        polys = self.session.icm_outlier_faces
        
        mesh = Poly3DCollection(
            polys,
            alpha=0.25,
            facecolor="goldenrod",
            edgecolor="darkgoldenrod",
            linewidth=0.6,
        )
        self.ax.add_collection3d(mesh)
        self._icm_surface_artists.append(mesh)

    def _rebuild_all_overlays(self):
        for cat in _CATEGORIES:
            self._rebuild_wireframe(cat)
            self._rebuild_volume(cat)


class ThresholdDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Neighbour Distance Threshold")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        lbl = QLabel("Select threshold method:")
        lbl.setStyleSheet("font-weight: 600; font-size: 12px; color: #1a1a1a;")
        layout.addWidget(lbl)

        self.method_group = QButtonGroup(self)
        for label, checked in [
            ("Automatic (cell position dependent)", True),
            ("Automatic (cell position independent)", False),
            ("Manual", False),
            ("None", False),
        ]:
            rb = QRadioButton(label)
            rb.setChecked(checked)
            self.method_group.addButton(rb)
            layout.addWidget(rb)

        self._val_row = QWidget()
        h = QHBoxLayout(self._val_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("k-value / threshold:"))
        self.val_edit = QLineEdit("0.5")
        self.val_edit.setFixedWidth(80)
        h.addWidget(self.val_edit)
        layout.addWidget(self._val_row)

        # Update visibility when method changes
        self.method_group.buttonToggled.connect(self._on_method_changed)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_method_changed(self, button, checked):
        if not checked:
            return
        method = button.text()
        # Hide k-value input
        needs_val = method not in ("None")
        self._val_row.setVisible(needs_val)

    def get_values(self):
        checked_button = self.method_group.checkedButton()
        if checked_button:
            text = checked_button.text()
        else:
            text = "None"
        return text, self.val_edit.text()


class CavitySettingsDialog(QDialog):
    """Dialog for configuring cavity detection settings (ICM outlier exclusion)."""

    def __init__(
        self,
        outlier_std,
        current_angle,
        use_pe_centroid,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Cavity Detection Settings")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        hdr1 = QLabel("ICM Outlier Exclusion")
        hdr1.setStyleSheet("font-weight: 700; font-size: 12px; color: #1a1a1a;")
        layout.addWidget(hdr1)

        row_std = QHBoxLayout()
        row_std.addWidget(QLabel("σ threshold:"))
        self._outlier_std_edit = QLineEdit(f"{outlier_std:.1f}")
        self._outlier_std_edit.setFixedWidth(60)
        row_std.addWidget(self._outlier_std_edit)
        row_std.addStretch()
        layout.addLayout(row_std)

        hdr2 = QLabel("Exclusion Angle")
        hdr2.setStyleSheet("font-weight: 700; font-size: 12px; color: #1a1a1a;")
        layout.addWidget(hdr2)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)
        theta_lbl = QLabel("θ")
        theta_lbl.setStyleSheet("font-size: 14px; font-weight: 600; color: #555;")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(current_angle)
        self.value_label = QLabel(str(current_angle) + "°")
        self.value_label.setFixedWidth(40)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setStyleSheet("font-size: 11px; color: #555;")
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(theta_lbl)
        slider_row.addWidget(self.slider)
        slider_row.addWidget(self.value_label)
        layout.addLayout(slider_row)

        self.checkbox = QCheckBox('Align ICM surface to TE centroid')
        self.checkbox.setChecked(use_pe_centroid)
        layout.addWidget(self.checkbox)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_slider_changed(self, val):
        self.value_label.setText(str(val) + "°")

    def get_values(self) -> dict:
        try:
            ostd = float(self._outlier_std_edit.text())
        except ValueError:
            ostd = 1.7
        return {
            "outlier_std": max(0.0, ostd),
            "angle_threshold": self.slider.value(),
            "use_pe_centroid": self.checkbox.isChecked()
        }


class IvenMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IVEN")
        self.setMinimumSize(1280, 850)
        self.session = Session()
        self.output_dir = None
        self._picking_nbr_toggle = False
        self._nbr_toggle_first = None
        self._build_ui()
        self._connect_signals()
        try:
            self._open_file()
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Top toolbar — icon-only, text appears on hover via tooltips
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.setStyleSheet(
            """
            QToolBar {
                background: #e4e4e4;
                border-bottom: 1px solid #cccccc;
                spacing: 2px;
                padding: 2px 6px;
            }
            QToolButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 5px;
                padding: 6px;
                font-size: 11px;
                font-weight: 500;
                color: #1a1a1a;
            }
            QToolButton:hover {
                background: #d0d0d0;
                border: 1px solid #bbbbbb;
            }
            QToolButton:pressed {
                background: #c0c0c0;
            }
            QToolButton:checked {
                background: #3b82f6;
                color: #ffffff;
                border: 1px solid #2563eb;
            }
            QToolButton::menu-indicator { image: none; width: 0; }
        """
        )
        self.addToolBar(toolbar)

        # File section
        self.btn_open_file = QAction(QIcon(abspath("assets/openfile.png")), "", parent=self)
        self.btn_open_file.setToolTip("Open data file [Ctrl+O]")
        self.btn_open_file.setShortcut("Ctrl+O")
        toolbar.addAction(self.btn_open_file)

        self.btn_save = QAction(QIcon(abspath("assets/savefile.png")), "", self)
        self.btn_save.setToolTip("Compute all results and save [Ctrl+S]")
        self.btn_save.setShortcut("Ctrl+S")
        toolbar.addAction(self.btn_save)

        toolbar.addSeparator()

        # Hull mesh toggle
        self.btn_hull_mesh = QAction(QIcon(abspath("assets/wire_off.png")), "", self)
        self.btn_hull_mesh.setCheckable(True)
        self.btn_hull_mesh.setChecked(True)
        self.btn_hull_mesh.setToolTip("Toggle outer hull mesh [H]")
        self.btn_hull_mesh.setShortcut("H")
        toolbar.addAction(self.btn_hull_mesh)

        toolbar.addSeparator()

        # Inside / Outside section
        self.btn_auto_outside = QAction(QIcon(abspath("assets/outside_auto.png")), "", self)
        self.btn_auto_outside.setToolTip("Classify cells as inside / outside [Ctrl+I]")
        self.btn_auto_outside.setShortcut("Ctrl+I")
        toolbar.addAction(self.btn_auto_outside)

        self.btn_manual_outside = QAction(QIcon(abspath("assets/outside_manual.png")), "", self)
        self.btn_manual_outside.setCheckable(True)
        self.btn_manual_outside.setToolTip("Click on cells to flip inside / outside [I]")
        self.btn_manual_outside.setShortcut("I")
        toolbar.addAction(self.btn_manual_outside)

        toolbar.addSeparator()

        self.btn_auto_migration = QAction(QIcon(abspath("assets/outlier_auto.png")), "", self)
        #self.btn_auto_migration.setCheckable(True)
        self.btn_auto_migration.setToolTip("Auto-detect migrating status for ICM cells [Ctrl+Shift+O]")
        self.btn_auto_migration.setShortcut("Ctrl+Shift+O")
        toolbar.addAction(self.btn_auto_migration)

        self.btn_manual_migration = QAction(QIcon(abspath("assets/outlier_manual.png")), "", self)
        self.btn_manual_migration.setCheckable(True)
        self.btn_manual_migration.setToolTip("Click on ICM cells to toggle migrating status [O]")
        self.btn_manual_migration.setShortcut("O")
        toolbar.addAction(self.btn_manual_migration)

        self.btn_migration_lines = QAction(QIcon(abspath("assets/lines.png")), "", self)
        self.btn_migration_lines.setCheckable(True)
        self.btn_migration_lines.setChecked(False)
        self.btn_migration_lines.setToolTip("Show/hide migration distance lines [G]")
        self.btn_migration_lines.setShortcut("G")
        toolbar.addAction(self.btn_migration_lines)

        toolbar.addSeparator()

        # Cavity section
        self.btn_auto_cavity = QAction(QIcon(abspath("assets/cavity_auto.png")), "", self)
        self.btn_auto_cavity.setToolTip("Auto-detect cavity-adjacent cells [Ctrl+C]")
        self.btn_auto_cavity.setShortcut("Ctrl+C")
        toolbar.addAction(self.btn_auto_cavity)

        self.btn_manual_cavity = QAction(QIcon(abspath("assets/cavity_manual.png")), "", self)
        self.btn_manual_cavity.setCheckable(True)
        self.btn_manual_cavity.setToolTip("Manually toggle cavity adjacency [C]")
        self.btn_manual_cavity.setShortcut("C")
        toolbar.addAction(self.btn_manual_cavity)

        self.btn_icm_surface = QAction(QIcon(abspath("assets/vol_on.png")), "", self)
        self.btn_icm_surface.setCheckable(True)
        self.btn_icm_surface.setChecked(False)
        self.btn_icm_surface.setToolTip("Toggle ICM cavity surface mesh [M]")
        self.btn_icm_surface.setShortcut("M")
        toolbar.addAction(self.btn_icm_surface)

        toolbar.addSeparator()

        # Neighbours section
        self.btn_auto_nbr = QAction(QIcon(abspath("assets/neighbours.png")), "", self)
        self.btn_auto_nbr.setToolTip("Compute Delaunay neighbours")
        toolbar.addAction(self.btn_auto_nbr)

        self.btn_manual_nbr = QAction(QIcon(abspath("assets/pair.png")), "", self)
        self.btn_manual_nbr.setCheckable(True)
        self.btn_manual_nbr.setToolTip("Click on two cells to toggle neighbour pair")
        toolbar.addAction(self.btn_manual_nbr)

        self.btn_show_nbr_lines = QAction(QIcon(abspath("assets/lines.png")), "", self)
        self.btn_show_nbr_lines.setCheckable(True)
        self.btn_show_nbr_lines.setToolTip("Show/hide neighbour lines")
        toolbar.addAction(self.btn_show_nbr_lines)

        toolbar.addSeparator()

        self.btn_cavity_settings = QAction(QIcon(abspath("assets/alpha.png")), "", self)
        self.btn_cavity_settings.setToolTip(
            "Cavity detection settings (outlier exclusion) [Ctrl+A]"
        )
        self.btn_cavity_settings.setShortcut("Ctrl+A")
        toolbar.addAction(self.btn_cavity_settings)

        self.btn_set_threshold = QAction(QIcon(abspath("assets/threshold.png")), "", self)
        self.btn_set_threshold.setToolTip("Configure distance threshold")
        toolbar.addAction(self.btn_set_threshold)

        # Legend toggle
        self.btn_toggle_legend = QAction(QIcon(abspath("assets/legend.png")), "", self)
        self.btn_toggle_legend.setCheckable(True)
        self.btn_toggle_legend.setChecked(True)
        self.btn_toggle_legend.setToolTip("Show/hide colour legend [L]")
        self.btn_toggle_legend.setShortcut("L")
        toolbar.addAction(self.btn_toggle_legend)

        # Colour settings (opens colour picker dialog for current channel)
        self.btn_colour_settings = QAction(QIcon(abspath("assets/colour.png")), "", self)
        self.btn_colour_settings.setToolTip("Colour settings for current channel")
        toolbar.addAction(self.btn_colour_settings)

        # canvas
        self.canvas = EmbryoCanvas(self)

        # right panel
        right_panel = QWidget()
        right_panel.setMinimumWidth(290)
        right_panel.setMaximumWidth(320)
        right_panel.setStyleSheet(
            "QWidget#rightPanel {"
            "  background-color: #f6f7f9;"
            "  border-left: 1px solid #dddfe3;"
            "}"
        )
        right_panel.setObjectName("rightPanel")
        rl = QVBoxLayout(right_panel)
        rl.setAlignment(Qt.AlignmentFlag.AlignTop)
        rl.setSpacing(12)
        rl.setContentsMargins(12, 14, 12, 12)

        # Colour Mode
        colour_section = QWidget()
        colour_section.setObjectName("colourSection")
        colour_section.setStyleSheet(
            "QWidget#colourSection {"
            "  background: #ffffff;"
            "  border: 1px solid #e2e4e8;"
            "  border-radius: 8px;"
            "}"
        )
        cs_layout = QVBoxLayout(colour_section)
        cs_layout.setSpacing(6)
        cs_layout.setContentsMargins(10, 10, 10, 10)

        colour_header = QLabel("Colour Mode")
        colour_header.setStyleSheet(
            "font-size: 10px; font-weight: 700; color: #8b8fa3;"
            "letter-spacing: 0.5px; text-transform: uppercase;"
            "background: transparent; padding: 0; margin: 0;"
        )
        cs_layout.addWidget(colour_header)

        self.colour_combo = QComboBox()
        self.colour_combo.addItems(["Inside / Outside", "Cavity Adjacency", "Migration"])
        self.colour_combo.setToolTip("Choose how cells are coloured in the 3-D view")
        self.colour_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        cs_layout.addWidget(self.colour_combo)
        cs_layout.addSpacing(4)

        rl.addWidget(colour_section)

        # Cell count stats section
        stats_section = QWidget()
        stats_section.setObjectName("statsSection")
        stats_section.setStyleSheet(
            "QWidget#statsSection {"
            "  background: #ffffff;"
            "  border: 1px solid #e2e4e8;"
            "  border-radius: 8px;"
            "}"
        )
        stats_layout = QVBoxLayout(stats_section)
        stats_layout.setSpacing(4)
        stats_layout.setContentsMargins(10, 10, 10, 10)

        stats_header = QLabel("Cell Counts")
        stats_header.setStyleSheet(
            "font-size: 10px; font-weight: 700; color: #8b8fa3;"
            "letter-spacing: 0.5px; text-transform: uppercase;"
            "background: transparent; padding: 0; margin: 0;"
        )
        stats_layout.addWidget(stats_header)

        self._lbl_total = QLabel("Total: —")
        self._lbl_outside = QLabel("Outside: —")
        self._lbl_inside = QLabel("Inside: —")
        self._lbl_cavity = QLabel("Cavity-adjacent: —")
        _count_style = "font-size: 11px; color: #3a3d4a; background: transparent;"
        for lbl in (self._lbl_total, self._lbl_outside, self._lbl_inside, self._lbl_cavity):
            lbl.setStyleSheet(_count_style)
            stats_layout.addWidget(lbl)

        # Layers
        layer_section = QWidget()
        layer_section.setObjectName("layerSection")
        layer_section.setStyleSheet(
            "QWidget#layerSection {"
            "  background: #ffffff;"
            "  border: 1px solid #e2e4e8;"
            "  border-radius: 8px;"
            "}"
        )
        layer_outer = QVBoxLayout(layer_section)
        layer_outer.setSpacing(0)
        layer_outer.setContentsMargins(10, 10, 10, 8)

        layer_header = QLabel("Layers")
        layer_header.setStyleSheet(
            "font-size: 10px; font-weight: 700; color: #8b8fa3;"
            "letter-spacing: 0.5px; text-transform: uppercase;"
            "background: transparent; padding: 0; margin-bottom: 6px;"
        )
        layer_outer.addWidget(layer_header)

        categories = [
            ("Inside", "#4aa8ff", "inside"),
            ("Outside", "#ff66cc", "outside"),
            ("Cavity-adjacent", "#ffcc00", "cavity"),
            ("Not cavity-adjacent", "#9966cc", "not_cavity"),
        ]

        self.chk_show_inside = IconToggle(
            abspath("assets/eye_on.png"), abspath("assets/eye_off.png"), checked=True
        )
        self.chk_show_outside = IconToggle(
            abspath("assets/eye_on.png"), abspath("assets/eye_off.png"), checked=True
        )
        self.chk_show_cavity = IconToggle(
            abspath("assets/eye_on.png"), abspath("assets/eye_off.png"), checked=True
        )
        self.chk_show_not_cavity = IconToggle(
            abspath("assets/eye_on.png"), abspath("assets/eye_off.png"), checked=True
        )
        self.chk_wire_inside = IconToggle(
            abspath("assets/wire_on.png"), abspath("assets/wire_off.png"), checked=False
        )
        self.chk_wire_outside = IconToggle(
            abspath("assets/wire_on.png"), abspath("assets/wire_off.png"), checked=False
        )
        self.chk_wire_cavity = IconToggle(
            abspath("assets/wire_on.png"), abspath("assets/wire_off.png"), checked=False
        )
        self.chk_wire_not_cavity = IconToggle(
            abspath("assets/wire_on.png"), abspath("assets/wire_off.png"), checked=False
        )
        self.chk_vol_inside = IconToggle(
            abspath("assets/vol_on.png"), abspath("assets/vol_off.png"), checked=False
        )
        self.chk_vol_outside = IconToggle(
            abspath("assets/vol_on.png"), abspath("assets/vol_off.png"), checked=False
        )
        self.chk_vol_cavity = IconToggle(
            abspath("assets/vol_on.png"), abspath("assets/vol_off.png"), checked=False
        )
        self.chk_vol_not_cavity = IconToggle(
            abspath("assets/vol_on.png"), abspath("assets/vol_off.png"), checked=False
        )

        show_checks = [
            self.chk_show_inside,
            self.chk_show_outside,
            self.chk_show_cavity,
            self.chk_show_not_cavity,
        ]
        wire_checks = [
            self.chk_wire_inside,
            self.chk_wire_outside,
            self.chk_wire_cavity,
            self.chk_wire_not_cavity,
        ]
        vol_checks = [
            self.chk_vol_inside,
            self.chk_vol_outside,
            self.chk_vol_cavity,
            self.chk_vol_not_cavity,
        ]

        for i, (label_text, colour, key) in enumerate(categories):
            row_widget = QWidget()
            row_widget.setObjectName(f"layerRow{i}")
            bg = "#ffffff"
            row_widget.setStyleSheet(
                f"QWidget#layerRow{i} {{ background: {bg}; border-radius: 4px; }}"
            )
            row_layout = QGridLayout(row_widget)
            row_layout.setContentsMargins(4, 5, 2, 5)
            row_layout.setHorizontalSpacing(0)
            row_layout.setVerticalSpacing(0)

            # Colour swatch + name
            name_container = QHBoxLayout()
            name_container.setSpacing(6)
            name_container.setContentsMargins(0, 0, 0, 0)
            swatch = ClickableSwatch(colour, 14)
            swatch.setToolTip(f"Click to change {label_text} mesh/volume colour")
            swatch.clicked.connect(
                lambda checked=False, k=key, c=colour, sw=swatch: self._pick_layer_colour(
                    k, sw
                )
            )
            name_lbl = QLabel(label_text)
            name_lbl.setStyleSheet(
                "font-size: 11px; font-weight: 500; color: #3a3d4a;"
                f"background: {bg};"
            )
            name_container.addWidget(swatch)
            name_container.addWidget(name_lbl)
            name_container.addStretch()

            name_widget = QWidget()
            name_widget.setStyleSheet(f"background: {bg};")
            name_widget.setLayout(name_container)
            row_layout.addWidget(name_widget, 0, 0)

            # Visibility / Wireframe / Volume toggles
            for ci, chk in enumerate(
                [show_checks[i], wire_checks[i], vol_checks[i]], start=1
            ):
                tips = [
                    f"Show/hide {label_text} points",
                    f"Wireframe mesh for {label_text}",
                    f"Transparent volume for {label_text}",
                ]
                chk.setToolTip(tips[ci - 1])
                chk_wrap = QWidget()
                chk_wrap.setStyleSheet(f"background: {bg};")
                cl = QHBoxLayout(chk_wrap)
                cl.setContentsMargins(0, 0, 0, 0)
                cl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cl.addWidget(chk)
                row_layout.addWidget(chk_wrap, 0, ci)

            row_layout.setColumnStretch(0, 1)
            row_layout.setColumnMinimumWidth(1, 32)
            row_layout.setColumnMinimumWidth(2, 32)
            row_layout.setColumnMinimumWidth(3, 32)

            layer_outer.addWidget(row_widget)

        rl.addWidget(layer_section)
        rl.addWidget(stats_section)
        rl.addStretch()

        # Assemble layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 260])

        # Prevent right panel from collapsing
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(splitter)

    def _connect_signals(self):
        self.btn_auto_outside.triggered.connect(self._auto_detect_outside)
        self.btn_manual_outside.toggled.connect(self._toggle_manual_outside_mode)
        self.btn_auto_cavity.triggered.connect(self._auto_detect_cavity)
        self.btn_manual_cavity.toggled.connect(self._toggle_manual_cavity_mode)
        self.btn_cavity_settings.triggered.connect(self._show_cavity_settings_dialog)
        self.btn_icm_surface.toggled.connect(self._toggle_icm_surface)
        self.btn_migration_lines.toggled.connect(self._toggle_migration_lines)
        self.btn_auto_migration.triggered.connect(self._auto_detect_outlier)
        self.btn_manual_migration.toggled.connect(self._toggle_manual_migration_mode)
        self.btn_auto_nbr.triggered.connect(self._auto_detect_neighbours)
        self.btn_set_threshold.triggered.connect(self._set_threshold)
        self.btn_manual_nbr.toggled.connect(self._toggle_manual_nbr_mode)
        self.btn_show_nbr_lines.toggled.connect(self._toggle_nbr_lines)
        self.btn_hull_mesh.toggled.connect(self._toggle_hull_mesh)

        self.colour_combo.currentTextChanged.connect(self._on_colour_changed)

        self.chk_show_inside.toggled.connect(
            lambda v: self.canvas.set_points_hidden("inside", not v)
        )
        self.chk_show_outside.toggled.connect(
            lambda v: self.canvas.set_points_hidden("outside", not v)
        )
        self.chk_show_cavity.toggled.connect(
            lambda v: self.canvas.set_points_hidden("cavity", not v)
        )
        self.chk_show_not_cavity.toggled.connect(
            lambda v: self.canvas.set_points_hidden("not_cavity", not v)
        )

        self.chk_wire_inside.toggled.connect(
            lambda v: self.canvas.set_wireframe_visible("inside", v)
        )
        self.chk_wire_outside.toggled.connect(
            lambda v: self.canvas.set_wireframe_visible("outside", v)
        )
        self.chk_wire_cavity.toggled.connect(
            lambda v: self.canvas.set_wireframe_visible("cavity", v)
        )
        self.chk_wire_not_cavity.toggled.connect(
            lambda v: self.canvas.set_wireframe_visible("not_cavity", v)
        )

        self.chk_vol_inside.toggled.connect(
            lambda v: self.canvas.set_volume_visible("inside", v)
        )
        self.chk_vol_outside.toggled.connect(
            lambda v: self.canvas.set_volume_visible("outside", v)
        )
        self.chk_vol_cavity.toggled.connect(
            lambda v: self.canvas.set_volume_visible("cavity", v)
        )
        self.chk_vol_not_cavity.toggled.connect(
            lambda v: self.canvas.set_volume_visible("not_cavity", v)
        )

        self.btn_open_file.triggered.connect(self._open_file)
        self.btn_save.triggered.connect(self._save_results)

        self.btn_toggle_legend.toggled.connect(self._toggle_legend)
        self.btn_colour_settings.triggered.connect(self._show_channel_colour_dialog)

        self.canvas.cell_picked.connect(self._on_cell_picked)

    def show_message(self, msg: str):
        sbar = self.statusBar()
        if sbar:
            sbar.showMessage(msg)

    def keyPressEvent(self, a0):
        """Handle arrow keys to cycle through colour modes."""
        if not a0:
            return
        key = a0.key()
        if key == Qt.Key.Key_Up:
            idx = self.colour_combo.currentIndex()
            if idx > 0:
                self.colour_combo.setCurrentIndex(idx - 1)
            return
        elif key == Qt.Key.Key_Down:
            idx = self.colour_combo.currentIndex()
            if idx < self.colour_combo.count() - 1:
                self.colour_combo.setCurrentIndex(idx + 1)
            return
        super().keyPressEvent(a0)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Data File",
            "",
            "All supported (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls)",
        )
        if not path:
            self.show_message("No file selected.")
            return
        self.session = Session()
        try:
            self.session.load(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        self.setWindowTitle(f"{Path(path).name}")
        self.show_message(
            f"Loaded {self.session.num_cells} cells from {Path(path).name}"
        )

        self.colour_combo.blockSignals(True)
        self.colour_combo.clear()
        self.colour_combo.addItem("Inside / Outside")
        self.colour_combo.addItem("Cavity Adjacency")
        self.colour_combo.addItem("Migration")
        for h in self.session.headings[4:]:
            self.colour_combo.addItem(h)
        self.colour_combo.blockSignals(False)

        self.output_dir = Path(path).parent / f"IVEN2_{Path(path).stem}"
        self.show_message(str(self.output_dir))

        # ── Reset sidebar & toolbar state for fresh data ──────────
        # Uncheck all manual/toggle modes
        self.btn_manual_outside.setChecked(False)
        self.btn_manual_cavity.setChecked(False)
        self.btn_manual_nbr.setChecked(False)
        self.btn_manual_migration.setChecked(False)
        self.btn_show_nbr_lines.setChecked(False)
        self.btn_icm_surface.setChecked(False)
        self.btn_migration_lines.setChecked(False)
        self._picking_nbr_toggle = False
        self._nbr_toggle_first = None

        # Reset hull mesh toggle to on
        self.btn_hull_mesh.setChecked(True)
        self.btn_toggle_legend.setChecked(True)

        # Reset layer visibility toggles (all visible, no wireframes/volumes)
        self.chk_show_inside.setChecked(True)
        self.chk_show_outside.setChecked(True)
        self.chk_show_cavity.setChecked(True)
        self.chk_show_not_cavity.setChecked(True)
        self.chk_wire_inside.setChecked(False)
        self.chk_wire_outside.setChecked(False)
        self.chk_wire_cavity.setChecked(False)
        self.chk_wire_not_cavity.setChecked(False)
        self.chk_vol_inside.setChecked(False)
        self.chk_vol_outside.setChecked(False)
        self.chk_vol_cavity.setChecked(False)
        self.chk_vol_not_cavity.setChecked(False)

        self.canvas.load_session(self.session)

        if self.session.is_checkpoint:
            self.show_message("Checkpoint restored. Previous classifications loaded.")
            if self.session.outside_loaded:
                self.canvas.update_edge_for_outside()
            self._apply_colour_mode()

        self._update_cell_counts()

    def _auto_detect_outside(self):
        s = self.session
        if s.num_cells == 0:
            return
        s.outside_bool1, s.outside_ids1 = classify_outside(s.xyz, s.num_cells)
        s.outside_bool2 = np.copy(s.outside_bool1)
        s.outside_ids2 = np.copy(s.outside_ids1)
        s.inside_ids2 = np.where(s.outside_bool2 == 0)[0]
        s.results_df["outside_bool"] = s.outside_bool2
        s.outside_loaded = True
        self.canvas.update_edge_for_outside()
        self._apply_colour_mode()
        self.canvas.refresh_overlays()
        self.show_message(
            f"Outside: {len(s.outside_ids2)} | Inside: {len(s.inside_ids2)}"
        )
        self._update_cell_counts()

    def _toggle_manual_outside_mode(self, checked):
        if checked:
            self.btn_manual_cavity.setChecked(False)
            self.btn_manual_nbr.setChecked(False)
            self.btn_manual_migration.setChecked(False)
        self.show_message(
            "Click on cells to toggle inside / outside."
            if checked
            else "Manual outside mode off."
        )

    def _make_icm_surface(self):
        """Build a triangulated surface along cavity-adjacent ICM points.

        Steps:
        1. Gather ICM points that are cavity-adjacent.
        2. PCA → smallest component gives the plane normal.
        3. Project those 3-D points onto the plane (2-D).
        4. 2-D Delaunay triangulation.
        5. Use the same triangle indices on the original 3-D points.
        """
        s = self.session
        if s is None or s.num_cells == 0:
            return
        if len(s.outside_bool2) == 0 or len(s.cav_adj_bool) == 0:
            return

        # ICM points that are cavity-adjacent (inside AND cavity-adj)
        icm_cav_mask = (s.outside_bool2 == 0) & (s.cav_adj_bool == 1)
        icm_cav_indices = np.where(icm_cav_mask)[0]
        icm_cav_indices = [idx for idx in icm_cav_indices if idx not in s.icm_outlier_ids]

        if len(icm_cav_indices) < 3:
            return

        pts_3d = s.xyz[icm_cav_indices]
        centroid = pts_3d.mean(axis=0)
        centered = pts_3d - centroid

        # PCA via covariance eigen-decomposition
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Smallest eigenvalue index → thinnest direction (plane normal)
        order = np.argsort(eigvals)[::-1]  # descending
        # Two largest components span the plane
        u = eigvecs[:, order[0]]
        v = eigvecs[:, order[1]]

        # Project onto the plane (2-D coordinates)
        pts_2d = np.column_stack([centered @ u, centered @ v])

        # 2-D Delaunay triangulation
        try:
            tri = Delaunay(pts_2d)
        except Exception:
            return

        # Build 3-D Poly3DCollection using original (un-projected) points
        polys = [pts_3d[face] for face in tri.simplices]
        s.icm_outlier_faces = polys

    def _auto_detect_outlier(self):
        s = self.session
        auto_outliers = detect_icm_outliers(
            s.xyz, s.inside_ids2, std_threshold=s.icm_outlier_std
        )
        s.icm_outlier_ids = auto_outliers
        s.icm_outlier_bool = np.zeros(s.num_cells, dtype=np.int64)
        s.icm_outlier_bool[s.icm_outlier_ids] = 1

    def _auto_detect_cavity(self):
        s = self.session
        if len(s.inside_ids2) < 4:
            QMessageBox.warning(
                self, "Cavity", "Need >= 4 inside cells. Run outside detection first."
            )
            return

        cavity_ids = detect_cavity_adjacent_angle(
            s.xyz,
            s.inside_ids2,
            s.outside_ids2,
            outlier_ids=s.icm_outlier_ids,
            angle_threshold_deg=s.angle_threshold,
            use_pe_centroid=s.use_pe_centroid
        )

        s.cav_adj_ids = np.array(cavity_ids, dtype=np.int64)
        s.cav_adj_bool = np.zeros(s.num_cells, dtype=np.int64)
        s.cav_adj_bool[s.cav_adj_ids] = 1
        s.results_df["cavity_adj_bool"] = s.cav_adj_bool
        s.cavity_pts = [s.xyz[np.array(cavity_ids)].mean(axis=0)] if cavity_ids else []

        s.cavity_loaded = True

        self._make_icm_surface()

        # Pre-compute migration distances so migration lines are available
        if len(s.icm_outlier_ids) > 0 and len(s.icm_outlier_faces) > 0:
            s = compile_migration(s)
            self.session = s

        self._apply_colour_mode()
        self.canvas.refresh_overlays()

        # Refresh migration lines if they are currently visible
        if self.btn_migration_lines.isChecked():
            self.canvas.draw_migration_lines()

        self.show_message(f"Cavity-adjacent: {len(cavity_ids)} cells ")
        self._update_cell_counts()

    def _toggle_manual_cavity_mode(self, checked):
        if checked:
            self.btn_manual_outside.setChecked(False)
            self.btn_manual_nbr.setChecked(False)
            self.btn_manual_migration.setChecked(False)
        self.show_message(
            "Click on cells to toggle cavity adjacency."
            if checked
            else "Manual cavity mode off."
        )

    def _toggle_manual_migration_mode(self, checked):
        if checked:
            self.btn_manual_outside.setChecked(False)
            self.btn_manual_cavity.setChecked(False)
            self.btn_manual_nbr.setChecked(False)
        self.show_message(
            "Click ICM cells to toggle migration (outlier) status."
            if checked
            else "Manual migration mode off."
        )

    def _show_cavity_settings_dialog(self):
        dlg = CavitySettingsDialog(
            outlier_std=self.session.icm_outlier_std,
            current_angle=self.session.angle_threshold,
            use_pe_centroid=self.session.use_pe_centroid,
            parent=self,
        )
        if dlg.exec():
            vals = dlg.get_values()
            self.session.icm_outlier_std = vals["outlier_std"]

            angle_threshold = vals["angle_threshold"]
            try:
                self.session.angle_threshold = int(angle_threshold)
            except ValueError:
                self.session.angle_threshold = 30

            self.session.use_pe_centroid = vals["use_pe_centroid"]

            self.show_message(rf"ICM Migration σ={vals['outlier_std']:.1f}")
            #self._auto_detect_cavity()

    def _auto_detect_neighbours(self):
        s = self.session
        if s.num_cells == 0:
            return
        if len(s.outside_bool2) == 0:
            QMessageBox.warning(self, "Neighbours", "Run outside detection first.")
            return
        s.nbr_matrix1 = nbr_matrix(s)
        if not s.thresh_method:
            s.thresh_method = "Automatic (cell position dependent)"
            s.thresh_k = 0.5
        s = eval_threshold(s)
        s = check_nbrs(s)
        s = compile_results(s)
        self.session = s
        self.show_message("Neighbours computed.")

    def _set_threshold(self):
        dlg = ThresholdDialog(self)
        if dlg.exec():
            method, val = dlg.get_values()
            self.session.thresh_method = method
            if method in ("None"):
                self.session.thresh_k = 0.5
                self.show_message(f"Threshold: {method}")
            else:
                try:
                    self.session.thresh_k = float(val)
                    self.show_message(f"Threshold: {method}, value={val}")
                except ValueError:
                    self.session.thresh_method = "Automatic (cell position dependent)"
                    self.session.thresh_k = 0.5
                    self.show_message(f"Threshold: {method}, k=0.5")

    def _toggle_manual_nbr_mode(self, checked):
        if checked:
            self.btn_manual_outside.setChecked(False)
            self.btn_manual_cavity.setChecked(False)
            self.btn_manual_migration.setChecked(False)
            self._picking_nbr_toggle = True
            self._nbr_toggle_first = None
            self.show_message("Click first cell of neighbour pair.")
        else:
            self._picking_nbr_toggle = False
            self._nbr_toggle_first = None
            self.show_message("Manual neighbour mode off.")

    def _toggle_nbr_lines(self, checked):
        if checked:
            if len(self.session.nbr_matrix2) == 0:
                self.show_message("Run neighbour detection first.")
                self.btn_show_nbr_lines.setChecked(False)
                return
            self.canvas.draw_neighbour_lines()
        else:
            self.canvas.clear_neighbour_lines()

    def _toggle_hull_mesh(self, checked):
        self.canvas.set_outer_hull_visible(checked)
        self.show_message("Hull mesh: on" if checked else "Hull mesh: off")

    def _toggle_icm_surface(self, checked):
        self.canvas.set_icm_surface_visible(checked)
        self.show_message(
            "ICM cavity surface: on" if checked else "ICM cavity surface: off"
        )

    def _toggle_migration_lines(self, checked):
        s = self.session
        if checked:
            if len(s.icm_outlier_ids) == 0 or len(s.icm_outlier_faces) == 0:
                self.show_message("Run cavity detection first.")
                self.btn_migration_lines.setChecked(False)
                return
            # Compute migration distances (and line endpoints) if not yet done
            if not hasattr(s, 'migration_lines') or len(s.migration_lines) == 0:
                s = compile_migration(s)
                self.session = s
            self.canvas.draw_migration_lines()
            self.show_message("Migration lines: on")
        else:
            self.canvas.clear_migration_lines()
            self.show_message("Migration lines: off")

    def _on_cell_picked(self, idx):
        s = self.session
        cell_id = s.ids.iloc[idx] if len(s.ids) > idx else idx

        if self.btn_manual_outside.isChecked():
            if len(s.outside_bool2) == 0:
                s.outside_bool2 = np.zeros(s.num_cells)
                s.outside_loaded = True
            s.outside_bool2[idx] = 1 - s.outside_bool2[idx]
            s.outside_ids2 = np.where(s.outside_bool2 == 1)[0]
            s.inside_ids2 = np.where(s.outside_bool2 == 0)[0]
            s.results_df["outside_bool"] = s.outside_bool2
            self.canvas.update_edge_for_outside()
            self._apply_colour_mode()
            self.canvas.refresh_overlays()
            label = "outside" if s.outside_bool2[idx] == 1 else "inside"
            self.show_message(f"Cell {cell_id} -> {label}")
            self._update_cell_counts()
            return

        if self.btn_manual_cavity.isChecked():
            if len(s.cav_adj_bool) == 0:
                s.cav_adj_bool = np.zeros(s.num_cells)
                s.cavity_loaded = True
            s.cav_adj_bool[idx] = 1 - s.cav_adj_bool[idx]
            s.cav_adj_ids = np.where(s.cav_adj_bool == 1)[0]
            s.results_df["cavity_adj_bool"] = s.cav_adj_bool
            s.cavity_pts = (
                [s.xyz[s.cav_adj_ids].mean(axis=0)] if len(s.cav_adj_ids) > 0 else []
            )
            self._make_icm_surface()
            self._apply_colour_mode()
            self.canvas.refresh_overlays()
            label = "cavity-adj" if s.cav_adj_bool[idx] == 1 else "not cavity-adj"
            self.show_message(f"Cell {cell_id} -> {label}")
            self._update_cell_counts()
            return

        if self.btn_manual_migration.isChecked():
            # Only allow toggling on ICM (inside) cells
            if len(s.outside_bool2) > 0 and s.outside_bool2[idx] == 1:
                self.show_message(f"Cell {cell_id} is an outside cell — only ICM cells can be toggled as migration.")
                return
            # Initialise outlier arrays if needed
            if len(s.icm_outlier_bool) == 0:
                s.icm_outlier_bool = np.zeros(s.num_cells, dtype=np.int64)
            # Toggle
            if idx in s.icm_outlier_ids:
                s.icm_outlier_ids.remove(idx)
            else:
                s.icm_outlier_ids.append(idx)
            s.icm_outlier_ids.sort()
            s.icm_outlier_bool = np.zeros(s.num_cells, dtype=np.int64)
            if len(s.icm_outlier_ids) > 0:
                s.icm_outlier_bool[s.icm_outlier_ids] = 1
            
            self._apply_colour_mode()
            self.canvas.refresh_overlays()
            if self.btn_migration_lines.isChecked():
                self.canvas.clear_migration_lines()
                self.canvas.draw_migration_lines()
            is_outlier = s.icm_outlier_bool[idx] == 1
            label = "Migration (outlier)" if is_outlier else "ICM (not migrating)"
            self.show_message(f"Cell {cell_id} -> {label}")
            return

        if self.btn_manual_nbr.isChecked() and self._picking_nbr_toggle:
            if self._nbr_toggle_first is None:
                self._nbr_toggle_first = idx
                self.show_message(f"First cell: {cell_id}. Now click the second cell.")
            else:
                i, j = self._nbr_toggle_first, idx
                self._nbr_toggle_first = None
                if len(s.nbr_matrix2) == 0:
                    self.show_message("Run neighbour detection first.")
                    return
                s.nbr_matrix2[i, j] = 1 - s.nbr_matrix2[i, j]
                s.nbr_matrix2[j, i] = s.nbr_matrix2[i, j]
                status = "neighbours" if s.nbr_matrix2[i, j] == 1 else "not neighbours"
                id_i = s.ids.iloc[i] if len(s.ids) > i else i
                id_j = s.ids.iloc[j] if len(s.ids) > j else j
                self.show_message(
                    f"Cells {id_i} & {id_j} -> {status}. Click next pair."
                )
                if self.btn_show_nbr_lines.isChecked():
                    self.canvas.draw_neighbour_lines()
            return

        self.show_message(f"Cell {cell_id} (index {idx})")

    def _on_colour_changed(self, text):
        is_channel = text not in ("Inside / Outside", "Cavity Adjacency", "Migration")
        if is_channel and text not in self.canvas._channel_settings:
            # Auto-determine mode for this channel
            vals = self.session.df[text]
            mode = self._auto_detect_mode(vals)
            settings = {"mode": mode}
            # Pre-populate categorical colours if categorical
            if mode == "categorical":
                try:
                    unique_vals = sorted(vals.dropna().unique(), key=lambda x: float(x))
                except (ValueError, TypeError):
                    unique_vals = sorted(vals.dropna().unique(), key=str)
                palette = ChannelColourDialog._default_palette(len(unique_vals))
                settings["categories"] = {  # type: ignore
                    str(v): palette[i % len(palette)] for i, v in enumerate(unique_vals)
                }
            self.canvas._channel_settings[text] = settings
        self._apply_colour_mode()

    @staticmethod
    def _auto_detect_mode(series: pd.Series) -> str:
        """Guess whether a column should be coloured continuously or categorically."""
        try:
            numeric = pd.to_numeric(series.dropna(), errors="raise")
        except (ValueError, TypeError):
            return "categorical"
        n_unique = numeric.nunique()
        n_total = len(numeric)
        if n_unique <= 10 and all(numeric.dropna() == numeric.dropna().astype(int)):
            return "categorical"
        if n_unique / max(n_total, 1) < 0.05 and n_unique <= 20:
            return "categorical"
        return "continuous"

    def _apply_colour_mode(self):
        text = self.colour_combo.currentText()
        if text == "Inside / Outside":
            self.canvas.colour_by_outside()
        elif text == "Cavity Adjacency":
            self.canvas.colour_by_cavity()
        elif text == "Migration":
            self.canvas.colour_by_migration()
        else:
            self.canvas.colour_by_channel(text)
        self.canvas.update_edge_for_outside()

    def _toggle_legend(self, checked):
        self.canvas.set_legend_visible(checked)
        self.show_message("Legend: on" if checked else "Legend: off")

    def _update_cell_counts(self):
        """Refresh the cell count labels in the stats section."""
        s = self.session
        total = s.num_cells if s else 0
        self._lbl_total.setText(f"Total: {total}")
        if total == 0:
            self._lbl_outside.setText("Outside: 0")
            self._lbl_inside.setText("Inside: 0")
            self._lbl_cavity.setText("Cavity-adjacent: 0")
            return
        if len(s.outside_bool2) > 0:
            n_out = int(np.sum(s.outside_bool2 == 1))
            n_in = int(np.sum(s.outside_bool2 == 0))
            self._lbl_outside.setText(f"Outside: {n_out}")
            self._lbl_inside.setText(f"Inside: {n_in}")
        else:
            self._lbl_outside.setText("Outside: 0")
            self._lbl_inside.setText("Inside: 0")
        if len(s.cav_adj_bool) > 0:
            n_cav = int(np.sum(s.cav_adj_bool == 1))
            self._lbl_cavity.setText(f"Cavity-adjacent: {n_cav}")
        else:
            self._lbl_cavity.setText("Cavity-adjacent: 0")

    def _pick_layer_colour(self, category_key, swatch):
        """Open a colour picker for the mesh/volume colour of a layer category."""
        current_hex = swatch.get_colour()
        c = QColorDialog.getColor(
            QColor(current_hex), self, f"Colour for {category_key}"
        )
        if c.isValid():
            swatch.set_colour(c.name())
            self.canvas.set_wire_colour(category_key, c.name())
            self.canvas.set_vol_colour(category_key, c.name())

    def _show_channel_colour_dialog(self):
        """Open the channel colour settings dialog."""
        col_name = self.colour_combo.currentText()

        if col_name == "Inside / Outside":
            self._show_fixed_colour_dialog(
                "Inside / Outside",
                ["outside", "inside"],
                self.canvas._outside_colours,
                "__outside__",
            )
            return
        if col_name == "Cavity Adjacency":
            self._show_fixed_colour_dialog(
                "Cavity Adjacency",
                ["cavity", "inside", "outside"],
                self.canvas._cavity_colours,
                "__cavity__",
            )
            return
        if col_name == "Migration":
            self._show_fixed_colour_dialog(
                "Migration",
                ["migration", "inside", "outside"],
                self.canvas._migration_colours,
                "__migration__",
            )
            return

        # Determine unique values for categorical mode
        vals = self.session.df[col_name]
        try:
            unique_vals = sorted(vals.dropna().unique(), key=lambda x: float(x))
        except (ValueError, TypeError):
            unique_vals = sorted(vals.dropna().unique(), key=str)

        settings = self.canvas._channel_settings.get(col_name, {})
        current_mode = settings.get("mode", "continuous")
        gradient = settings.get("gradient", ("#ff66cc", "#ffcc00"))
        cat_colours = settings.get("categories", {})

        dlg = ChannelColourDialog(
            column_name=col_name,
            unique_values=unique_vals,
            current_mode=current_mode,
            gradient_colours=gradient,
            category_colours=cat_colours,
            edge_outside_colour=self.canvas._edge_outside_colour,
            edge_inside_colour=self.canvas._edge_inside_colour,
            bg_colour=self.canvas._bg_colour,
            force_categorical=False,
            parent=self,
        )
        if dlg.exec():
            new_settings = {
                "mode": dlg.get_mode(),
                "gradient": dlg.get_gradient_colours(),
                "categories": dlg.get_category_colours(),
            }
            self.canvas._edge_outside_colour = dlg.edge_outside_colour
            self.canvas._edge_inside_colour = dlg.edge_inside_colour
            self.canvas.set_background_colour(dlg.bg_colour)
            self.canvas.update_edge_for_outside()

            self.canvas._channel_settings[col_name] = new_settings
            self._apply_colour_mode()

    def _show_fixed_colour_dialog(self, title, labels_keys, colour_dict, mode_key):
        """
        "Inside / Outside",
        [("Outside", "outside"), ("Inside", "inside")],
        self.canvas._outside_colours,
        "__outside__",
        """
        dlg = ChannelColourDialog(
            column_name=title,
            unique_values=labels_keys,
            current_mode="categorical",
            gradient_colours=(),
            category_colours=colour_dict,
            edge_outside_colour=self.canvas._edge_outside_colour,
            edge_inside_colour=self.canvas._edge_inside_colour,
            bg_colour=self.canvas._bg_colour,
            force_categorical=True,
            parent=self,
        )
        if dlg.exec():
            self.canvas._edge_outside_colour = dlg.edge_outside_colour
            self.canvas._edge_inside_colour = dlg.edge_inside_colour
            self.canvas.set_background_colour(dlg.bg_colour)
            self.canvas.update_edge_for_outside()

            category_colours = dlg.get_category_colours()
            for k in labels_keys:
                colour_dict[k] = category_colours[k]

            self._apply_colour_mode()

    def _show_fixed_colour_dialog_v1(self, title, labels_keys, colour_dict, mode_key):
        """Open a simple colour picker for fixed-mode (inside / outside or cavity) colours."""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Colour Settings: {title}")
        dlg.setMinimumWidth(320)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        hdr = QLabel(f"Colour settings for '{title}':")
        hdr.setStyleSheet("font-weight: 600; font-size: 12px; color: #1a1a1a;")
        layout.addWidget(hdr)

        swatches = {}
        for label, key in labels_keys:
            row = QHBoxLayout()
            row.setSpacing(8)
            sw = ClickableSwatch(colour_dict[key], 16)
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 11px; color: #333;")

            def make_picker(k, s):
                def pick():
                    c = QColorDialog.getColor(
                        QColor(colour_dict[k]), dlg, f"Colour for '{label}'"
                    )
                    if c.isValid():
                        colour_dict[k] = c.name()
                        s.set_colour(c.name())

                return pick

            sw.clicked.connect(make_picker(key, sw))
            swatches[key] = sw
            row.addWidget(sw)
            row.addWidget(lbl)
            row.addStretch()
            layout.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec():
            # colour_dict was mutated in place via the closures; just reapply
            self._apply_colour_mode()

    def _save_results(self):
        s = self.session
        if s.num_cells == 0:
            QMessageBox.warning(self, "Save", "No data loaded.")
            return
        if "outside_bool" not in s.results_df.columns:
            QMessageBox.warning(self, "Save", "Run inside / outside detection first.")
            return

        # Always prompt for save location
        default_dir = str(self.output_dir) if self.output_dir else ""
        d = QFileDialog.getExistingDirectory(self, "Choose output folder", default_dir)
        if not d:
            return
        self.output_dir = Path(d)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fname = self.output_dir / f"output_{self.output_dir.stem}.xlsx"

        if os.path.exists(fname):
            QMessageBox.critical(self, "Save Error", "Output file already exists. Skipping save.")
            return


        if len(s.nbr_matrix2) == 0:
            s.nbr_matrix1 = nbr_matrix(s)
            if not s.thresh_method:
                s.thresh_method = "Automatic (cell position dependent)"
                s.thresh_k = 0.5
            s = eval_threshold(s)
            s = check_nbrs(s)

        s = compile_results(s)
        s = assign_cell_lineage(s)
        s = compile_distances(s)
        s = compile_migration(s)

        cav_icm_ids = [id for id in s.inside_ids2 if (id in s.cav_adj_ids) and (id not in s.icm_outlier_ids)]
        cav_icm_nbr_distances = get_neighbour_distances(s.xyz, cav_icm_ids)
        s.mean_cav_icm_distance = float(np.mean(cav_icm_nbr_distances))

        te_ids = s.outside_ids2
        te_nbr_distances = get_neighbour_distances(s.xyz, te_ids)
        s.mean_te_distance = float(np.mean(te_nbr_distances))
        s.mean_cav_icm_distance_norm = s.mean_cav_icm_distance / s.mean_te_distance

        hull = ConvexHull(s.xyz)
        s.info = pd.DataFrame(
            {
                "Volume": [hull.volume],
                "Mean Distance Cavity-adjacent ICM": s.mean_cav_icm_distance,
                "Mean Distance Cavity-adjacent ICM (norm)": s.mean_cav_icm_distance_norm,
                "Mean Distance TE": s.mean_te_distance,
            }
        )

        for col in s.headings[4:-1]:
            id_to_val = dict(zip(s.results_df["ID"], s.results_df[col]))
            s.dist[f"{col}_cell1"] = s.dist["cell_id1"].map(id_to_val)
            s.dist[f"{col}_cell2"] = s.dist["cell_id2"].map(id_to_val)

        self.session = s

        s.save_checkpoint(fname)

        img_path = self.output_dir / f"figure.png"
        self.canvas.fig.savefig(str(img_path), dpi=300)

        self.show_message(f"Saved to {fname}")
        QMessageBox.information(
            self, "Saved", f"Results saved to:\n{fname}\n\nImage saved to:\n{img_path}"
        )


def main():
    style_file = abspath('assets/style.txt')
    with open(style_file, "r") as f:
        stylesheet = f.read()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Arial", 10))
    app.setStyleSheet(stylesheet)
    window = IvenMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()