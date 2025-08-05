# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:51:10 2024

@author: n.nyberg
"""

%matplotlib qt

import sys
sys.path.append('F:\\backups\\from_office_computer\\Post_doc\\NeuroDataReHack_2024') 
# import pickle as pkl
import dill as pkl
from parti_class import ParTI
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QListWidget, QCheckBox, QButtonGroup, QRadioButton, QMessageBox, QSpinBox
)
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from scipy.linalg import det
sys.path.append('F:\\backups\\from_office_computer\\Post_doc\\lib-unmixing-master') 
from unmixing import sisal#, mvsa
from mvsa import mvsa
import math
# from matplotlib import cm
from matplotlib import colormaps

from pyhull.delaunay import DelaunayTri
from PyQt5.QtWidgets import QSlider
from PyQt5.QtGui import QFont
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox

import sys
import numpy as np
import pandas as pd
import dill as pkl
from PyQt5.QtWidgets import (
    QGroupBox, QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QListWidget, QButtonGroup, QRadioButton, QMessageBox, QComboBox, QHBoxLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigsh
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import norm
from numpy import arccos, dot
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtWidgets import QStyle, QStyleOptionSlider
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QGridLayout
# from PyQt5.QtWidgets import QSlider
from PyQt5.QtGui import QPainter
from scipy.interpolate import splprep, splev
import matplotlib
import colorsys
from rastermap import Rastermap
#%% Functions (TEMP)
# TODO: put somewhere else, either in a class or read into this script from other script
# TODO: fix why its so slow now?

def findMinSimplex(numIter,DataPCA,algNum,NArchetypes,silent=0):
    
    minArchsIter = [None] * (3 * numIter)
    VolArch = np.zeros(3 * numIter)
    DataDim = DataPCA.shape[1]
    
    if algNum == 1:
        # if verbose:
        #     print('Calculating archetypes positions with SISAL (Bioucas-Dias JM, 2009)\n')
        for i in range(0, 3 * numIter):
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            
            
            # This is where: ResourceWarning: unclosed file <_io.BufferedWriter name=5>" happens
            Archs, _, _, _ = sisal(DataPCA[:, :NArchetypes].T, NArchetypes, verbose=0) # Needs at least NArchetypes = 4 to give 3 dimensions... Look up why you cant get a film (2D) and plot in 3D - should be possible
            # Archs, _, _, _ = mvsa(DataPCA[:, :NArchetypes].T, NArchetypes, verbose=0) # Needs at least NArchetypes = 4 to give 3 dimensions... Look up why you cant get a film (2D) and plot in 3D - should be possible

            
            if not np.all(np.isnan(Archs)):
                # calculate the volume of the simplex
                Arch1Red = Archs - Archs[:, NArchetypes-1][:, np.newaxis] # NILS TODO: probably dont need minus on here? Check
                VolArch[i] = np.abs(det(Arch1Red[:-1, :-1])) / math.factorial(NArchetypes - 1)
                #save the archetypes
                minArchsIter[i] = Archs[:-1, :NArchetypes]
            else:
                VolArch[i]= np.nan
                minArchsIter[i] = np.nan
        
        IndminVol = np.argmin(VolArch) # find the minimal volume simplex
        VolArchReal = VolArch[IndminVol]
        ArchsMin = minArchsIter[IndminVol] # get the minimal archetypes that form this simplex
        
    # TODO: add other algorithms here, MVSA, etc

    return ArchsMin, VolArchReal

def n_findr(data, num_endmembers, max_iter=100, tol=1e-6):
    """
    Perform the N-FINDR algorithm to find endmembers.
    
    Parameters:
        data (np.ndarray): Input data matrix (n_samples x n_features).
        num_endmembers (int): Number of endmembers (vertices of the simplex).
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        endmembers (np.ndarray): Indices of the selected endmembers.
    """
    n_samples, n_features = data.shape
    if num_endmembers > n_samples:
        raise ValueError("Number of endmembers cannot exceed the number of samples.")
    
    # Step 1: Initialize endmember indices randomly
    endmember_indices = np.random.choice(n_samples, size=num_endmembers, replace=False)

    def calculate_simplex_volume(points):
        """Calculate the volume of the simplex defined by points."""
        ref_point = points[0]
        matrix = points[1:] - ref_point
        volume = np.abs(np.linalg.det(matrix)) / np.math.factorial(len(points) - 1)
        return volume

    # Step 2: Iterate to maximize simplex volume
    prev_volume = 0
    for iteration in range(max_iter):
        endmember_points = data[endmember_indices]
        current_volume = calculate_simplex_volume(endmember_points)
        
        if np.abs(current_volume - prev_volume) < tol:
            break
        prev_volume = current_volume
        
        # Test replacing each endmember with all other points
        for i in range(num_endmembers):
            for j in range(n_samples):
                temp_indices = endmember_indices.copy()
                temp_indices[i] = j
                temp_points = data[temp_indices]
                temp_volume = calculate_simplex_volume(temp_points)
                if temp_volume > current_volume:
                    endmember_indices[i] = j
                    current_volume = temp_volume
    
    return endmember_indices

#%% TEMP 13/01/25

# TODO: fix so you can select archetype when attribute x bin
# TODO: add rastermap plot
# TODO: add aROC plots
# TODO: have default: #D, 2D, heatmap (arch x bin), heatmap (attr x arch), and same for bottom plots, then have the ability to adjust plots as you like with buttons adjusted somehow accordingly
# Order to implement stuff into the GUI: 
    # 1. Rastermap (with different options for organising data) (highlighted points shown in colormap)
    # 2. Spike count plots (histogram plots summary)
    # 3. aROC and spike plots per neuron (with able to adjust archetype nr etc)
    # 3. aROC heatmaps (with different options for organising data) and cumulative line plots
class LabeledSlider(QtWidgets.QWidget):
    def __init__(self, minimum, maximum, interval=1, parent=None):
        super().__init__(parent)

        levels = range(minimum, maximum + interval, interval)
        self.levels = list(zip(levels, map(str, levels)))

        self.left_margin = 10
        self.top_margin = 10
        self.right_margin = 10
        self.bottom_margin = 20  # Adjust bottom margin to fit labels

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(self.left_margin, self.top_margin, self.right_margin, self.bottom_margin)

        self.sl = QtWidgets.QSlider(Qt.Horizontal, self)
        self.sl.setMinimum(minimum)
        self.sl.setMaximum(maximum)
        self.sl.setValue(minimum)
        self.sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sl.setTickInterval(interval)
        self.sl.setSingleStep(1)
        self.layout.addWidget(self.sl)

    def paintEvent(self, e):
        super().paintEvent(e)

        style = self.sl.style()
        painter = QPainter(self)
        st_slider = QStyleOptionSlider()
        st_slider.initFrom(self.sl)
        st_slider.orientation = self.sl.orientation()

        length = style.pixelMetric(QStyle.PM_SliderLength, st_slider, self.sl)
        available = style.pixelMetric(QStyle.PM_SliderSpaceAvailable, st_slider, self.sl)

        for v, v_str in self.levels:
            rect = painter.drawText(QRect(), Qt.TextDontPrint, v_str)

            x_loc = QStyle.sliderPositionFromValue(self.sl.minimum(),
                                                   self.sl.maximum(),
                                                   v,
                                                   available) + length // 2

            left = x_loc - rect.width() // 2 + self.left_margin
            bottom = self.rect().bottom() - 10  # Adjust position of labels

            pos = QPoint(left, bottom)
            painter.drawText(pos, v_str)

class SliderWithText(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paintEvent(self, event):
        # Call the base class's paintEvent to render the slider
        super().paintEvent(event)
        
        # Create a QPainter object to draw the text
        painter = QPainter(self)

        # Set a readable font size
        painter.setFont(QFont("Arial", 10))

        # Create and initialize the style option
        option = QStyleOptionSlider()
        self.initStyleOption(option)

        # Get the position of the slider handle
        handle_rect = self.style().subControlRect(
            self.style().CC_Slider, option, self.style().SC_SliderHandle, self
        )
        handle_center = handle_rect.center()

        # Calculate percentage text
        percentage = self.value()
        text = f"{percentage}%"

        # Draw the text centered above the slider handle
        text_rect = painter.boundingRect(
            handle_center.x() - 20, handle_center.y() - 30, 40, 20, Qt.AlignCenter, text
        )
        painter.drawText(text_rect, Qt.AlignCenter, text)

        # End painting
        painter.end()
        
class SharedState(QObject):
    """Shared state for communicating between windows."""
    updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.selected_points = []

    def set_selected_points(self, points):
        """Update the selected points and notify all listeners."""
        self.selected_points = points
        self.updated.emit()


class SelectFromCollectionLasso:
    def __init__(self, ax, scatter, on_select_callback, is_3d=False):
        self.ax = ax
        self.scatter = scatter
        self.is_3d = is_3d
        self.on_select_callback = on_select_callback
        self.ind = set()
        self._initialize_data()
        self.lasso = None
        self.selector_enabled = False

    def _initialize_data(self):
        """Initialize scatter plot data."""
        if self.is_3d:
            # For 3D scatter, _offsets3d is a (3, N) tuple
            self.data = np.array(self.scatter._offsets3d).T
        else:
            # For 2D scatter, get_offsets() is shape (N, 2)
            self.data = self.scatter.get_offsets()

    def toggle_selection_mode(self):
        """Enable or disable lasso selection."""
        self.selector_enabled = not self.selector_enabled
        if self.selector_enabled:
            if self.is_3d:
                self.ax.disable_mouse_rotation()
            self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        else:
            if self.is_3d:
                self.ax.mouse_init()
            if self.lasso:
                self.lasso.disconnect_events()
                self.lasso = None

    def on_select(self, verts):
        """Handle lasso selection."""
        path = Path(verts)
        if self.is_3d:
            projected = self.project_3d_to_2d(self.data)
        else:
            projected = self.data
        selected_indices = np.nonzero(path.contains_points(projected))[0]
        self.ind = set(selected_indices)
        self.on_select_callback(list(self.ind))

    def project_3d_to_2d(self, points):
        """Project 3D points to 2D for selection logic."""
        trans = self.ax.get_proj()
        points_4d = np.column_stack((points, np.ones(points.shape[0])))
        projected = np.dot(points_4d, trans.T)
        projected[:, :2] /= projected[:, 3, None]
        return projected[:, :2]


class AnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.session = None
        self.viewers = []
        self.shared_state = SharedState()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Session Analysis GUI')

        window_width = 300  # Desired width
        window_height = 600  # Desired height
        self.resize(window_width, window_height)  # Adjust the size of the window
        
        # Center the window on the screen
        self.show()  # Ensure the window is initialized before measuring geometry
        frame_geometry = self.frameGeometry()  # Includes window decorations
        screen_geometry = QApplication.desktop().screenGeometry()
        
        # Calculate center
        x = (screen_geometry.width() - frame_geometry.width()) // 2 - (400 + 150 + 25)
        y = (screen_geometry.height() - frame_geometry.height()) // 2
        self.move(x, y)
        
        self.layout = QVBoxLayout()

        self.load_file_button = QPushButton('Load .pkl File', self)
        self.load_file_button.clicked.connect(self.load_pkl_file)
        self.layout.addWidget(self.load_file_button)

        self.file_label = QLabel('Selected File: None', self)
        self.layout.addWidget(self.file_label)

        self.region_label = QLabel('Select Brain Regions:')
        self.layout.addWidget(self.region_label)
        self.region_list = QListWidget(self)
        self.region_list.setSelectionMode(QListWidget.MultiSelection)
        self.layout.addWidget(self.region_list)

        self.embedding_label = QLabel('Select Embedding Method:')
        self.layout.addWidget(self.embedding_label)
        self.radio_group = QButtonGroup(self)

        self.pca_radio = QRadioButton('PCA', self)
        self.radio_group.addButton(self.pca_radio)
        self.layout.addWidget(self.pca_radio)

        self.fa_radio = QRadioButton('FA', self)
        self.radio_group.addButton(self.fa_radio)
        self.layout.addWidget(self.fa_radio)

        self.lem_radio = QRadioButton('Laplacian', self)
        self.radio_group.addButton(self.lem_radio)
        self.layout.addWidget(self.lem_radio)

        # Archetype Method
        self.archetype_label = QLabel('Select Archetype Method:')
        self.layout.addWidget(self.archetype_label)
        self.archetype_dropdown = QComboBox(self)
        self.archetype_dropdown.addItems(['None', 'ConvexHull', 'nfindr', 'SISAL', 'alpha_shape'])
        self.layout.addWidget(self.archetype_dropdown)
        
        self.analyse_button = QPushButton('Proceed with Analysis', self)
        self.analyse_button.clicked.connect(self.proceed_with_analysis)
        self.layout.addWidget(self.analyse_button)
    
        self.setLayout(self.layout)

    def load_pkl_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select .pkl File', '', 'Pickle Files (*.pkl)')
        if file_path:
            self.file_label.setText(f'Selected File: {file_path}')
            try:
                with open(file_path, 'rb') as f:
                    self.session = pkl.load(f)

                if hasattr(self.session, 'unit_channels') and 'structure_acronym' in self.session.unit_channels:
                    brain_regions = sorted(self.session.unit_channels['structure_acronym'].unique())
                    self.region_list.clear()
                    self.region_list.addItems(brain_regions)
                else:
                    QMessageBox.warning(self, 'Error', 'No brain region information found.')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load file: {e}')

    def proceed_with_analysis(self):
        if not self.session:
            QMessageBox.warning(self, 'Error', 'No session loaded.')
            return

        selected_regions = [item.text() for item in self.region_list.selectedItems()]
        if not selected_regions:
            QMessageBox.warning(self, 'Error', 'No brain regions selected.')
            return

        selected_method = None
        if self.pca_radio.isChecked():
            selected_method = 'PCA'
        elif self.fa_radio.isChecked():
            selected_method = 'FA'
        elif self.lem_radio.isChecked():
            selected_method = 'Laplacian'

        if not selected_method:
            QMessageBox.warning(self, 'Error', 'No embedding method selected.')
            return

        neuron_ids = self.session.unit_channels[self.session.unit_channels['structure_acronym'].isin(selected_regions)].index
        self.session.neuron_x_time_matrix_subset = self.session.neuron_x_time_matrix.loc[neuron_ids]

        smoothed = gaussian_filter1d(self.session.neuron_x_time_matrix_subset.values, sigma=3.5, axis=1)
        z_scored_data = StandardScaler().fit_transform(smoothed.T).T

        embeddings = None
        if selected_method == 'PCA':
            embeddings = PCA().fit_transform(z_scored_data.T)
        elif selected_method == 'FA':
            embeddings = FactorAnalysis().fit_transform(z_scored_data.T)
        elif selected_method == 'Laplacian':
            n_neighbors = max(1, int(z_scored_data.shape[1] * 0.005))
            adjacency_matrix = kneighbors_graph(z_scored_data.T, n_neighbors=n_neighbors,
                                                mode='connectivity', include_self=False).toarray()
            degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
            laplacian_rw = np.eye(adjacency_matrix.shape[0]) - np.linalg.inv(degree_matrix) @ adjacency_matrix
            _, eigenvectors = eigsh(laplacian_rw, k=4, which='SM')
            embeddings = eigenvectors[:, 1:]

        session_id = "Session_01"
        brain_regions = selected_regions
        algorithm = selected_method

        # We assume self.session.discrete_attributes includes all attributes
        # attrs = self.session.discrete_attributes
        attrs = self.session.continuous_attributes
        
        self.archetype_method = self.archetype_dropdown.currentText()
        viewer = EmbeddingWindow(embeddings, self.session,
                                 session_id, brain_regions, algorithm,
                                 self.shared_state, self.archetype_method)
        self.viewers.append(viewer)
        viewer.show()

class EmbeddingWindow(QWidget):
    def __init__(
        self,
        embeddings,
        session,
        session_id,
        brain_regions,
        algorithm,
        shared_state,
        archetype_method,
        alpha_other=0.3
    ):
        super().__init__()
        
        self.session = session
        self.embeddings = embeddings
        self._scaled_embeddings = embeddings.copy()
        self.attributes = self.session.continuous_attributes  # DataFrame (n_points, n_attributes)
        self.session_id = session_id
        self.brain_regions = brain_regions
        self.algorithm = algorithm
        self.shared_state = shared_state
        self.archetype_method = archetype_method
        self.alpha_other = alpha_other
        self.selector = None
        self.lasso_active = False
        self.highlighted_indices = []
        self.selected_points = []
        
        # Caches for convex hull and SISAL (top usage)
        self.cached_convex_hull = None
        self.cached_sisal_hull = None

        # If embeddings are very small, scale them
        self._scale_embedding = False
        self._scale_factor = 100
        if np.max(self.embeddings) < 1:
            self._scaled_embeddings *= self._scale_factor
            self._scale_embedding = True

        # -- For the TOP logic --
        self.closest_indices_per_archetype = None    # used by top
        self.means_matrix = None                     # shape = (n_attrs, n_archetypes) [top usage]
        self.bin_matrix = None                       # shape = (n_attrs, n_bins) [top usage]
        self.last_valid_attribute = None             # for the top

        # -- For the BOTTOM-LEFT logic --
        self.last_valid_attribute_left = None
        self.closest_indices_per_archetype_left = None
        self.means_matrix_left = None
        self.bin_matrix_left = None

        # -- For the BOTTOM-RIGHT logic --
        self.last_valid_attribute_right = None
        self.closest_indices_per_archetype_right = None
        self.means_matrix_right = None
        self.bin_matrix_right = None

        # -- For the Rastermap --
        self.rastermap_archetype = None
        self.rastermap_percentage = 5  # default 5%
        self.rastermap_selected_archetype = None
        self.rastermap_slider_value = 5

        # Connect to shared state for multi-window sync
        self.shared_state.updated.connect(self.sync_highlighted_points)

        self.initUI()

    # -------------------------------------------------------------------------
    #  INIT UI
    # -------------------------------------------------------------------------
    def initUI(self):
        self.setWindowTitle('3D Embedding Viewer with Rastermap Integration')
        self.showMaximized()
    
        # Define constants for margins and spacing
        MAIN_CONTROLS_MARGIN = (5, 5, 5, 5)  # Left, Top, Right, Bottom
        HEATMAP_CONTROLS_MARGIN = (5, 5, 5, 5)
        CONTROLS_COLUMN_MARGIN = (5, 5, 5, 5)
        CONTROLS_COLUMN_SPACING = 5
        GROUPBOX_SPACING = 10
    
        # Create the main grid layout
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(*CONTROLS_COLUMN_MARGIN)
        grid_layout.setSpacing(10)  # Space between main widgets
        self.setLayout(grid_layout)
    
        # -----------------------------
        # Row 0: Four Main Plots
        # -----------------------------
        # 1) 3D Figure
        self.figure_3d = Figure()
        self.canvas_3d = FigureCanvas(self.figure_3d)
        self.canvas_3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)
    
        threeD_layout = QVBoxLayout()
        threeD_layout.setContentsMargins(0, 0, 0, 0)
        threeD_layout.setSpacing(5)
        threeD_layout.addWidget(self.canvas_3d)
        threeD_layout.addWidget(self.toolbar_3d)
        threeD_widget = QWidget()
        threeD_widget.setLayout(threeD_layout)
    
        # 2) 2D Figure
        self.figure_2d = Figure()
        self.canvas_2d = FigureCanvas(self.figure_2d)
        self.canvas_2d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        twoD_layout = QVBoxLayout()
        twoD_layout.setContentsMargins(0, 0, 0, 0)
        twoD_layout.setSpacing(5)
        twoD_layout.addWidget(self.canvas_2d)
        twoD_widget = QWidget()
        twoD_widget.setLayout(twoD_layout)
    
        # 3) Heatmap-Left
        self.figure_btm_left = Figure()
        self.canvas_btm_left = FigureCanvas(self.figure_btm_left)
        self.canvas_btm_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
        # 4) Heatmap-Right
        self.figure_btm_right = Figure()
        self.canvas_btm_right = FigureCanvas(self.figure_btm_right)
        self.canvas_btm_right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
        # Place these four plots in row=0
        grid_layout.addWidget(threeD_widget,          0, 0)
        grid_layout.addWidget(twoD_widget,            0, 1)
        grid_layout.addWidget(self.canvas_btm_left,   0, 2)
        grid_layout.addWidget(self.canvas_btm_right,  0, 3)
    
        # --------------------------------
        # Row 1: Four Placeholder Plots
        # --------------------------------
    
        # Placeholder 1: Under 3D Plot (Column 0) - Now replaced with Rastermap
        self.figure_rastermap = Figure()
        self.canvas_rastermap = FigureCanvas(self.figure_rastermap)
        self.canvas_rastermap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ax_raster = self.figure_rastermap.add_subplot(111)
        ax_raster.set_title("Spiking Rastermap")
        ax_raster.axis('off')  # Initial off
        self.placeholder_canvas_1 = self.canvas_rastermap  # Assign Rastermap to Placeholder 1
        grid_layout.addWidget(self.placeholder_canvas_1, 1, 0)
    
        # Placeholder 2: Under 2D Plot (Column 1)
        self.placeholder_2 = Figure()
        self.placeholder_canvas_2 = FigureCanvas(self.placeholder_2)
        self.placeholder_canvas_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ax2 = self.placeholder_2.add_subplot(111)
        ax2.set_title("Placeholder 2")
        ax2.axis('off')  # Hide axes
        grid_layout.addWidget(self.placeholder_canvas_2, 1, 1)
    
        # Placeholder 3: Under Heatmap-Left (Column 2) - Now a standard placeholder
        self.placeholder_3 = Figure()
        self.placeholder_canvas_3 = FigureCanvas(self.placeholder_3)
        self.placeholder_canvas_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ax3 = self.placeholder_3.add_subplot(111)
        ax3.set_title("Placeholder 3")
        ax3.axis('off')  # Hide axes
        grid_layout.addWidget(self.placeholder_canvas_3, 1, 2)
    
        # Placeholder 4: Under Heatmap-Right (Column 3)
        self.placeholder_4 = Figure()
        self.placeholder_canvas_4 = FigureCanvas(self.placeholder_4)
        self.placeholder_canvas_4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ax4 = self.placeholder_4.add_subplot(111)
        ax4.set_title("Placeholder 4")
        ax4.axis('off')
        grid_layout.addWidget(self.placeholder_canvas_4, 1, 3)
    
        # ----------------------------------------------------
        # Controls (in the far-right column, spanning two rows)
        # ----------------------------------------------------
        controls_column = QVBoxLayout()
        controls_column.setContentsMargins(*CONTROLS_COLUMN_MARGIN)
        controls_column.setSpacing(CONTROLS_COLUMN_SPACING)
    
        # ----- Main Plot Controls -----
        main_plot_group = QGroupBox("Main Plot Controls")
        main_plot_group.setStyleSheet("font-weight: bold;")
        main_plot_layout = QVBoxLayout()
        main_plot_layout.setContentsMargins(*MAIN_CONTROLS_MARGIN)
        main_plot_layout.setSpacing(5)
    
        # (A) Attribute Dropdown
        self.attribute_dropdown = QComboBox(self)
        self.attribute_dropdown.addItems(
            [
                'None',
                'highlighted points',
                'points_closest_to_archetypes',
                'points_between_archetypes',
            ]
            + list(self.attributes.columns)
        )
        self.attribute_dropdown.currentIndexChanged.connect(self.update_plot)
        main_plot_layout.addWidget(QLabel("Attribute Selection:"))
        main_plot_layout.addWidget(self.attribute_dropdown)
    
        # (B) Archetype Selection
        self.archetype_selection = QComboBox(self)
        self.archetype_selection.setVisible(False)
        self.archetype_selection.currentIndexChanged.connect(self.update_plot)
        main_plot_layout.addWidget(QLabel("Archetype Selection:"))
        main_plot_layout.addWidget(self.archetype_selection)
    
        # (C) Toggle Lasso
        self.toggle_button = QPushButton("Toggle Lasso Selection", self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.toggle_lasso_selection)
        main_plot_layout.addWidget(self.toggle_button)
    
        # (D) Save Points
        self.save_button = QPushButton("Save Selected Points", self)
        self.save_button.clicked.connect(self.save_selected_points)
        main_plot_layout.addWidget(self.save_button)
    
        # (E) Top Slider + Label
        slider_layout = QHBoxLayout()
        self.percentage_slider = QSlider(Qt.Horizontal, self)
        self.percentage_slider.setRange(1, 50)
        self.percentage_slider.setValue(5)
        self.percentage_slider.setTickPosition(QSlider.NoTicks)
        self.percentage_slider.setVisible(False)
        self.percentage_slider.valueChanged.connect(self.update_percentage)
        slider_layout.addWidget(self.percentage_slider)
    
        self.slider_value_label = QLabel(f"{self.percentage_slider.value()}%", self)
        self.slider_value_label.setVisible(False)
        slider_layout.addWidget(self.slider_value_label)
        main_plot_layout.addLayout(slider_layout)
    
        # (F) Bin Spinbox (Top)
        bins_layout = QHBoxLayout()
        self.bins_label = QLabel("Num Bins:")
        self.bins_label.setVisible(False)
        self.bin_spinbox = QSpinBox(self)
        self.bin_spinbox.setRange(1, 50)
        self.bin_spinbox.setValue(2)
        self.bin_spinbox.setVisible(False)
        self.bin_spinbox.valueChanged.connect(self.update_plot)
        bins_layout.addWidget(self.bins_label)
        bins_layout.addWidget(self.bin_spinbox)
        main_plot_layout.addLayout(bins_layout)
    
        main_plot_group.setLayout(main_plot_layout)
        controls_column.addWidget(main_plot_group, alignment=Qt.AlignTop)
    
        # ----- Heatmap Controls -----
        heatmap_group = QGroupBox("Heatmap Controls")
        heatmap_group.setStyleSheet("font-weight: bold;")
        heatmap_layout = QVBoxLayout()
        heatmap_layout.setContentsMargins(*HEATMAP_CONTROLS_MARGIN)
        heatmap_layout.setSpacing(5)
    
        # Left Heatmap Controls
        heatmap_layout.addWidget(QLabel("Left Heatmap Axes:"))
        self.x_left_dropdown = QComboBox(self)
        self.x_left_dropdown.addItems(["attribute", "archetype", "bin"])
        self.x_left_dropdown.setCurrentText("attribute")
        self.x_left_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)
        heatmap_layout.addWidget(self.x_left_dropdown)
    
        self.y_left_dropdown = QComboBox(self)
        self.y_left_dropdown.addItems(["attribute", "archetype", "bin"])
        self.y_left_dropdown.setCurrentText("archetype")
        self.y_left_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)
        heatmap_layout.addWidget(self.y_left_dropdown)
    
        # Bottom-Left Heatmap Color Attribute
        heatmap_layout.addWidget(QLabel("Left Heatmap Color Attribute:"))
        self.bottom_left_attr_dropdown = QComboBox(self)
        self.bottom_left_attr_dropdown.addItems(
            ["None", "highlighted points", "points_closest_to_archetypes", "points_between_archetypes"]
            + list(self.attributes.columns)
        )
        self.bottom_left_attr_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)
        heatmap_layout.addWidget(self.bottom_left_attr_dropdown)
    
        self.bottom_left_percentage_slider = QSlider(Qt.Horizontal, self)
        self.bottom_left_percentage_slider.setRange(1, 50)
        self.bottom_left_percentage_slider.setValue(5)
        self.bottom_left_percentage_slider.setTickPosition(QSlider.NoTicks)
        self.bottom_left_percentage_slider.valueChanged.connect(self.update_btm_left_heatmap)
        heatmap_layout.addWidget(self.bottom_left_percentage_slider)
    
        self.bottom_left_slider_label = QLabel("5%", self)
        heatmap_layout.addWidget(self.bottom_left_slider_label)
    
        self.bottom_left_bin_label = QLabel("Num Bins (Left):")
        heatmap_layout.addWidget(self.bottom_left_bin_label)
    
        self.bottom_left_bin_spinbox = QSpinBox(self)
        self.bottom_left_bin_spinbox.setRange(1, 50)
        self.bottom_left_bin_spinbox.setValue(2)
        self.bottom_left_bin_spinbox.valueChanged.connect(self.update_btm_left_heatmap)
        heatmap_layout.addWidget(self.bottom_left_bin_spinbox)
    
        # Right Heatmap Controls
        heatmap_layout.addWidget(QLabel("Right Heatmap Axes:"))
        self.x_right_dropdown = QComboBox(self)
        self.x_right_dropdown.addItems(["attribute", "archetype", "bin"])
        self.x_right_dropdown.setCurrentText("bin")
        self.x_right_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)
        heatmap_layout.addWidget(self.x_right_dropdown)
    
        self.y_right_dropdown = QComboBox(self)
        self.y_right_dropdown.addItems(["attribute", "archetype", "bin"])
        self.y_right_dropdown.setCurrentText("attribute")
        self.y_right_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)
        heatmap_layout.addWidget(self.y_right_dropdown)
    
        # Bottom-Right Heatmap Color Attribute
        heatmap_layout.addWidget(QLabel("Right Heatmap Color Attribute:"))
        self.bottom_right_attr_dropdown = QComboBox(self)
        self.bottom_right_attr_dropdown.addItems(
            ["None", "highlighted points", "points_closest_to_archetypes", "points_between_archetypes"]
            + list(self.attributes.columns)
        )
        self.bottom_right_attr_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)
        heatmap_layout.addWidget(self.bottom_right_attr_dropdown)
    
        self.bottom_right_percentage_slider = QSlider(Qt.Horizontal, self)
        self.bottom_right_percentage_slider.setRange(1, 50)
        self.bottom_right_percentage_slider.setValue(5)
        self.bottom_right_percentage_slider.setTickPosition(QSlider.NoTicks)
        self.bottom_right_percentage_slider.valueChanged.connect(self.update_btm_right_heatmap)
        heatmap_layout.addWidget(self.bottom_right_percentage_slider)
    
        self.bottom_right_slider_label = QLabel("5%", self)
        heatmap_layout.addWidget(self.bottom_right_slider_label)
    
        self.bottom_right_bin_label = QLabel("Num Bins (Right):")
        heatmap_layout.addWidget(self.bottom_right_bin_label)
    
        self.bottom_right_bin_spinbox = QSpinBox(self)
        self.bottom_right_bin_spinbox.setRange(1, 50)
        self.bottom_right_bin_spinbox.setValue(2)
        self.bottom_right_bin_spinbox.valueChanged.connect(self.update_btm_right_heatmap)
        heatmap_layout.addWidget(self.bottom_right_bin_spinbox)
    
        heatmap_group.setLayout(heatmap_layout)
        controls_column.addWidget(heatmap_group, alignment=Qt.AlignTop)
    
        # ----- Rastermap Controls -----
        rastermap_group = QGroupBox("Rastermap Controls")
        rastermap_group.setStyleSheet("font-weight: bold;")
        rastermap_layout = QVBoxLayout()
        rastermap_layout.setContentsMargins(5, 5, 5, 5)
        rastermap_layout.setSpacing(5)
    
        # (A) Archetype Selection Dropdown
        rastermap_layout.addWidget(QLabel("Select Archetype:"))
        self.rastermap_archetype_dropdown = QComboBox(self)
        self.populate_rastermap_archetypes()
        self.rastermap_archetype_dropdown.currentIndexChanged.connect(self.update_rastermap)
        rastermap_layout.addWidget(self.rastermap_archetype_dropdown)
    
        # (B) Percentage Slider
        rastermap_layout.addWidget(QLabel("Select X% of Datapoints:"))
        self.rastermap_percentage_slider = QSlider(Qt.Horizontal, self)
        self.rastermap_percentage_slider.setRange(1, 50)
        self.rastermap_percentage_slider.setValue(self.rastermap_percentage)
        self.rastermap_percentage_slider.setTickPosition(QSlider.NoTicks)
        self.rastermap_percentage_slider.valueChanged.connect(self.update_rastermap)
        rastermap_layout.addWidget(self.rastermap_percentage_slider)
    
        self.rastermap_slider_label = QLabel(f"{self.rastermap_percentage_slider.value()}%", self)
        rastermap_layout.addWidget(self.rastermap_slider_label)
    
        # (C) Run Rastermap Button
        self.run_rastermap_button = QPushButton("Run Rastermap", self)
        self.run_rastermap_button.clicked.connect(self.run_rastermap)
        rastermap_layout.addWidget(self.run_rastermap_button)
    
        rastermap_group.setLayout(rastermap_layout)
        controls_column.addWidget(rastermap_group, alignment=Qt.AlignTop)
    
        # Add a single stretch at the end of controls_column to push all controls up
        controls_column.addStretch()
    
        # Finalize the controls layout
        controls_widget = QWidget()
        controls_widget.setLayout(controls_column)
        # Place controls in column=4, spanning both rows (0 and 1)
        grid_layout.addWidget(controls_widget, 0, 4, 2, 1)
    
        # ------------------------------
        # Configure Grid Layout Stretches
        # ------------------------------
        # Set equal stretch for all four main columns
        for i in range(4):
            grid_layout.setColumnStretch(i, 1)
        # Controls column does not stretch
        grid_layout.setColumnStretch(4, 0)
    
        # Set equal row stretches for equal vertical sizes
        grid_layout.setRowStretch(0, 1)  # Top row
        grid_layout.setRowStretch(1, 1)  # Bottom row
    
        # ------------------------------
        # Uniform Widths for Controls
        # ------------------------------
        # Determine the maximum width among specific buttons
        button_width = max(
            self.toggle_button.sizeHint().width(),
            self.save_button.sizeHint().width(),
            self.run_rastermap_button.sizeHint().width()
        )
        # Apply the same width to all relevant widgets
        for w in [
            self.attribute_dropdown,
            self.archetype_selection,
            self.toggle_button,
            self.save_button,
            self.run_rastermap_button,
            self.percentage_slider,
            self.slider_value_label,
            self.bin_spinbox,
            self.bins_label,
            self.x_left_dropdown,
            self.y_left_dropdown,
            self.x_right_dropdown,
            self.y_right_dropdown,
        ]:
            w.setFixedWidth(button_width)
    
        for w in [
            self.bottom_left_attr_dropdown,
            self.bottom_left_percentage_slider,
            self.bottom_left_slider_label,
            self.bottom_left_bin_label,
            self.bottom_left_bin_spinbox,
            self.bottom_right_attr_dropdown,
            self.bottom_right_percentage_slider,
            self.bottom_right_slider_label,
            self.bottom_right_bin_label,
            self.bottom_right_bin_spinbox,
            self.rastermap_archetype_dropdown,
            self.rastermap_percentage_slider,
            self.rastermap_slider_label,
        ]:
            w.setFixedWidth(button_width)
    
        # ------------------------------
        # Initial Plotting
        # ------------------------------
        self.plot_embedding()
        self.update_rastermap()

    # -------------------------------------------------------------------------
    #  SYNCHRONIZE HIGHLIGHTED POINTS
    # -------------------------------------------------------------------------
    def sync_highlighted_points(self):
        self.highlighted_indices = self.shared_state.selected_points
        if self.attribute_dropdown.currentText() == 'highlighted points':
            self.plot_embedding(color_attribute='highlighted points')

    # -------------------------------------------------------------------------
    #  MASTER UPDATE (TOP PLOTS)
    # -------------------------------------------------------------------------
    def update_plot(self):
        selected_attr = self.attribute_dropdown.currentText()

        if selected_attr in self.attributes.columns:
            self.last_valid_attribute = selected_attr

        if self.last_valid_attribute is None and len(self.attributes.columns) > 0:
            self.last_valid_attribute = self.attributes.columns[0]

        if selected_attr in ["points_between_archetypes", "points_closest_to_archetypes"]:
            self.percentage_slider.setVisible(True)
            self.slider_value_label.setVisible(True)
            self.bin_spinbox.setVisible(True)
            self.bins_label.setVisible(True)

            if selected_attr == "points_between_archetypes":
                self.archetype_selection.setVisible(True)
                if not self.archetype_selection.count():
                    self.populate_archetype_selection()
            else:
                self.archetype_selection.setVisible(False)
        else:
            self.percentage_slider.setVisible(False)
            self.slider_value_label.setVisible(False)
            self.bin_spinbox.setVisible(False)
            self.bins_label.setVisible(False)
            self.archetype_selection.setVisible(False)

        # Re-plot (top)
        self.plot_embedding(color_attribute=selected_attr)
        # Update Rastermap archetypes if needed
        self.populate_rastermap_archetypes()

    def update_percentage(self):
        val = self.percentage_slider.value()
        self.slider_value_label.setText(f"{val}%")
        if not self.slider_value_label.isVisible():
            self.slider_value_label.setVisible(True)
        self.alpha_other = val / 100
        self.plot_embedding(color_attribute=self.attribute_dropdown.currentText())

    def populate_archetype_selection(self):
        self.ensure_archetypes_computed()
        if self.archetype_method == "ConvexHull" and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == "SISAL" and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            arch_points = None

        if arch_points is None:
            return

        self.archetype_selection.clear()
        pairs = []
        for i in range(len(arch_points)):
            for j in range(i + 1, len(arch_points)):
                pairs.append(f"{i + 1} - {j + 1}")
        self.archetype_selection.addItems(pairs)

    # -------------------------------------------------------------------------
    #  TOP PLOT + THEN calls bottom heatmap updates
    # -------------------------------------------------------------------------
    def plot_embedding(self, color_attribute=None):
        # Original 3D axis logic
        if not self.figure_3d.axes:
            self.figure_3d.clear()
            self.ax = self.figure_3d.add_subplot(111, projection='3d')
        else:
            self.ax = self.figure_3d.axes[0]

        n_points = self.embeddings.shape[0]
        colors = np.array(["k"] * n_points)

        # 1) "highlighted points"
        if color_attribute == "highlighted points":
            colors = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other))
            for idx in self.shared_state.selected_points:
                if idx < n_points:
                    colors[idx] = (1, 0, 0, 1)

        elif color_attribute == "points_closest_to_archetypes":
            pass

        elif color_attribute == "points_between_archetypes":
            pass

        elif color_attribute and color_attribute != "None":
            import matplotlib
            vals = self.attributes[color_attribute]
            norm = matplotlib.colors.Normalize(vals.min(), vals.max())
            colors = plt.cm.viridis_r(norm(vals))

        # Always compute top X% if recognized (for top usage)
        self.compute_top_xpct_indices()  # modifies self.closest_indices_per_archetype

        # Points_closest_to_archetypes (color logic)
        if color_attribute == "points_closest_to_archetypes" and self.closest_indices_per_archetype:
            colors = self.cumulative_bins_closest_archetypes()

        # Points_between_archetypes
        elif color_attribute == "points_between_archetypes":
            self.ensure_archetypes_computed()
            arch_points = None
            if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
                arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
            elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
                arch_points = self.cached_sisal_hull.T

            if arch_points is not None:
                selected_index = self.archetype_selection.currentIndex()
                if selected_index != -1:
                    pairs = [
                        (i, j)
                        for i in range(len(arch_points))
                        for j in range(i + 1, len(arch_points))
                    ]
                    if selected_index < len(pairs):
                        a, b = pairs[selected_index]
                        arch_a = arch_points[a]
                        arch_b = arch_points[b]
                        n_intermediate = self.bin_spinbox.value()
                        dist = np.linalg.norm(
                            self.embeddings[:, None, :3]
                            - np.linspace(arch_a, arch_b, n_intermediate)[None, :, :],
                            axis=2,
                        )
                        x_pct = self.percentage_slider.value()
                        n_closest = max(1, int((x_pct / 100) * n_points))
                        closest_indices = np.argsort(dist, axis=0)[:n_closest]
                        cmap = plt.get_cmap('viridis')
                        icolors = cmap(np.linspace(0, 1, n_intermediate))
                        colarray = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other), dtype=float)
                        for i_col, idxs_i in enumerate(closest_indices.T):
                            colarray[idxs_i, :3] = icolors[i_col, :3]
                            colarray[idxs_i, 3] = 1.0
                        colors = colarray

        # Recompute or update standard top-level heatmaps
        self.means_matrix = self.compute_attribute_archetype_means()
        self.bin_matrix = self.compute_bin_attribute_means()

        # Now update both bottom heatmaps (still the old approach for top)
        self.update_btm_left_heatmap()
        self.update_btm_right_heatmap()

        # Draw 3D scatter
        self.ax.clear()
        self.scatter = self.ax.scatter(
            self._scaled_embeddings[:, 0],
            self._scaled_embeddings[:, 1],
            self._scaled_embeddings[:, 2],
            c=colors, marker='.'
        )
        self.ax.set_title(
            f"Session: {self.session_id}\n"
            f"Regions: {', '.join(sorted(self.brain_regions))}\n"
            f"Algorithm: {self.algorithm}",
            fontsize=12
        )
        self.ax.set_xlabel('Dimension 1')
        self.ax.set_ylabel('Dimension 2')
        self.ax.set_zlabel('Dimension 3')

        # 2D XY, XZ, YZ
        self.figure_2d.clear()
        self.ax_xy = self.figure_2d.add_subplot(311, aspect='equal')
        self.ax_xz = self.figure_2d.add_subplot(312, aspect='equal')
        self.ax_yz = self.figure_2d.add_subplot(313, aspect='equal')

        self.scatter_xy = self.ax_xy.scatter(
            self.embeddings[:, 0], self.embeddings[:, 1], c=colors, marker='.'
        )
        self.scatter_xz = self.ax_xz.scatter(
            self.embeddings[:, 0], self.embeddings[:, 2], c=colors, marker='.'
        )
        self.scatter_yz = self.ax_yz.scatter(
            self.embeddings[:, 1], self.embeddings[:, 2], c=colors, marker='.'
        )

        # Axis buffer
        for ax_obj, (xx, yy) in zip(
            [self.ax_xy, self.ax_xz, self.ax_yz],
            [
                (self.embeddings[:, 0], self.embeddings[:, 1]),
                (self.embeddings[:, 0], self.embeddings[:, 2]),
                (self.embeddings[:, 1], self.embeddings[:, 2])
            ]
        ):
            x_min, x_max = xx.min(), xx.max()
            y_min, y_max = yy.min(), yy.max()
            buffer_x = (x_max - x_min) * 0.2
            buffer_y = (y_max - y_min) * 0.2
            ax_obj.set_xlim(x_min - buffer_x, x_max + buffer_x)
            ax_obj.set_ylim(y_min - buffer_y, y_max + buffer_y)
            ax_obj.set_aspect('equal', adjustable='datalim')

        self.ax_xy.set_title("1-2 Dim")
        self.ax_xz.set_title("1-3 Dim")
        self.ax_yz.set_title("2-3 Dim")

        for ax_obj in [self.ax_xy, self.ax_xz, self.ax_yz]:
            ax_obj.spines['top'].set_visible(False)
            ax_obj.spines['right'].set_visible(False)
            ax_obj.spines['left'].set_visible(False)
            ax_obj.spines['bottom'].set_visible(False)
            ax_obj.set_xticks([])
            ax_obj.set_yticks([])

        self.canvas_2d.draw_idle()
        self.canvas_3d.draw_idle()

        # If hull lines
        if self.archetype_method == 'ConvexHull':
            self.add_convex_hull(self.ax, fig_type='3d')
            self.add_convex_hull(self.ax_xy, fig_type='2d', projection=(0, 1))
            self.add_convex_hull(self.ax_xz, fig_type='2d', projection=(0, 2))
            self.add_convex_hull(self.ax_yz, fig_type='2d', projection=(1, 2))
        if self.archetype_method == 'SISAL':
            self.add_sisal_hull(self.ax, fig_type='3d')
            self.add_sisal_hull(self.ax_xy, fig_type='2d', projection=(0, 1))
            self.add_sisal_hull(self.ax_xz, fig_type='2d', projection=(0, 2))
            self.add_sisal_hull(self.ax_yz, fig_type='2d', projection=(1, 2))

        # Lasso reinit
        if hasattr(self, 'selector') and self.selector and self.selector.selector_enabled:
            self.selector.toggle_selection_mode()
        if hasattr(self, 'lasso_xy') and self.lasso_xy and self.lasso_xy.selector_enabled:
            self.lasso_xy.toggle_selection_mode()
        if hasattr(self, 'lasso_xz') and self.lasso_xz and self.lasso_xz.selector_enabled:
            self.lasso_xz.toggle_selection_mode()
        if hasattr(self, 'lasso_yz') and self.lasso_yz and self.lasso_yz.selector_enabled:
            self.lasso_yz.toggle_selection_mode()

        self.selector = SelectFromCollectionLasso(self.ax, self.scatter, self.on_select_3d, is_3d=True)
        self.lasso_xy = SelectFromCollectionLasso(self.ax_xy, self.scatter_xy, self.on_select_2d)
        self.lasso_xz = SelectFromCollectionLasso(self.ax_xz, self.scatter_xz, self.on_select_2d)
        self.lasso_yz = SelectFromCollectionLasso(self.ax_yz, self.scatter_yz, self.on_select_2d)
        if self.lasso_active:
            self.selector.toggle_selection_mode()
            self.lasso_xy.toggle_selection_mode()
            self.lasso_xz.toggle_selection_mode()
            self.lasso_yz.toggle_selection_mode()
    
    def compute_bin_attribute_means(self):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None

        arch_a = arch_points[0]
        arch_b = arch_points[1]
        inter_pts = np.linspace(arch_a, arch_b, self.bin_spinbox.value())

        n_points = self.embeddings.shape[0]
        x_pct = self.percentage_slider.value()
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :],
            axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, inter_pts.shape[0]), dtype=float)
        for bin_i in range(inter_pts.shape[0]):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_attribute_archetype_means(self):
        if not self.closest_indices_per_archetype:
            return None
        n_archetypes = len(self.closest_indices_per_archetype)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M

    # -------------------------------------------------------------------------
    #  BOTTOM-LEFT HEATMAP
    # -------------------------------------------------------------------------
    def update_btm_left_heatmap(self):
        """Recompute & redraw ONLY the bottom-left heatmap using the bottom-left slider/spinbox."""
        selected_attr = self.bottom_left_attr_dropdown.currentText()
        if selected_attr in self.attributes.columns:
            self.last_valid_attribute_left = selected_attr
        elif not self.last_valid_attribute_left and len(self.attributes.columns) > 0:
            self.last_valid_attribute_left = self.attributes.columns[0]

        # Grab bottom-left slider/spinbox
        x_pct = self.bottom_left_percentage_slider.value()
        self.bottom_left_slider_label.setText(f"{x_pct}%")
        n_bins = self.bottom_left_bin_spinbox.value()

        # 1) Compute a new set of 'closest_indices_per_archetype_left' based on x_pct
        self.compute_bottom_left_xpct_indices(x_pct)

        # 2) Build means_matrix_left / bin_matrix_left from that
        self.means_matrix_left = self.compute_attribute_archetype_means_bottom_left()
        self.bin_matrix_left = self.compute_bin_attribute_means_bottom_left(x_pct, n_bins)

        # Now do the bottom-left pcolormesh
        dim_x = self.x_left_dropdown.currentText()
        dim_y = self.y_left_dropdown.currentText()

        self.figure_btm_left.clear()
        ax = self.figure_btm_left.add_subplot(111)
        ax.set_title(f"Bottom-Left Heatmap ({selected_attr})", fontsize=10)

        matrix, x_labels, y_labels = self.get_heatmap_data_bottom_left(dim_x, dim_y)
        if matrix is None:
            self.canvas_btm_left.draw_idle()
            return

        pcm = ax.pcolormesh(
            matrix, cmap="viridis_r", edgecolors="k", linewidth=1, shading="flat"
        )
        cbar = self.figure_btm_left.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Value", rotation=270, labelpad=15)

        ax.set_xticks(np.arange(len(y_labels)) + 0.5)
        ax.set_xticklabels(y_labels)
        ax.set_yticks(np.arange(len(x_labels)) + 0.5)
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(dim_y.capitalize())
        ax.set_ylabel(dim_x.capitalize())
        ax.invert_yaxis()

        self.canvas_btm_left.draw_idle()

    # -------------------------------------------------------------------------
    #  BOTTOM-RIGHT HEATMAP
    # -------------------------------------------------------------------------
    def update_btm_right_heatmap(self):
        """Recompute & redraw ONLY the bottom-right heatmap using the bottom-right slider/spinbox."""
        selected_attr = self.bottom_right_attr_dropdown.currentText()
        if selected_attr in self.attributes.columns:
            self.last_valid_attribute_right = selected_attr
        elif not self.last_valid_attribute_right and len(self.attributes.columns) > 0:
            self.last_valid_attribute_right = self.attributes.columns[0]

        # Grab bottom-right slider/spinbox
        x_pct = self.bottom_right_percentage_slider.value()
        self.bottom_right_slider_label.setText(f"{x_pct}%")
        n_bins = self.bottom_right_bin_spinbox.value()

        # 1) Compute a new set of 'closest_indices_per_archetype_right' based on x_pct
        self.compute_bottom_right_xpct_indices(x_pct)

        # 2) Build means_matrix_right / bin_matrix_right
        self.means_matrix_right = self.compute_attribute_archetype_means_bottom_right()
        self.bin_matrix_right = self.compute_bin_attribute_means_bottom_right(x_pct, n_bins)

        # Now do the bottom-right pcolormesh
        dim_x = self.x_right_dropdown.currentText()
        dim_y = self.y_right_dropdown.currentText()

        self.figure_btm_right.clear()
        ax = self.figure_btm_right.add_subplot(111)
        ax.set_title(f"Bottom-Right Heatmap ({selected_attr})", fontsize=10)

        matrix, x_labels, y_labels = self.get_heatmap_data_bottom_right(dim_x, dim_y)
        if matrix is None:
            self.canvas_btm_right.draw_idle()
            return

        pcm = ax.pcolormesh(
            matrix, cmap="viridis_r", edgecolors="k", linewidth=1, shading="flat"
        )
        cbar = self.figure_btm_right.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Value", rotation=270, labelpad=15)

        ax.set_xticks(np.arange(len(y_labels)) + 0.5)
        ax.set_xticklabels(y_labels)
        ax.set_yticks(np.arange(len(x_labels)) + 0.5)
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(dim_y.capitalize())
        ax.set_ylabel(dim_x.capitalize())
        ax.invert_yaxis()

        self.canvas_btm_right.draw_idle()

    # -------------------------------------------------------------------------
    #  BOTTOM LEFT: get_heatmap_data_bottom_left
    # -------------------------------------------------------------------------
    def get_heatmap_data_bottom_left(self, dim_x, dim_y):
        """
        Same concept as get_heatmap_data, but uses self.means_matrix_left / self.bin_matrix_left
        and self.last_valid_attribute_left instead of top's variables.
        """
        combos = {}

        # 1) attribute x archetype
        if self.means_matrix_left is not None:
            n_archetypes = self.means_matrix_left.shape[1]
            arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "archetype")] = (self.means_matrix_left, attrs, arch_labels)
            combos[("archetype", "attribute")] = (self.means_matrix_left.T, arch_labels, attrs)

        # 2) attribute x bin
        if self.bin_matrix_left is not None:
            n_bins = self.bin_matrix_left.shape[1]
            bin_labels = [f"Bin {i+1}" for i in range(n_bins)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "bin")] = (self.bin_matrix_left, attrs, bin_labels)
            combos[("bin", "attribute")] = (self.bin_matrix_left.T, bin_labels, attrs)

        # 3) If user chooses (archetype, bin) or (bin, archetype), do bottom-left cumulative approach
        if (dim_x, dim_y) in [("archetype", "bin"), ("bin", "archetype")]:
            chosen_attr = self.last_valid_attribute_left
            if not chosen_attr or chosen_attr not in self.attributes.columns:
                chosen_attr = self.attributes.columns[0]  # fallback
            A = self.compute_archetype_bin_attribute_means_cumulative_bottom_left(chosen_attr)
            if A is not None:
                n_archetypes = A.shape[0]
                n_bins = A.shape[1]
                arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
                bin_labels = [f"Bin {j+1}" for j in range(n_bins)]
                if (dim_x, dim_y) == ("archetype", "bin"):
                    return A, arch_labels, bin_labels
                else:
                    return A.T, bin_labels, arch_labels
            return None, None, None

        return combos.get((dim_x, dim_y), (None, None, None))

    # -------------------------------------------------------------------------
    #  BOTTOM RIGHT: get_heatmap_data_bottom_right
    # -------------------------------------------------------------------------
    def get_heatmap_data_bottom_right(self, dim_x, dim_y):
        """
        Same concept as get_heatmap_data_bottom_left, but for right side.
        """
        combos = {}

        if self.means_matrix_right is not None:
            n_archetypes = self.means_matrix_right.shape[1]
            arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "archetype")] = (self.means_matrix_right, attrs, arch_labels)
            combos[("archetype", "attribute")] = (self.means_matrix_right.T, arch_labels, attrs)

        if self.bin_matrix_right is not None:
            n_bins = self.bin_matrix_right.shape[1]
            bin_labels = [f"Bin {i+1}" for i in range(n_bins)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "bin")] = (self.bin_matrix_right, attrs, bin_labels)
            combos[("bin", "attribute")] = (self.bin_matrix_right.T, bin_labels, attrs)

        if (dim_x, dim_y) in [("archetype", "bin"), ("bin", "archetype")]:
            chosen_attr = self.last_valid_attribute_right
            if not chosen_attr or chosen_attr not in self.attributes.columns:
                chosen_attr = self.attributes.columns[0]
            A = self.compute_archetype_bin_attribute_means_cumulative_bottom_right(chosen_attr)
            if A is not None:
                n_archetypes = A.shape[0]
                n_bins = A.shape[1]
                arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
                bin_labels = [f"Bin {j+1}" for j in range(n_bins)]
                if (dim_x, dim_y) == ("archetype", "bin"):
                    return A, arch_labels, bin_labels
                else:
                    return A.T, bin_labels, arch_labels
            return None, None, None

        return combos.get((dim_x, dim_y), (None, None, None))

    # -------------------------------------------------------------------------
    #  BOTTOM LEFT: separate compute logic
    # -------------------------------------------------------------------------
    def compute_bottom_left_xpct_indices(self, x_pct):
        """
        We'll define a new 'closest_indices_per_archetype_left' using x_pct
        and ignoring self.percentage_slider from the top.
        """
        self.closest_indices_per_archetype_left = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        arch_pts_3d = None
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        if arch_pts_3d is None or len(arch_pts_3d) == 0:
            return

        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )  # shape (n_points, n_archetypes)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        n_archetypes = arch_pts_3d.shape[0]
        self.closest_indices_per_archetype_left = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def compute_attribute_archetype_means_bottom_left(self):
        if not self.closest_indices_per_archetype_left:
            return None
        n_archetypes = len(self.closest_indices_per_archetype_left)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype_left):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        # row-wise normalize
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M

    def compute_bin_attribute_means_bottom_left(self, x_pct, n_bins):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None

        arch_a = arch_points[0]
        arch_b = arch_points[1]
        inter_pts = np.linspace(arch_a, arch_b, n_bins)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :], axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, n_bins), dtype=float)
        for bin_i in range(n_bins):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_archetype_bin_attribute_means_cumulative_bottom_left(self, chosen_attr):
        if self.closest_indices_per_archetype_left is None:
            return None
        if chosen_attr not in self.attributes.columns:
            return None

        n_archetypes = len(self.closest_indices_per_archetype_left)
        if n_archetypes == 0:
            return None

        n_points = self.embeddings.shape[0]
        x_pct = self.bottom_left_percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bottom_left_bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        M = np.full((n_archetypes, n_bins), np.nan, dtype=float)

        for j in range(n_archetypes):
            idxs_j = self.closest_indices_per_archetype_left[j]
            if len(idxs_j) == 0:
                continue
            # Distances for shading logic
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) > 0:
                    val = self.attributes.iloc[bin_points][chosen_attr].mean()
                    M[j, i_bin] = val
        return M

    # -------------------------------------------------------------------------
    #  BOTTOM RIGHT: separate compute logic
    # -------------------------------------------------------------------------
    def compute_bottom_right_xpct_indices(self, x_pct):
        self.closest_indices_per_archetype_right = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        arch_pts_3d = None
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        if arch_pts_3d is None or len(arch_pts_3d) == 0:
            return

        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )
        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        n_archetypes = arch_pts_3d.shape[0]
        self.closest_indices_per_archetype_right = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def compute_attribute_archetype_means_bottom_right(self):
        if not self.closest_indices_per_archetype_right:
            return None
        n_archetypes = len(self.closest_indices_per_archetype_right)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype_right):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M

    def compute_bin_attribute_means_bottom_right(self, x_pct, n_bins):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None

        arch_a = arch_points[0]
        arch_b = arch_points[1]
        inter_pts = np.linspace(arch_a, arch_b, n_bins)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :], axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, n_bins), dtype=float)
        for bin_i in range(n_bins):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_archetype_bin_attribute_means_cumulative_bottom_right(self, chosen_attr):
        if self.closest_indices_per_archetype_right is None:
            return None
        if chosen_attr not in self.attributes.columns:
            return None

        n_archetypes = len(self.closest_indices_per_archetype_right)
        if n_archetypes == 0:
            return None

        n_points = self.embeddings.shape[0]
        x_pct = self.bottom_right_percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bottom_right_bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        M = np.full((n_archetypes, n_bins), np.nan, dtype=float)

        for j in range(n_archetypes):
            idxs_j = self.closest_indices_per_archetype_right[j]
            if len(idxs_j) == 0:
                continue
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) > 0:
                    val = self.attributes.iloc[bin_points][chosen_attr].mean()
                    M[j, i_bin] = val
        return M

    def ensure_archetypes_computed(self):
        if self.archetype_method == "ConvexHull" and self.cached_convex_hull is None:
            self.cached_convex_hull = ConvexHull(self.embeddings[:, :3])
        elif self.archetype_method == "SISAL" and self.cached_sisal_hull is None:
            ArchsMin, _ = findMinSimplex(10, self.embeddings, 1, 4)
            self.cached_sisal_hull = ArchsMin[:, np.argsort(ArchsMin[0, :])]

    # -------------------------------------------------------------------------
    #  CONVEX / SISAL hulls
    # -------------------------------------------------------------------------
    def add_convex_hull(self, ax, fig_type='3d', projection=None):
        try:
            if self.cached_convex_hull is None:
                self.cached_convex_hull = ConvexHull(self.embeddings[:, :3])
            hull = self.cached_convex_hull
            if fig_type == '3d':
                vertices_3d = self.embeddings[hull.vertices, :3]
                edges = []
                for simplex in hull.simplices:
                    edges.append(self.embeddings[simplex, :3])
                edge_collection = Line3DCollection(edges, colors='k', linewidths=0.5)
                ax.add_collection3d(edge_collection)
                ax.scatter(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2],
                           c='b', s=10, label='Archetypes')
                for i, (x, y, z) in enumerate(vertices_3d):
                    ax.text(x, y, z, str(i + 1), color='red', fontsize=10)
            elif fig_type == '2d':
                if projection is None:
                    return
                vertices_2d = self.embeddings[hull.vertices, projection]
                edges = []
                for simplex in hull.simplices:
                    edge = self.embeddings[simplex, projection]
                    edges.append(edge)
                    ax.plot(*edge.T, c='k', linewidth=0.5)
                ax.scatter(vertices_2d[:, 0], vertices_2d[:, 1],
                           c='b', s=10, label='Archetypes', zorder=5)
                for i, (xx, yy) in enumerate(vertices_2d):
                    ax.text(xx, yy, str(i + 1), color='red', fontsize=10, zorder=6)
                self.adjust_axis_limits(ax, self.embeddings[:, projection], vertices_2d)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error in adding ConvexHull: {e}')

    def add_sisal_hull(self, ax, fig_type='3d', projection=None):
        try:
            if self.cached_sisal_hull is None:
                ArchsMin, VolArchReal = findMinSimplex(10, self.embeddings, 1, 4)
                ArchsOrder = np.argsort(ArchsMin[0, :])
                self.cached_sisal_hull = ArchsMin[:, ArchsOrder]
            ArchsMin = self.cached_sisal_hull
            NArchetypes = ArchsMin.shape[1]
            if fig_type == '3d':
                ArchsMin_3d = ArchsMin[:3, :]
                edges = []
                for i in range(NArchetypes):
                    for j in range(i + 1, NArchetypes):
                        edge = np.array([ArchsMin_3d[:, i], ArchsMin_3d[:, j]])
                        edges.append(edge)
                edge_collection = Line3DCollection(edges, colors='k', linewidths=0.5)
                ax.add_collection3d(edge_collection)
                ax.scatter(ArchsMin_3d[0, :], ArchsMin_3d[1, :], ArchsMin_3d[2, :],
                           c='b', s=10, label='Archetypes')
                for i, (x, y, z) in enumerate(ArchsMin_3d.T):
                    ax.text(x, y, z, str(i + 1), color='red', fontsize=10)
            elif fig_type == '2d':
                if projection is None:
                    return
                ArchsMin_2d = ArchsMin[list(projection), :]
                edges = []
                for i in range(NArchetypes):
                    for j in range(i + 1, NArchetypes):
                        edge = np.array([ArchsMin_2d[:, i], ArchsMin_2d[:, j]])
                        edges.append(edge)
                        ax.plot(*edge.T, c='k', linewidth=0.5)
                ax.scatter(ArchsMin_2d[0, :], ArchsMin_2d[1, :],
                           c='b', s=10, label='Archetypes', zorder=5)
                for i, (xx, yy) in enumerate(ArchsMin_2d.T):
                    ax.text(xx, yy, str(i + 1), color='red', fontsize=10, zorder=6)
                scatter_points = self.embeddings[:, projection]
                self.adjust_axis_limits(ax, scatter_points, ArchsMin_2d.T)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error in adding SISAL hull: {e}')

    def adjust_axis_limits(self, ax, scatter_points, polytope_points):
        all_points = np.vstack([scatter_points, polytope_points])
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        buffer_x = (x_max - x_min) * 0.1
        buffer_y = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
        ax.set_ylim(y_min - buffer_y, y_max + buffer_y)

    def invalidate_cache(self):
        self.cached_convex_hull = None
        self.cached_sisal_hull = None

    # -------------------------------------------------------------------------
    #  LASSO
    # -------------------------------------------------------------------------
    def on_select_3d(self, indices):
        self.highlighted_indices = list(indices)

    def on_select_2d(self, indices):
        self.highlighted_indices = list(indices)

    def toggle_lasso_selection(self):
        if self.lasso_active:
            self.lasso_active = False
            if self.selector and self.selector.selector_enabled:
                self.selector.toggle_selection_mode()
            if self.lasso_xy and self.lasso_xy.selector_enabled:
                self.lasso_xy.toggle_selection_mode()
            if self.lasso_xz and self.lasso_xz.selector_enabled:
                self.lasso_xz.toggle_selection_mode()
            if self.lasso_yz and self.lasso_yz.selector_enabled:
                self.lasso_yz.toggle_selection_mode()
            self.canvas_3d.draw_idle()
            self.canvas_2d.draw_idle()
        else:
            self.lasso_active = True
            if self.selector:
                self.selector.toggle_selection_mode()
            if self.lasso_xy:
                self.lasso_xy.toggle_selection_mode()
            if self.lasso_xz:
                self.lasso_xz.toggle_selection_mode()
            if self.lasso_yz:
                self.lasso_yz.toggle_selection_mode()

    def save_selected_points(self):
        self.selected_points = list(self.highlighted_indices)
        self.shared_state.set_selected_points(self.selected_points)
        print(f"Selected points saved: {self.selected_points}")
        if self.attribute_dropdown.currentText() == 'highlighted points':
            self.plot_embedding(color_attribute='highlighted points')

    # -------------------------------------------------------------------------
    #  "points_closest_to_archetypes" color logic
    # -------------------------------------------------------------------------
    def compute_top_xpct_indices(self):
        """
        For each archetype j, define top X% by distance => closest_indices_per_archetype[j].
        We do this for potential usage in color or in the heatmaps.
        """
        self.closest_indices_per_archetype = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        else:
            return
        n_archetypes = arch_pts_3d.shape[0]
        if n_archetypes == 0:
            return
        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )  # shape (n_points, n_archetypes)

        x_pct = self.percentage_slider.value()
        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        self.closest_indices_per_archetype = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def cumulative_bins_closest_archetypes(self):
        """
        This sets the scatter color per chunk for each archetype in 'points_closest_to_archetypes'.
        Returns RGBA array of shape (n_points, 4).
        """
        import matplotlib
        import colorsys
        n_points = self.embeddings.shape[0]
        colors = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other), dtype=float)

        n_archetypes = len(self.closest_indices_per_archetype)
        if n_archetypes == 0:
            return colors

        base_cmap = matplotlib.cm.get_cmap("viridis")
        base_colors = base_cmap(np.linspace(0, 1, n_archetypes))

        x_pct = self.percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        for j, idxs_j in enumerate(self.closest_indices_per_archetype):
            if len(idxs_j) == 0:
                continue
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            base_r, base_g, base_b, _ = base_colors[j]
            h, l, s = colorsys.rgb_to_hls(base_r, base_g, base_b)

            min_lightness = 0.3
            if l < min_lightness:
                l = min_lightness

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) == 0:
                    break

                fraction = (i_bin + 1) / (n_bins + 1)
                lightness_increment = fraction * 0.3
                new_l = min(l + lightness_increment, 1.0)

                shaded_r, shaded_g, shaded_b = colorsys.hls_to_rgb(h, new_l, s)
                colors[bin_points, :3] = (shaded_r, shaded_g, shaded_b)
                colors[bin_points, 3] = 1.0

        return colors

    def get_archetype_coords(self, j):
        """Return the j-th archetype's 3D coords from either convex hull or SISAL."""
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull is not None:
            arch_points_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
            if j < len(arch_points_3d):
                return arch_points_3d[j]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points_3d = self.cached_sisal_hull.T[:, :3]
            if j < len(arch_points_3d):
                return arch_points_3d[j]
        return np.array([0, 0, 0], dtype=float)

    # -------------------------------------------------------------------------
    #  Rastermap Controls and Plotting
    # -------------------------------------------------------------------------
    def populate_rastermap_archetypes(self):
        """Populate the rastermap archetype dropdown based on current archetypes."""
        self.rastermap_archetype_dropdown.blockSignals(True)
        self.rastermap_archetype_dropdown.clear()
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T[:, :3]
        else:
            arch_points = None

        if arch_points is not None:
            for i in range(len(arch_points)):
                self.rastermap_archetype_dropdown.addItem(f"Archetype {i+1}")
        else:
            self.rastermap_archetype_dropdown.addItem("No Archetypes")
        self.rastermap_archetype_dropdown.blockSignals(False)

    def update_rastermap(self):
        """Update the rastermap based on selected archetype and percentage."""
        selected_text = self.rastermap_archetype_dropdown.currentText()
        if selected_text.startswith("Archetype"):
            try:
                archetype_index = int(selected_text.split(" ")[1]) - 1
            except:
                archetype_index = None
        else:
            archetype_index = None

        if archetype_index is None:
            # Clear the rastermap
            ax = self.figure_rastermap.axes[0]
            ax.clear()
            ax.set_title("Spiking Rastermap")
            ax.axis('off')
            self.canvas_rastermap.draw_idle()
            return

        # Get x% from slider
        x_pct = self.rastermap_percentage_slider.value()
        self.rastermap_slider_label.setText(f"{x_pct}%")

        # Get the closest datapoints to the selected archetype
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T[:, :3]
        else:
            arch_points = None

        if arch_points is None or archetype_index >= len(arch_points):
            # Clear the rastermap
            ax = self.figure_rastermap.axes[0]
            ax.clear()
            ax.set_title("Spiking Rastermap")
            ax.axis('off')
            self.canvas_rastermap.draw_idle()
            return

        arch_coord = arch_points[archetype_index]
        dist = np.linalg.norm(self.embeddings[:, :3] - arch_coord, axis=1)
        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist)[:n_closest]

        # Retrieve the spike counts for the selected subset
        spike_matrix = self.session.neuron_x_time_matrix_subset.iloc[:, closest_indices]

        # Plot the rastermap
        ax = self.figure_rastermap.axes[0]
        ax.clear()

        if spike_matrix.empty:
            ax.set_title("Spiking Rastermap")
            ax.axis('off')
            self.canvas_rastermap.draw_idle()
            return

        # Create a binary rastermap where spikes are marked
        # raster_data = spike_matrix.values > 0  # True where spike count > 0
        raster_data = spike_matrix.values #> 0  # True where spike count > 0
        import matplotlib.colors as colors
        norm = colors.Normalize(vmin=0, vmax=np.percentile(raster_data, 99))
        ax.imshow(raster_data, aspect='auto', cmap='Greys', interpolation='none', norm=norm)
        ax.set_title(f"Spiking Rastermap - {selected_text} ({x_pct}%)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Neuron")
        ax.set_yticks([])  # Hide neuron ticks
        ax.set_xticks([])  # Hide time ticks

        self.canvas_rastermap.draw_idle()

    # -------------------------------------------------------------------------
    #  Run Rastermap Algorithm
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
#  Run Rastermap Algorithm
# -------------------------------------------------------------------------
    def run_rastermap(self):
        """Apply the Rastermap algorithm to the spike data and visualize the embedding."""
        try:
            # Get selected archetype
            selected_text = self.rastermap_archetype_dropdown.currentText()
            if selected_text.startswith("Archetype"):
                try:
                    archetype_index = int(selected_text.split(" ")[1]) - 1
                except:
                    archetype_index = None
            else:
                archetype_index = None
    
            if archetype_index is None:
                QMessageBox.warning(self, 'Error', 'No valid archetype selected.')
                return
    
            # Get x% from slider
            x_pct = self.rastermap_percentage_slider.value()
            n_points = self.embeddings.shape[0]
            n_closest = max(1, int((x_pct / 100) * n_points))
    
            # Get archetype coordinates
            if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
                arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
            elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
                arch_points = self.cached_sisal_hull.T[:, :3]
            else:
                arch_points = None
    
            if arch_points is None or archetype_index >= len(arch_points):
                QMessageBox.warning(self, 'Error', 'Selected archetype has no data points.')
                return
    
            arch_coord = arch_points[archetype_index]
            dist = np.linalg.norm(self.embeddings[:, :3] - arch_coord, axis=1)
            closest_indices = np.argsort(dist)[:n_closest]
    
            # Extract spike data for the closest_indices
            spks_df = self.session.neuron_x_time_matrix_subset.iloc[:, closest_indices]
            if spks_df.empty:
                QMessageBox.warning(self, 'Error', 'Selected subset of spike data is empty.')
                return
    
            spks = spks_df.values.astype("float32")
            
            epsilon = 1e-10  # Very small value
            # # Adding epsilon to the columns of the rows indicated by non_zero_rows
            non_zero_rows = ~np.all(spks == 0, axis=1)
            spks[~non_zero_rows, :] = 0
            spks[~non_zero_rows, 0] = epsilon
    
            # Disable the button to prevent multiple clicks
            self.run_rastermap_button.setEnabled(False)
            self.run_rastermap_button.setText("Running...")
    
    
            
            # Fit Rastermap
            model = Rastermap(n_PCs=200, n_clusters=100, 
                              locality=0.75, time_lag_window=5).fit(spks)
    
            # Retrieve embeddings
            y = model.embedding  # Typically neurons x 1
            isort = model.isort
            X_embedding = model.X_embedding
    
            # Visualize Rastermap embedding in the Rastermap plot area
            ax = self.figure_rastermap.axes[0]
            ax.clear()
    
            if X_embedding is None or X_embedding.size == 0:
                ax.set_title("Rastermap Embedding")
                ax.axis('off')
                self.canvas_rastermap.draw_idle()
                QMessageBox.warning(self, 'Error', 'Rastermap embedding is empty.')
                self.run_rastermap_button.setEnabled(True)
                self.run_rastermap_button.setText("Run Rastermap")
                return
    
            # ----- Updated Part Begins -----
            # Create a raster map scaled by spike counts
            raster_data = spks_df.values  # Use actual spike counts
    
            import matplotlib.colors as colors
            # Normalize to the 99th percentile to reduce the impact of outliers
            norm = colors.Normalize(vmin=0, vmax=np.percentile(raster_data, 99))
            ax.imshow(spks[isort], aspect='auto', cmap='Greys', interpolation='none', norm=norm)
            # Alternatively, use normalization with max value
            # norm = colors.Normalize(vmin=0, vmax=raster_data.max())
            # ax.imshow(raster_data, aspect='auto', cmap='Greys', interpolation='none', norm=norm)
    
            # If you prefer logarithmic scaling (optional)
            # from matplotlib.colors import LogNorm
            # raster_data = np.where(spks_df.values > 0, spks_df.values, 1)  # Replace 0 with 1 to avoid log(0)
            # norm = LogNorm(vmin=1, vmax=np.percentile(raster_data, 99))
            # ax.imshow(raster_data, aspect='auto', cmap='Greys', interpolation='none', norm=norm)
            # ----- Updated Part Ends -----
    
            ax.set_title("Rastermap Embedding", fontsize=10)
            ax.set_xlabel("Bins")
            ax.set_ylabel("Neurons")
            ax.set_yticks([])  # Hide neuron ticks
            ax.set_xticks([])  # Hide bin ticks
    
            self.canvas_rastermap.draw_idle()
    
            QMessageBox.information(self, 'Success', 'Rastermap algorithm applied successfully.')
    
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error in running Rastermap: {e}')
        finally:
            # Re-enable the button
            self.run_rastermap_button.setEnabled(True)
            self.run_rastermap_button.setText("Run Rastermap")


    # -------------------------------------------------------------------------
    #  "points_closest_to_archetypes" color logic
    # -------------------------------------------------------------------------
    def compute_top_xpct_indices(self):
        """
        For each archetype j, define top X% by distance => closest_indices_per_archetype[j].
        We do this for potential usage in color or in the heatmaps.
        """
        self.closest_indices_per_archetype = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        else:
            return
        n_archetypes = arch_pts_3d.shape[0]
        if n_archetypes == 0:
            return
        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )  # shape (n_points, n_archetypes)

        x_pct = self.percentage_slider.value()
        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        self.closest_indices_per_archetype = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def cumulative_bins_closest_archetypes(self):
        """
        This sets the scatter color per chunk for each archetype in 'points_closest_to_archetypes'.
        Returns RGBA array of shape (n_points, 4).
        """
        import matplotlib
        import colorsys
        n_points = self.embeddings.shape[0]
        colors = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other), dtype=float)

        n_archetypes = len(self.closest_indices_per_archetype)
        if n_archetypes == 0:
            return colors

        base_cmap = matplotlib.cm.get_cmap("viridis")
        base_colors = base_cmap(np.linspace(0, 1, n_archetypes))

        x_pct = self.percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        for j, idxs_j in enumerate(self.closest_indices_per_archetype):
            if len(idxs_j) == 0:
                continue
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            base_r, base_g, base_b, _ = base_colors[j]
            h, l, s = colorsys.rgb_to_hls(base_r, base_g, base_b)

            min_lightness = 0.3
            if l < min_lightness:
                l = min_lightness

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) == 0:
                    break

                fraction = (i_bin + 1) / (n_bins + 1)
                lightness_increment = fraction * 0.3
                new_l = min(l + lightness_increment, 1.0)

                shaded_r, shaded_g, shaded_b = colorsys.hls_to_rgb(h, new_l, s)
                colors[bin_points, :3] = (shaded_r, shaded_g, shaded_b)
                colors[bin_points, 3] = 1.0

        return colors

    # -------------------------------------------------------------------------
    #  Rastermap Controls and Plotting - continued
    # -------------------------------------------------------------------------
    # All Rastermap related methods are already included above.

    # -------------------------------------------------------------------------
    #  Additional Helper Methods
    # -------------------------------------------------------------------------
    # These methods are already included above.





if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = AnalysisGUI()
    main_window.show()
    sys.exit(app.exec_())

#%% WORKING 13/01/25

# TODO: fix so you can select archetype when attribute x bin

class LabeledSlider(QtWidgets.QWidget):
    def __init__(self, minimum, maximum, interval=1, parent=None):
        super().__init__(parent)

        levels = range(minimum, maximum + interval, interval)
        self.levels = list(zip(levels, map(str, levels)))

        self.left_margin = 10
        self.top_margin = 10
        self.right_margin = 10
        self.bottom_margin = 20  # Adjust bottom margin to fit labels

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(self.left_margin, self.top_margin, self.right_margin, self.bottom_margin)

        self.sl = QtWidgets.QSlider(Qt.Horizontal, self)
        self.sl.setMinimum(minimum)
        self.sl.setMaximum(maximum)
        self.sl.setValue(minimum)
        self.sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sl.setTickInterval(interval)
        self.sl.setSingleStep(1)
        self.layout.addWidget(self.sl)

    def paintEvent(self, e):
        super().paintEvent(e)

        style = self.sl.style()
        painter = QPainter(self)
        st_slider = QStyleOptionSlider()
        st_slider.initFrom(self.sl)
        st_slider.orientation = self.sl.orientation()

        length = style.pixelMetric(QStyle.PM_SliderLength, st_slider, self.sl)
        available = style.pixelMetric(QStyle.PM_SliderSpaceAvailable, st_slider, self.sl)

        for v, v_str in self.levels:
            rect = painter.drawText(QRect(), Qt.TextDontPrint, v_str)

            x_loc = QStyle.sliderPositionFromValue(self.sl.minimum(),
                                                   self.sl.maximum(),
                                                   v,
                                                   available) + length // 2

            left = x_loc - rect.width() // 2 + self.left_margin
            bottom = self.rect().bottom() - 10  # Adjust position of labels

            pos = QPoint(left, bottom)
            painter.drawText(pos, v_str)

class SliderWithText(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paintEvent(self, event):
        # Call the base class's paintEvent to render the slider
        super().paintEvent(event)
        
        # Create a QPainter object to draw the text
        painter = QPainter(self)

        # Set a readable font size
        painter.setFont(QFont("Arial", 10))

        # Create and initialize the style option
        option = QStyleOptionSlider()
        self.initStyleOption(option)

        # Get the position of the slider handle
        handle_rect = self.style().subControlRect(
            self.style().CC_Slider, option, self.style().SC_SliderHandle, self
        )
        handle_center = handle_rect.center()

        # Calculate percentage text
        percentage = self.value()
        text = f"{percentage}%"

        # Draw the text centered above the slider handle
        text_rect = painter.boundingRect(
            handle_center.x() - 20, handle_center.y() - 30, 40, 20, Qt.AlignCenter, text
        )
        painter.drawText(text_rect, Qt.AlignCenter, text)

        # End painting
        painter.end()
        
class SharedState(QObject):
    """Shared state for communicating between windows."""
    updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.selected_points = []

    def set_selected_points(self, points):
        """Update the selected points and notify all listeners."""
        self.selected_points = points
        self.updated.emit()


class SelectFromCollectionLasso:
    def __init__(self, ax, scatter, on_select_callback, is_3d=False):
        self.ax = ax
        self.scatter = scatter
        self.is_3d = is_3d
        self.on_select_callback = on_select_callback
        self.ind = set()
        self._initialize_data()
        self.lasso = None
        self.selector_enabled = False

    def _initialize_data(self):
        """Initialize scatter plot data."""
        if self.is_3d:
            # For 3D scatter, _offsets3d is a (3, N) tuple
            self.data = np.array(self.scatter._offsets3d).T
        else:
            # For 2D scatter, get_offsets() is shape (N, 2)
            self.data = self.scatter.get_offsets()

    def toggle_selection_mode(self):
        """Enable or disable lasso selection."""
        self.selector_enabled = not self.selector_enabled
        if self.selector_enabled:
            if self.is_3d:
                self.ax.disable_mouse_rotation()
            self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        else:
            if self.is_3d:
                self.ax.mouse_init()
            if self.lasso:
                self.lasso.disconnect_events()
                self.lasso = None

    def on_select(self, verts):
        """Handle lasso selection."""
        path = Path(verts)
        if self.is_3d:
            projected = self.project_3d_to_2d(self.data)
        else:
            projected = self.data
        selected_indices = np.nonzero(path.contains_points(projected))[0]
        self.ind = set(selected_indices)
        self.on_select_callback(list(self.ind))

    def project_3d_to_2d(self, points):
        """Project 3D points to 2D for selection logic."""
        trans = self.ax.get_proj()
        points_4d = np.column_stack((points, np.ones(points.shape[0])))
        projected = np.dot(points_4d, trans.T)
        projected[:, :2] /= projected[:, 3, None]
        return projected[:, :2]


class AnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.session = None
        self.viewers = []
        self.shared_state = SharedState()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Session Analysis GUI')

        window_width = 300  # Desired width
        window_height = 600  # Desired height
        self.resize(window_width, window_height)  # Adjust the size of the window
        
        # Center the window on the screen
        self.show()  # Ensure the window is initialized before measuring geometry
        frame_geometry = self.frameGeometry()  # Includes window decorations
        screen_geometry = QApplication.desktop().screenGeometry()
        
        # Calculate center
        x = (screen_geometry.width() - frame_geometry.width()) // 2 - (400 + 150 + 25)
        y = (screen_geometry.height() - frame_geometry.height()) // 2
        self.move(x, y)
        
        self.layout = QVBoxLayout()

        self.load_file_button = QPushButton('Load .pkl File', self)
        self.load_file_button.clicked.connect(self.load_pkl_file)
        self.layout.addWidget(self.load_file_button)

        self.file_label = QLabel('Selected File: None', self)
        self.layout.addWidget(self.file_label)

        self.region_label = QLabel('Select Brain Regions:')
        self.layout.addWidget(self.region_label)
        self.region_list = QListWidget(self)
        self.region_list.setSelectionMode(QListWidget.MultiSelection)
        self.layout.addWidget(self.region_list)

        self.embedding_label = QLabel('Select Embedding Method:')
        self.layout.addWidget(self.embedding_label)
        self.radio_group = QButtonGroup(self)

        self.pca_radio = QRadioButton('PCA', self)
        self.radio_group.addButton(self.pca_radio)
        self.layout.addWidget(self.pca_radio)

        self.fa_radio = QRadioButton('FA', self)
        self.radio_group.addButton(self.fa_radio)
        self.layout.addWidget(self.fa_radio)

        self.lem_radio = QRadioButton('Laplacian', self)
        self.radio_group.addButton(self.lem_radio)
        self.layout.addWidget(self.lem_radio)

        # Archetype Method
        self.archetype_label = QLabel('Select Archetype Method:')
        self.layout.addWidget(self.archetype_label)
        self.archetype_dropdown = QComboBox(self)
        self.archetype_dropdown.addItems(['None', 'ConvexHull', 'nfindr', 'SISAL', 'alpha_shape'])
        self.layout.addWidget(self.archetype_dropdown)
        
        self.analyse_button = QPushButton('Proceed with Analysis', self)
        self.analyse_button.clicked.connect(self.proceed_with_analysis)
        self.layout.addWidget(self.analyse_button)
    
        self.setLayout(self.layout)

    def load_pkl_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select .pkl File', '', 'Pickle Files (*.pkl)')
        if file_path:
            self.file_label.setText(f'Selected File: {file_path}')
            try:
                with open(file_path, 'rb') as f:
                    self.session = pkl.load(f)

                if hasattr(self.session, 'unit_channels') and 'structure_acronym' in self.session.unit_channels:
                    brain_regions = sorted(self.session.unit_channels['structure_acronym'].unique())
                    self.region_list.clear()
                    self.region_list.addItems(brain_regions)
                else:
                    QMessageBox.warning(self, 'Error', 'No brain region information found.')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load file: {e}')

    def proceed_with_analysis(self):
        if not self.session:
            QMessageBox.warning(self, 'Error', 'No session loaded.')
            return

        selected_regions = [item.text() for item in self.region_list.selectedItems()]
        if not selected_regions:
            QMessageBox.warning(self, 'Error', 'No brain regions selected.')
            return

        selected_method = None
        if self.pca_radio.isChecked():
            selected_method = 'PCA'
        elif self.fa_radio.isChecked():
            selected_method = 'FA'
        elif self.lem_radio.isChecked():
            selected_method = 'Laplacian'

        if not selected_method:
            QMessageBox.warning(self, 'Error', 'No embedding method selected.')
            return

        neuron_ids = self.session.unit_channels[self.session.unit_channels['structure_acronym'].isin(selected_regions)].index
        filtered_matrix = self.session.neuron_x_time_matrix.loc[neuron_ids]

        smoothed = gaussian_filter1d(filtered_matrix.values, sigma=3.5, axis=1)
        z_scored_data = StandardScaler().fit_transform(smoothed.T).T

        embeddings = None
        if selected_method == 'PCA':
            embeddings = PCA().fit_transform(z_scored_data.T)
        elif selected_method == 'FA':
            embeddings = FactorAnalysis().fit_transform(z_scored_data.T)
        elif selected_method == 'Laplacian':
            n_neighbors = max(1, int(z_scored_data.shape[1] * 0.005))
            adjacency_matrix = kneighbors_graph(z_scored_data.T, n_neighbors=n_neighbors,
                                                mode='connectivity', include_self=False).toarray()
            degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
            laplacian_rw = np.eye(adjacency_matrix.shape[0]) - np.linalg.inv(degree_matrix) @ adjacency_matrix
            _, eigenvectors = eigsh(laplacian_rw, k=4, which='SM')
            embeddings = eigenvectors[:, 1:]

        session_id = "Session_01"
        brain_regions = selected_regions
        algorithm = selected_method

        # We assume self.session.discrete_attributes includes all attributes
        # attrs = self.session.discrete_attributes
        attrs = self.session.continuous_attributes
        
        self.archetype_method = self.archetype_dropdown.currentText()
        viewer = EmbeddingWindow(embeddings, attrs,
                                 session_id, brain_regions, algorithm,
                                 self.shared_state, self.archetype_method)
        self.viewers.append(viewer)
        viewer.show()

class EmbeddingWindow(QWidget):
    def __init__(
        self,
        embeddings,
        attributes,
        session_id,
        brain_regions,
        algorithm,
        shared_state,
        archetype_method,
        alpha_other=0.3
    ):
        super().__init__()

        self.embeddings = embeddings
        self._scaled_embeddings = embeddings.copy()
        self.attributes = attributes  # DataFrame (n_points, n_attributes)
        self.session_id = session_id
        self.brain_regions = brain_regions
        self.algorithm = algorithm
        self.shared_state = shared_state
        self.archetype_method = archetype_method
        self.alpha_other = alpha_other
        self.selector = None
        self.lasso_active = False
        self.highlighted_indices = []

        # Caches for convex hull and SISAL (top usage)
        self.cached_convex_hull = None
        self.cached_sisal_hull = None

        # If embeddings are very small, scale them
        self._scale_embedding = False
        self._scale_factor = 100
        if np.max(self.embeddings) < 1:
            self._scaled_embeddings *= self._scale_factor
            self._scale_embedding = True

        # -- For the TOP logic --
        self.closest_indices_per_archetype = None    # used by top
        self.means_matrix = None                     # shape = (n_attrs, n_archetypes) [top usage]
        self.bin_matrix = None                       # shape = (n_attrs, n_bins) [top usage]
        self.last_valid_attribute = None             # for the top

        # -- For the BOTTOM-LEFT logic --
        self.last_valid_attribute_left = None
        self.closest_indices_per_archetype_left = None
        self.means_matrix_left = None
        self.bin_matrix_left = None

        # -- For the BOTTOM-RIGHT logic --
        self.last_valid_attribute_right = None
        self.closest_indices_per_archetype_right = None
        self.means_matrix_right = None
        self.bin_matrix_right = None

        # Connect to shared state for multi-window sync
        self.shared_state.updated.connect(self.sync_highlighted_points)

        self.initUI()

    # -------------------------------------------------------------------------
    #  INIT UI
    # -------------------------------------------------------------------------
    def initUI(self):
        self.setWindowTitle('3D Embedding Viewer with EXACT Schematic Layout')
        self.showMaximized()

        grid_layout = QGridLayout()
        self.setLayout(grid_layout)

        # 1) 3D figure
        self.figure_3d = Figure()
        self.canvas_3d = FigureCanvas(self.figure_3d)
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)

        threeD_layout = QVBoxLayout()
        threeD_layout.addWidget(self.canvas_3d)
        threeD_layout.addWidget(self.toolbar_3d)
        threeD_widget = QWidget()
        threeD_widget.setLayout(threeD_layout)
        grid_layout.addWidget(threeD_widget, 0, 0)

        # 2) 2D figure (XY, XZ, YZ)
        self.figure_2d = Figure()
        self.canvas_2d = FigureCanvas(self.figure_2d)
        twoD_layout = QVBoxLayout()
        twoD_layout.addWidget(self.canvas_2d)
        twoD_widget = QWidget()
        twoD_widget.setLayout(twoD_layout)
        grid_layout.addWidget(twoD_widget, 0, 1)

        # 3) Bottom-left heatmap
        self.figure_btm_left = Figure()
        self.canvas_btm_left = FigureCanvas(self.figure_btm_left)
        grid_layout.addWidget(self.canvas_btm_left, 1, 0)

        # 4) Bottom-right heatmap
        self.figure_btm_right = Figure()
        self.canvas_btm_right = FigureCanvas(self.figure_btm_right)
        grid_layout.addWidget(self.canvas_btm_right, 1, 1)

        # Expand columns equally
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        # --------------------------------------------------------------------
        # Right Column: Top + Bottom Controls
        # --------------------------------------------------------------------
        controls_column = QVBoxLayout()

        # ----- Top Controls (for the 3D & top-2D plots) -----
        top_controls_layout = QVBoxLayout()
        top_controls_label = QLabel("Controls (Top Plots):")
        top_controls_layout.addWidget(top_controls_label)

        # (A) attribute_dropdown
        self.attribute_dropdown = QComboBox(self)
        self.attribute_dropdown.addItems(
            [
                'None',
                'highlighted points',
                'points_closest_to_archetypes',
                'points_between_archetypes',
            ]
            + list(self.attributes.columns)
        )
        self.attribute_dropdown.currentIndexChanged.connect(self.update_plot)
        top_controls_layout.addWidget(self.attribute_dropdown)

        # (B) archetype_selection
        self.archetype_selection = QComboBox(self)
        self.archetype_selection.setVisible(False)
        self.archetype_selection.currentIndexChanged.connect(self.update_plot)
        top_controls_layout.addWidget(self.archetype_selection)

        # (C) Toggle Lasso
        self.toggle_button = QPushButton("Toggle Lasso Selection", self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.toggle_lasso_selection)
        top_controls_layout.addWidget(self.toggle_button)

        # (D) Save points
        self.save_button = QPushButton("Save Selected Points", self)
        self.save_button.clicked.connect(self.save_selected_points)
        top_controls_layout.addWidget(self.save_button)

        # (E) Top slider + label
        slider_row = QHBoxLayout()
        self.percentage_slider = QSlider(Qt.Horizontal, self)
        self.percentage_slider.setRange(1, 50)
        self.percentage_slider.setValue(5)
        self.percentage_slider.setTickPosition(QSlider.NoTicks)
        self.percentage_slider.setVisible(False)
        self.percentage_slider.valueChanged.connect(self.update_percentage)
        slider_row.addWidget(self.percentage_slider)

        self.slider_value_label = QLabel(f"{self.percentage_slider.value()}%", self)
        self.slider_value_label.setVisible(False)
        slider_row.addWidget(self.slider_value_label)
        top_controls_layout.addLayout(slider_row)

        # (F) Bin spinbox (top)
        bins_row = QHBoxLayout()
        self.bins_label = QLabel("Num Bins:")
        self.bins_label.setVisible(False)
        self.bin_spinbox = QSpinBox(self)
        self.bin_spinbox.setRange(1, 50)
        self.bin_spinbox.setValue(2)
        self.bin_spinbox.setVisible(False)
        self.bin_spinbox.valueChanged.connect(self.update_plot)
        bins_row.addWidget(self.bins_label)
        bins_row.addWidget(self.bin_spinbox)
        top_controls_layout.addLayout(bins_row)

        top_controls_layout.addStretch()
        top_controls_widget = QWidget()
        top_controls_widget.setLayout(top_controls_layout)
        controls_column.addWidget(top_controls_widget)

        # --------------------------------------------------------------------
        # Bottom Plot Axes & Controls
        # --------------------------------------------------------------------
        self.x_left_dropdown = QComboBox(self)
        self.x_left_dropdown.addItems(["attribute", "archetype", "bin"])
        self.x_left_dropdown.setCurrentText("attribute")
        self.x_left_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)

        self.y_left_dropdown = QComboBox(self)
        self.y_left_dropdown.addItems(["attribute", "archetype", "bin"])
        self.y_left_dropdown.setCurrentText("archetype")
        self.y_left_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)

        self.x_right_dropdown = QComboBox(self)
        self.x_right_dropdown.addItems(["attribute", "archetype", "bin"])
        self.x_right_dropdown.setCurrentText("bin")
        self.x_right_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)

        self.y_right_dropdown = QComboBox(self)
        self.y_right_dropdown.addItems(["attribute", "archetype", "bin"])
        self.y_right_dropdown.setCurrentText("attribute")
        self.y_right_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)

        bottom_controls_layout = QVBoxLayout()
        bottom_controls_label = QLabel("Controls (Bottom Plots):")
        bottom_controls_layout.addWidget(bottom_controls_label)

        # Left Plot Axes
        left_plot_label = QLabel("Left Plot Axes:")
        bottom_controls_layout.addWidget(left_plot_label)
        bottom_controls_layout.addWidget(self.x_left_dropdown)
        bottom_controls_layout.addWidget(self.y_left_dropdown)

        # Right Plot Axes
        right_plot_label = QLabel("Right Plot Axes:")
        bottom_controls_layout.addWidget(right_plot_label)
        bottom_controls_layout.addWidget(self.x_right_dropdown)
        bottom_controls_layout.addWidget(self.y_right_dropdown)

        # BOTTOM LEFT
        self.bottom_left_attr_dropdown = QComboBox(self)
        self.bottom_left_attr_dropdown.addItems(
            ["None", "highlighted points", "points_closest_to_archetypes", "points_between_archetypes"]
            + list(self.attributes.columns)
        )
        self.bottom_left_attr_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)
        bottom_controls_layout.addWidget(QLabel("Bottom-Left Color Attribute:"))
        bottom_controls_layout.addWidget(self.bottom_left_attr_dropdown)

        self.bottom_left_percentage_slider = QSlider(Qt.Horizontal, self)
        self.bottom_left_percentage_slider.setRange(1, 50)
        self.bottom_left_percentage_slider.setValue(5)
        self.bottom_left_percentage_slider.setTickPosition(QSlider.NoTicks)
        self.bottom_left_percentage_slider.valueChanged.connect(self.update_btm_left_heatmap)
        bottom_controls_layout.addWidget(self.bottom_left_percentage_slider)

        self.bottom_left_slider_label = QLabel("5%", self)
        bottom_controls_layout.addWidget(self.bottom_left_slider_label)

        self.bottom_left_bin_label = QLabel("Num Bins (Left):")
        bottom_controls_layout.addWidget(self.bottom_left_bin_label)

        self.bottom_left_bin_spinbox = QSpinBox(self)
        self.bottom_left_bin_spinbox.setRange(1, 50)
        self.bottom_left_bin_spinbox.setValue(2)
        self.bottom_left_bin_spinbox.valueChanged.connect(self.update_btm_left_heatmap)
        bottom_controls_layout.addWidget(self.bottom_left_bin_spinbox)

        # BOTTOM RIGHT
        self.bottom_right_attr_dropdown = QComboBox(self)
        self.bottom_right_attr_dropdown.addItems(
            ["None", "highlighted points", "points_closest_to_archetypes", "points_between_archetypes"]
            + list(self.attributes.columns)
        )
        self.bottom_right_attr_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)
        bottom_controls_layout.addWidget(QLabel("Bottom-Right Color Attribute:"))
        bottom_controls_layout.addWidget(self.bottom_right_attr_dropdown)

        self.bottom_right_percentage_slider = QSlider(Qt.Horizontal, self)
        self.bottom_right_percentage_slider.setRange(1, 50)
        self.bottom_right_percentage_slider.setValue(5)
        self.bottom_right_percentage_slider.setTickPosition(QSlider.NoTicks)
        self.bottom_right_percentage_slider.valueChanged.connect(self.update_btm_right_heatmap)
        bottom_controls_layout.addWidget(self.bottom_right_percentage_slider)

        self.bottom_right_slider_label = QLabel("5%", self)
        bottom_controls_layout.addWidget(self.bottom_right_slider_label)

        self.bottom_right_bin_label = QLabel("Num Bins (Right):")
        bottom_controls_layout.addWidget(self.bottom_right_bin_label)

        self.bottom_right_bin_spinbox = QSpinBox(self)
        self.bottom_right_bin_spinbox.setRange(1, 50)
        self.bottom_right_bin_spinbox.setValue(2)
        self.bottom_right_bin_spinbox.valueChanged.connect(self.update_btm_right_heatmap)
        bottom_controls_layout.addWidget(self.bottom_right_bin_spinbox)

        bottom_controls_layout.addStretch()
        bottom_controls_widget = QWidget()
        bottom_controls_widget.setLayout(bottom_controls_layout)
        controls_column.addWidget(bottom_controls_widget)

        # Finalize the right column
        controls_widget = QWidget()
        controls_widget.setLayout(controls_column)
        grid_layout.addWidget(controls_widget, 0, 2, 2, 1)

        # Uniform widths
        button_width = max(
            self.toggle_button.sizeHint().width(),
            self.save_button.sizeHint().width()
        )
        # Original top widgets:
        for w in [
            self.attribute_dropdown,
            self.archetype_selection,
            self.toggle_button,
            self.save_button,
            self.percentage_slider,
            self.slider_value_label,
            self.bin_spinbox,
            self.bins_label,
            self.x_left_dropdown,
            self.y_left_dropdown,
            self.x_right_dropdown,
            self.y_right_dropdown,
        ]:
            w.setFixedWidth(button_width)

        # Bottom-left set
        for w in [
            self.bottom_left_attr_dropdown,
            self.bottom_left_percentage_slider,
            self.bottom_left_slider_label,
            self.bottom_left_bin_label,
            self.bottom_left_bin_spinbox,
        ]:
            w.setFixedWidth(button_width)

        # Bottom-right set
        for w in [
            self.bottom_right_attr_dropdown,
            self.bottom_right_percentage_slider,
            self.bottom_right_slider_label,
            self.bottom_right_bin_label,
            self.bottom_right_bin_spinbox,
        ]:
            w.setFixedWidth(button_width)

        # Initial plot (top)
        self.plot_embedding()

    # -------------------------------------------------------------------------
    #  SYNCHRONIZE HIGHLIGHTED POINTS
    # -------------------------------------------------------------------------
    def sync_highlighted_points(self):
        self.highlighted_indices = self.shared_state.selected_points
        if self.attribute_dropdown.currentText() == 'highlighted points':
            self.plot_embedding(color_attribute='highlighted points')

    # -------------------------------------------------------------------------
    #  MASTER UPDATE (TOP PLOTS)
    # -------------------------------------------------------------------------
    def update_plot(self):
        selected_attr = self.attribute_dropdown.currentText()

        if selected_attr in self.attributes.columns:
            self.last_valid_attribute = selected_attr

        if self.last_valid_attribute is None and len(self.attributes.columns) > 0:
            self.last_valid_attribute = self.attributes.columns[0]

        if selected_attr in ["points_between_archetypes", "points_closest_to_archetypes"]:
            self.percentage_slider.setVisible(True)
            self.slider_value_label.setVisible(True)
            self.bin_spinbox.setVisible(True)
            self.bins_label.setVisible(True)

            if selected_attr == "points_between_archetypes":
                self.archetype_selection.setVisible(True)
                if not self.archetype_selection.count():
                    self.populate_archetype_selection()
            else:
                self.archetype_selection.setVisible(False)
        else:
            self.percentage_slider.setVisible(False)
            self.slider_value_label.setVisible(False)
            self.bin_spinbox.setVisible(False)
            self.bins_label.setVisible(False)
            self.archetype_selection.setVisible(False)

        # Re-plot (top)
        self.plot_embedding(color_attribute=selected_attr)

    def update_percentage(self):
        val = self.percentage_slider.value()
        self.slider_value_label.setText(f"{val}%")
        if not self.slider_value_label.isVisible():
            self.slider_value_label.setVisible(True)
        self.alpha_other = val / 100
        self.plot_embedding(color_attribute=self.attribute_dropdown.currentText())

    def populate_archetype_selection(self):
        self.ensure_archetypes_computed()
        if self.archetype_method == "ConvexHull" and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == "SISAL" and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            arch_points = None

        if arch_points is None:
            return

        self.archetype_selection.clear()
        pairs = []
        for i in range(len(arch_points)):
            for j in range(i + 1, len(arch_points)):
                pairs.append(f"{i + 1} - {j + 1}")
        self.archetype_selection.addItems(pairs)

    # -------------------------------------------------------------------------
    #  TOP PLOT + THEN calls bottom heatmap updates
    # -------------------------------------------------------------------------
    def plot_embedding(self, color_attribute=None):
        # Original 3D axis logic
        if not self.figure_3d.axes:
            self.figure_3d.clear()
            self.ax = self.figure_3d.add_subplot(111, projection='3d')
        else:
            self.ax = self.figure_3d.axes[0]

        n_points = self.embeddings.shape[0]
        colors = np.array(["k"] * n_points)

        # 1) "highlighted points"
        if color_attribute == "highlighted points":
            colors = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other))
            for idx in self.shared_state.selected_points:
                if idx < n_points:
                    colors[idx] = (1, 0, 0, 1)

        elif color_attribute == "points_closest_to_archetypes":
            pass

        elif color_attribute == "points_between_archetypes":
            pass

        elif color_attribute and color_attribute != "None":
            import matplotlib
            vals = self.attributes[color_attribute]
            norm = matplotlib.colors.Normalize(vals.min(), vals.max())
            colors = plt.cm.viridis_r(norm(vals))

        # Always compute top X% if recognized (for top usage)
        self.compute_top_xpct_indices()  # modifies self.closest_indices_per_archetype

        # Points_closest_to_archetypes (color logic)
        if color_attribute == "points_closest_to_archetypes" and self.closest_indices_per_archetype:
            colors = self.cumulative_bins_closest_archetypes()

        # Points_between_archetypes
        elif color_attribute == "points_between_archetypes":
            self.ensure_archetypes_computed()
            arch_points = None
            if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
                arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
            elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
                arch_points = self.cached_sisal_hull.T

            if arch_points is not None:
                selected_index = self.archetype_selection.currentIndex()
                if selected_index != -1:
                    pairs = [
                        (i, j)
                        for i in range(len(arch_points))
                        for j in range(i + 1, len(arch_points))
                    ]
                    if selected_index < len(pairs):
                        a, b = pairs[selected_index]
                        arch_a = arch_points[a]
                        arch_b = arch_points[b]
                        n_intermediate = self.bin_spinbox.value()
                        dist = np.linalg.norm(
                            self.embeddings[:, None, :3]
                            - np.linspace(arch_a, arch_b, n_intermediate)[None, :, :],
                            axis=2,
                        )
                        x_pct = self.percentage_slider.value()
                        n_closest = max(1, int((x_pct / 100) * n_points))
                        closest_indices = np.argsort(dist, axis=0)[:n_closest]
                        cmap = plt.get_cmap('viridis')
                        icolors = cmap(np.linspace(0, 1, n_intermediate))
                        colarray = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other), dtype=float)
                        for i_col, idxs_i in enumerate(closest_indices.T):
                            colarray[idxs_i, :3] = icolors[i_col, :3]
                            colarray[idxs_i, 3] = 1.0
                        colors = colarray

        # Recompute or update standard top-level heatmaps
        self.means_matrix = self.compute_attribute_archetype_means()
        self.bin_matrix = self.compute_bin_attribute_means()

        # Now update both bottom heatmaps (still the old approach for top)
        self.update_btm_left_heatmap()
        self.update_btm_right_heatmap()

        # Draw 3D scatter
        self.ax.clear()
        self.scatter = self.ax.scatter(
            self._scaled_embeddings[:, 0],
            self._scaled_embeddings[:, 1],
            self._scaled_embeddings[:, 2],
            c=colors, marker='.'
        )
        self.ax.set_title(
            f"Session: {self.session_id}\n"
            f"Regions: {', '.join(sorted(self.brain_regions))}\n"
            f"Algorithm: {self.algorithm}",
            fontsize=12
        )
        self.ax.set_xlabel('Dimension 1')
        self.ax.set_ylabel('Dimension 2')
        self.ax.set_zlabel('Dimension 3')

        # 2D XY, XZ, YZ
        self.figure_2d.clear()
        self.ax_xy = self.figure_2d.add_subplot(311, aspect='equal')
        self.ax_xz = self.figure_2d.add_subplot(312, aspect='equal')
        self.ax_yz = self.figure_2d.add_subplot(313, aspect='equal')

        self.scatter_xy = self.ax_xy.scatter(
            self.embeddings[:, 0], self.embeddings[:, 1], c=colors, marker='.'
        )
        self.scatter_xz = self.ax_xz.scatter(
            self.embeddings[:, 0], self.embeddings[:, 2], c=colors, marker='.'
        )
        self.scatter_yz = self.ax_yz.scatter(
            self.embeddings[:, 1], self.embeddings[:, 2], c=colors, marker='.'
        )

        # Axis buffer
        for ax_obj, (xx, yy) in zip(
            [self.ax_xy, self.ax_xz, self.ax_yz],
            [
                (self.embeddings[:, 0], self.embeddings[:, 1]),
                (self.embeddings[:, 0], self.embeddings[:, 2]),
                (self.embeddings[:, 1], self.embeddings[:, 2])
            ]
        ):
            x_min, x_max = xx.min(), xx.max()
            y_min, y_max = yy.min(), yy.max()
            buffer_x = (x_max - x_min) * 0.2
            buffer_y = (y_max - y_min) * 0.2
            ax_obj.set_xlim(x_min - buffer_x, x_max + buffer_x)
            ax_obj.set_ylim(y_min - buffer_y, y_max + buffer_y)
            ax_obj.set_aspect('equal', adjustable='datalim')

        self.ax_xy.set_title("1-2 Dim")
        self.ax_xz.set_title("1-3 Dim")
        self.ax_yz.set_title("2-3 Dim")

        for ax_obj in [self.ax_xy, self.ax_xz, self.ax_yz]:
            ax_obj.spines['top'].set_visible(False)
            ax_obj.spines['right'].set_visible(False)
            ax_obj.spines['left'].set_visible(False)
            ax_obj.spines['bottom'].set_visible(False)
            ax_obj.set_xticks([])
            ax_obj.set_yticks([])

        self.canvas_2d.draw_idle()
        self.canvas_3d.draw_idle()

        # If hull lines
        if self.archetype_method == 'ConvexHull':
            self.add_convex_hull(self.ax, fig_type='3d')
            self.add_convex_hull(self.ax_xy, fig_type='2d', projection=(0, 1))
            self.add_convex_hull(self.ax_xz, fig_type='2d', projection=(0, 2))
            self.add_convex_hull(self.ax_yz, fig_type='2d', projection=(1, 2))
        if self.archetype_method == 'SISAL':
            self.add_sisal_hull(self.ax, fig_type='3d')
            self.add_sisal_hull(self.ax_xy, fig_type='2d', projection=(0, 1))
            self.add_sisal_hull(self.ax_xz, fig_type='2d', projection=(0, 2))
            self.add_sisal_hull(self.ax_yz, fig_type='2d', projection=(1, 2))

        # Lasso reinit
        if hasattr(self, 'selector') and self.selector and self.selector.selector_enabled:
            self.selector.toggle_selection_mode()
        if hasattr(self, 'lasso_xy') and self.lasso_xy and self.lasso_xy.selector_enabled:
            self.lasso_xy.toggle_selection_mode()
        if hasattr(self, 'lasso_xz') and self.lasso_xz and self.lasso_xz.selector_enabled:
            self.lasso_xz.toggle_selection_mode()
        if hasattr(self, 'lasso_yz') and self.lasso_yz and self.lasso_yz.selector_enabled:
            self.lasso_yz.toggle_selection_mode()

        self.selector = SelectFromCollectionLasso(self.ax, self.scatter, self.on_select_3d, is_3d=True)
        self.lasso_xy = SelectFromCollectionLasso(self.ax_xy, self.scatter_xy, self.on_select_2d)
        self.lasso_xz = SelectFromCollectionLasso(self.ax_xz, self.scatter_xz, self.on_select_2d)
        self.lasso_yz = SelectFromCollectionLasso(self.ax_yz, self.scatter_yz, self.on_select_2d)
        if self.lasso_active:
            self.selector.toggle_selection_mode()
            self.lasso_xy.toggle_selection_mode()
            self.lasso_xz.toggle_selection_mode()
            self.lasso_yz.toggle_selection_mode()
        
        
    def compute_bin_attribute_means(self):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None
    
        arch_a = arch_points[0]
        arch_b = arch_points[1]
        n_intermediate = self.bin_spinbox.value()
        inter_pts = np.linspace(arch_a, arch_b, n_intermediate)
    
        n_points = self.embeddings.shape[0]
        x_pct = self.percentage_slider.value()
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :], axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]
    
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, n_intermediate), dtype=float)
        for bin_i in range(n_intermediate):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_attribute_archetype_means(self):
        if not self.closest_indices_per_archetype:
            return None
        n_archetypes = len(self.closest_indices_per_archetype)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M
    
   
    
    # -------------------------------------------------------------------------
    #  BOTTOM-LEFT HEATMAP
    # -------------------------------------------------------------------------
    def update_btm_left_heatmap(self):
        """Recompute & redraw ONLY the bottom-left heatmap using the bottom-left slider/spinbox."""
        selected_attr = self.bottom_left_attr_dropdown.currentText()
        if selected_attr in self.attributes.columns:
            self.last_valid_attribute_left = selected_attr
        elif not self.last_valid_attribute_left and len(self.attributes.columns) > 0:
            self.last_valid_attribute_left = self.attributes.columns[0]

        # Grab bottom-left slider/spinbox
        x_pct = self.bottom_left_percentage_slider.value()
        self.bottom_left_slider_label.setText(f"{x_pct}%")
        n_bins = self.bottom_left_bin_spinbox.value()

        # 1) Compute a new set of 'closest_indices_per_archetype_left' based on x_pct
        self.compute_bottom_left_xpct_indices(x_pct)

        # 2) Build means_matrix_left / bin_matrix_left from that
        self.means_matrix_left = self.compute_attribute_archetype_means_bottom_left()
        self.bin_matrix_left = self.compute_bin_attribute_means_bottom_left(x_pct, n_bins)

        # Now do the bottom-left pcolormesh
        dim_x = self.x_left_dropdown.currentText()
        dim_y = self.y_left_dropdown.currentText()

        self.figure_btm_left.clear()
        ax = self.figure_btm_left.add_subplot(111)
        ax.set_title(f"Bottom-Left Heatmap ({selected_attr})", fontsize=10)

        matrix, x_labels, y_labels = self.get_heatmap_data_bottom_left(dim_x, dim_y)
        if matrix is None:
            self.canvas_btm_left.draw_idle()
            return

        pcm = ax.pcolormesh(
            matrix, cmap="viridis_r", edgecolors="k", linewidth=1, shading="flat"
        )
        cbar = self.figure_btm_left.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Value", rotation=270, labelpad=15)

        ax.set_xticks(np.arange(len(y_labels)) + 0.5)
        ax.set_xticklabels(y_labels)
        ax.set_yticks(np.arange(len(x_labels)) + 0.5)
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(dim_y.capitalize())
        ax.set_ylabel(dim_x.capitalize())
        ax.invert_yaxis()

        self.canvas_btm_left.draw_idle()

    # -------------------------------------------------------------------------
    #  BOTTOM-RIGHT HEATMAP
    # -------------------------------------------------------------------------
    def update_btm_right_heatmap(self):
        """Recompute & redraw ONLY the bottom-right heatmap using the bottom-right slider/spinbox."""
        selected_attr = self.bottom_right_attr_dropdown.currentText()
        if selected_attr in self.attributes.columns:
            self.last_valid_attribute_right = selected_attr
        elif not self.last_valid_attribute_right and len(self.attributes.columns) > 0:
            self.last_valid_attribute_right = self.attributes.columns[0]

        # Grab bottom-right slider/spinbox
        x_pct = self.bottom_right_percentage_slider.value()
        self.bottom_right_slider_label.setText(f"{x_pct}%")
        n_bins = self.bottom_right_bin_spinbox.value()

        # 1) Compute a new set of 'closest_indices_per_archetype_right' based on x_pct
        self.compute_bottom_right_xpct_indices(x_pct)

        # 2) Build means_matrix_right / bin_matrix_right
        self.means_matrix_right = self.compute_attribute_archetype_means_bottom_right()
        self.bin_matrix_right = self.compute_bin_attribute_means_bottom_right(x_pct, n_bins)

        # Now do the bottom-right pcolormesh
        dim_x = self.x_right_dropdown.currentText()
        dim_y = self.y_right_dropdown.currentText()

        self.figure_btm_right.clear()
        ax = self.figure_btm_right.add_subplot(111)
        ax.set_title(f"Bottom-Right Heatmap ({selected_attr})", fontsize=10)

        matrix, x_labels, y_labels = self.get_heatmap_data_bottom_right(dim_x, dim_y)
        if matrix is None:
            self.canvas_btm_right.draw_idle()
            return

        pcm = ax.pcolormesh(
            matrix, cmap="viridis_r", edgecolors="k", linewidth=1, shading="flat"
        )
        cbar = self.figure_btm_right.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Value", rotation=270, labelpad=15)

        ax.set_xticks(np.arange(len(y_labels)) + 0.5)
        ax.set_xticklabels(y_labels)
        ax.set_yticks(np.arange(len(x_labels)) + 0.5)
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(dim_y.capitalize())
        ax.set_ylabel(dim_x.capitalize())
        ax.invert_yaxis()

        self.canvas_btm_right.draw_idle()

    # -------------------------------------------------------------------------
    #  BOTTOM LEFT: get_heatmap_data_bottom_left
    # -------------------------------------------------------------------------
    def get_heatmap_data_bottom_left(self, dim_x, dim_y):
        """
        Same concept as get_heatmap_data, but uses self.means_matrix_left / self.bin_matrix_left
        and self.last_valid_attribute_left instead of top's variables.
        """
        combos = {}

        # 1) attribute x archetype
        if self.means_matrix_left is not None:
            n_archetypes = self.means_matrix_left.shape[1]
            arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "archetype")] = (self.means_matrix_left, attrs, arch_labels)
            combos[("archetype", "attribute")] = (self.means_matrix_left.T, arch_labels, attrs)

        # 2) attribute x bin
        if self.bin_matrix_left is not None:
            n_bins = self.bin_matrix_left.shape[1]
            bin_labels = [f"Bin {i+1}" for i in range(n_bins)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "bin")] = (self.bin_matrix_left, attrs, bin_labels)
            combos[("bin", "attribute")] = (self.bin_matrix_left.T, bin_labels, attrs)

        # 3) If user chooses (archetype, bin) or (bin, archetype), do bottom-left cumulative approach
        if (dim_x, dim_y) in [("archetype", "bin"), ("bin", "archetype")]:
            chosen_attr = self.last_valid_attribute_left
            if not chosen_attr or chosen_attr not in self.attributes.columns:
                chosen_attr = self.attributes.columns[0]  # fallback
            A = self.compute_archetype_bin_attribute_means_cumulative_bottom_left(chosen_attr)
            if A is not None:
                n_archetypes = A.shape[0]
                n_bins = A.shape[1]
                arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
                bin_labels = [f"Bin {j+1}" for j in range(n_bins)]
                if (dim_x, dim_y) == ("archetype", "bin"):
                    return A, arch_labels, bin_labels
                else:
                    return A.T, bin_labels, arch_labels
            return None, None, None

        return combos.get((dim_x, dim_y), (None, None, None))

    # -------------------------------------------------------------------------
    #  BOTTOM RIGHT: get_heatmap_data_bottom_right
    # -------------------------------------------------------------------------
    def get_heatmap_data_bottom_right(self, dim_x, dim_y):
        """
        Same concept as get_heatmap_data_bottom_left, but for right side.
        """
        combos = {}

        if self.means_matrix_right is not None:
            n_archetypes = self.means_matrix_right.shape[1]
            arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "archetype")] = (self.means_matrix_right, attrs, arch_labels)
            combos[("archetype", "attribute")] = (self.means_matrix_right.T, arch_labels, attrs)

        if self.bin_matrix_right is not None:
            n_bins = self.bin_matrix_right.shape[1]
            bin_labels = [f"Bin {i+1}" for i in range(n_bins)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "bin")] = (self.bin_matrix_right, attrs, bin_labels)
            combos[("bin", "attribute")] = (self.bin_matrix_right.T, bin_labels, attrs)

        if (dim_x, dim_y) in [("archetype", "bin"), ("bin", "archetype")]:
            chosen_attr = self.last_valid_attribute_right
            if not chosen_attr or chosen_attr not in self.attributes.columns:
                chosen_attr = self.attributes.columns[0]
            A = self.compute_archetype_bin_attribute_means_cumulative_bottom_right(chosen_attr)
            if A is not None:
                n_archetypes = A.shape[0]
                n_bins = A.shape[1]
                arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
                bin_labels = [f"Bin {j+1}" for j in range(n_bins)]
                if (dim_x, dim_y) == ("archetype", "bin"):
                    return A, arch_labels, bin_labels
                else:
                    return A.T, bin_labels, arch_labels
            return None, None, None

        return combos.get((dim_x, dim_y), (None, None, None))

    # -------------------------------------------------------------------------
    #  BOTTOM LEFT: separate compute logic
    # -------------------------------------------------------------------------
    def compute_bottom_left_xpct_indices(self, x_pct):
        """
        We'll define a new 'closest_indices_per_archetype_left' using x_pct
        and ignoring self.percentage_slider from the top.
        """
        self.closest_indices_per_archetype_left = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        arch_pts_3d = None
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        if arch_pts_3d is None or len(arch_pts_3d) == 0:
            return

        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )  # shape (n_points, n_archetypes)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        n_archetypes = arch_pts_3d.shape[0]
        self.closest_indices_per_archetype_left = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def compute_attribute_archetype_means_bottom_left(self):
        if not self.closest_indices_per_archetype_left:
            return None
        n_archetypes = len(self.closest_indices_per_archetype_left)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype_left):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        # row-wise normalize
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M

    def compute_bin_attribute_means_bottom_left(self, x_pct, n_bins):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None

        arch_a = arch_points[0]
        arch_b = arch_points[1]
        inter_pts = np.linspace(arch_a, arch_b, n_bins)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :], axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, n_bins), dtype=float)
        for bin_i in range(n_bins):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_archetype_bin_attribute_means_cumulative_bottom_left(self, chosen_attr):
        if self.closest_indices_per_archetype_left is None:
            return None
        if chosen_attr not in self.attributes.columns:
            return None

        n_archetypes = len(self.closest_indices_per_archetype_left)
        if n_archetypes == 0:
            return None

        n_points = self.embeddings.shape[0]
        x_pct = self.bottom_left_percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bottom_left_bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        M = np.full((n_archetypes, n_bins), np.nan, dtype=float)

        for j in range(n_archetypes):
            idxs_j = self.closest_indices_per_archetype_left[j]
            if len(idxs_j) == 0:
                continue
            # Distances for shading logic
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) > 0:
                    val = self.attributes.iloc[bin_points][chosen_attr].mean()
                    M[j, i_bin] = val
        return M

    # -------------------------------------------------------------------------
    #  BOTTOM RIGHT: separate compute logic
    # -------------------------------------------------------------------------
    def compute_bottom_right_xpct_indices(self, x_pct):
        self.closest_indices_per_archetype_right = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        arch_pts_3d = None
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        if arch_pts_3d is None or len(arch_pts_3d) == 0:
            return

        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )
        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        n_archetypes = arch_pts_3d.shape[0]
        self.closest_indices_per_archetype_right = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def compute_attribute_archetype_means_bottom_right(self):
        if not self.closest_indices_per_archetype_right:
            return None
        n_archetypes = len(self.closest_indices_per_archetype_right)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype_right):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M

    def compute_bin_attribute_means_bottom_right(self, x_pct, n_bins):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None

        arch_a = arch_points[0]
        arch_b = arch_points[1]
        inter_pts = np.linspace(arch_a, arch_b, n_bins)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :], axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, n_bins), dtype=float)
        for bin_i in range(n_bins):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_archetype_bin_attribute_means_cumulative_bottom_right(self, chosen_attr):
        if self.closest_indices_per_archetype_right is None:
            return None
        if chosen_attr not in self.attributes.columns:
            return None

        n_archetypes = len(self.closest_indices_per_archetype_right)
        if n_archetypes == 0:
            return None

        n_points = self.embeddings.shape[0]
        x_pct = self.bottom_right_percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bottom_right_bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        M = np.full((n_archetypes, n_bins), np.nan, dtype=float)

        for j in range(n_archetypes):
            idxs_j = self.closest_indices_per_archetype_right[j]
            if len(idxs_j) == 0:
                continue
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) > 0:
                    val = self.attributes.iloc[bin_points][chosen_attr].mean()
                    M[j, i_bin] = val
        return M

    # -------------------------------------------------------------------------
    #  ensure_archetypes_computed, hull drawing, lasso, etc. remain the same
    # -------------------------------------------------------------------------
    def ensure_archetypes_computed(self):
        if self.archetype_method == "ConvexHull" and self.cached_convex_hull is None:
            self.cached_convex_hull = ConvexHull(self.embeddings[:, :3])
        elif self.archetype_method == "SISAL" and self.cached_sisal_hull is None:
            ArchsMin, _ = findMinSimplex(10, self.embeddings, 1, 4)
            self.cached_sisal_hull = ArchsMin[:, np.argsort(ArchsMin[0, :])]

    # -------------------------------------------------------------------------
    #  CONVEX / SISAL hulls
    # -------------------------------------------------------------------------
    def add_convex_hull(self, ax, fig_type='3d', projection=None):
        try:
            if self.cached_convex_hull is None:
                self.cached_convex_hull = ConvexHull(self.embeddings[:, :3])
            hull = self.cached_convex_hull
            if fig_type == '3d':
                vertices_3d = self.embeddings[hull.vertices, :3]
                edges = []
                for simplex in hull.simplices:
                    edges.append(self.embeddings[simplex, :3])
                edge_collection = Line3DCollection(edges, colors='k', linewidths=0.5)
                ax.add_collection3d(edge_collection)
                ax.scatter(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2],
                           c='b', s=10, label='Archetypes')
                for i, (x, y, z) in enumerate(vertices_3d):
                    ax.text(x, y, z, str(i + 1), color='red', fontsize=10)
            elif fig_type == '2d':
                if projection is None:
                    return
                vertices_2d = self.embeddings[hull.vertices, projection]
                edges = []
                for simplex in hull.simplices:
                    edge = self.embeddings[simplex, projection]
                    edges.append(edge)
                    ax.plot(*edge.T, c='k', linewidth=0.5)
                ax.scatter(vertices_2d[:, 0], vertices_2d[:, 1],
                           c='b', s=10, label='Archetypes', zorder=5)
                for i, (xx, yy) in enumerate(vertices_2d):
                    ax.text(xx, yy, str(i + 1), color='red', fontsize=10, zorder=6)
                self.adjust_axis_limits(ax, self.embeddings[:, projection], vertices_2d)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error in adding ConvexHull: {e}')

    def add_sisal_hull(self, ax, fig_type='3d', projection=None):
        try:
            if self.cached_sisal_hull is None:
                ArchsMin, VolArchReal = findMinSimplex(10, self.embeddings, 1, 4)
                ArchsOrder = np.argsort(ArchsMin[0, :])
                self.cached_sisal_hull = ArchsMin[:, ArchsOrder]
            ArchsMin = self.cached_sisal_hull
            NArchetypes = ArchsMin.shape[1]
            if fig_type == '3d':
                ArchsMin_3d = ArchsMin[:3, :]
                edges = []
                for i in range(NArchetypes):
                    for j in range(i + 1, NArchetypes):
                        edge = np.array([ArchsMin_3d[:, i], ArchsMin_3d[:, j]])
                        edges.append(edge)
                edge_collection = Line3DCollection(edges, colors='k', linewidths=0.5)
                ax.add_collection3d(edge_collection)
                ax.scatter(ArchsMin_3d[0, :], ArchsMin_3d[1, :], ArchsMin_3d[2, :],
                           c='b', s=10, label='Archetypes')
                for i, (x, y, z) in enumerate(ArchsMin_3d.T):
                    ax.text(x, y, z, str(i + 1), color='red', fontsize=10)
            elif fig_type == '2d':
                if projection is None:
                    return
                ArchsMin_2d = ArchsMin[list(projection), :]
                edges = []
                for i in range(NArchetypes):
                    for j in range(i + 1, NArchetypes):
                        edge = np.array([ArchsMin_2d[:, i], ArchsMin_2d[:, j]])
                        edges.append(edge)
                        ax.plot(*edge.T, c='k', linewidth=0.5)
                ax.scatter(ArchsMin_2d[0, :], ArchsMin_2d[1, :],
                           c='b', s=10, label='Archetypes', zorder=5)
                for i, (xx, yy) in enumerate(ArchsMin_2d.T):
                    ax.text(xx, yy, str(i + 1), color='red', fontsize=10, zorder=6)
                scatter_points = self.embeddings[:, projection]
                self.adjust_axis_limits(ax, scatter_points, ArchsMin_2d.T)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error in adding SISAL hull: {e}')

    def adjust_axis_limits(self, ax, scatter_points, polytope_points):
        all_points = np.vstack([scatter_points, polytope_points])
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        buffer_x = (x_max - x_min) * 0.1
        buffer_y = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
        ax.set_ylim(y_min - buffer_y, y_max + buffer_y)

    def invalidate_cache(self):
        self.cached_convex_hull = None
        self.cached_sisal_hull = None

    # -------------------------------------------------------------------------
    #  LASSO
    # -------------------------------------------------------------------------
    def on_select_3d(self, indices):
        self.highlighted_indices = list(indices)

    def on_select_2d(self, indices):
        self.highlighted_indices = list(indices)

    def toggle_lasso_selection(self):
        if self.lasso_active:
            self.lasso_active = False
            if self.selector and self.selector.selector_enabled:
                self.selector.toggle_selection_mode()
            if self.lasso_xy and self.lasso_xy.selector_enabled:
                self.lasso_xy.toggle_selection_mode()
            if self.lasso_xz and self.lasso_xz.selector_enabled:
                self.lasso_xz.toggle_selection_mode()
            if self.lasso_yz and self.lasso_yz.selector_enabled:
                self.lasso_yz.toggle_selection_mode()
            self.canvas_3d.draw_idle()
            self.canvas_2d.draw_idle()
        else:
            self.lasso_active = True
            if self.selector:
                self.selector.toggle_selection_mode()
            if self.lasso_xy:
                self.lasso_xy.toggle_selection_mode()
            if self.lasso_xz:
                self.lasso_xz.toggle_selection_mode()
            if self.lasso_yz:
                self.lasso_yz.toggle_selection_mode()

    def save_selected_points(self):
        self.selected_points = list(self.highlighted_indices)
        self.shared_state.set_selected_points(self.selected_points)
        print(f"Selected points saved: {self.selected_points}")
        if self.attribute_dropdown.currentText() == 'highlighted points':
            self.plot_embedding(color_attribute='highlighted points')

    # -------------------------------------------------------------------------
    #  "points_closest_to_archetypes" color logic
    # -------------------------------------------------------------------------
    def compute_top_xpct_indices(self):
        """
        For each archetype j, define top X% by distance => closest_indices_per_archetype[j].
        We do this for potential usage in color or in the heatmaps.
        """
        self.closest_indices_per_archetype = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        else:
            return
        n_archetypes = arch_pts_3d.shape[0]
        if n_archetypes == 0:
            return
        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )  # shape (n_points, n_archetypes)

        x_pct = self.percentage_slider.value()
        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        self.closest_indices_per_archetype = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def cumulative_bins_closest_archetypes(self):
        """
        This sets the scatter color per chunk for each archetype in 'points_closest_to_archetypes'.
        Returns RGBA array of shape (n_points, 4).
        """
        import matplotlib
        import colorsys
        n_points = self.embeddings.shape[0]
        colors = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other), dtype=float)

        n_archetypes = len(self.closest_indices_per_archetype)
        if n_archetypes == 0:
            return colors

        base_cmap = matplotlib.colormaps.get_cmap("viridis")
        base_colors = base_cmap(np.linspace(0, 1, n_archetypes))

        x_pct = self.percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        for j, idxs_j in enumerate(self.closest_indices_per_archetype):
            if len(idxs_j) == 0:
                continue
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            base_r, base_g, base_b, _ = base_colors[j]
            h, l, s = colorsys.rgb_to_hls(base_r, base_g, base_b)

            min_lightness = 0.3
            if l < min_lightness:
                l = min_lightness

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) == 0:
                    break

                fraction = (i_bin + 1) / (n_bins + 1)
                lightness_increment = fraction * 0.3
                new_l = min(l + lightness_increment, 1.0)

                shaded_r, shaded_g, shaded_b = colorsys.hls_to_rgb(h, new_l, s)
                colors[bin_points, :3] = (shaded_r, shaded_g, shaded_b)
                colors[bin_points, 3] = 1.0

        return colors

    def get_archetype_coords(self, j):
        """Return the j-th archetype's 3D coords from either convex hull or SISAL."""
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull is not None:
            arch_points_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
            if j < len(arch_points_3d):
                return arch_points_3d[j]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points_3d = self.cached_sisal_hull.T[:, :3]
            if j < len(arch_points_3d):
                return arch_points_3d[j]
        return np.array([0, 0, 0], dtype=float)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = AnalysisGUI()
    main_window.show()
    sys.exit(app.exec_())

#%%

class LabeledSlider(QtWidgets.QWidget):
    def __init__(self, minimum, maximum, interval=1, parent=None):
        super().__init__(parent)

        levels = range(minimum, maximum + interval, interval)
        self.levels = list(zip(levels, map(str, levels)))

        self.left_margin = 10
        self.top_margin = 10
        self.right_margin = 10
        self.bottom_margin = 20  # Adjust bottom margin to fit labels

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(self.left_margin, self.top_margin, self.right_margin, self.bottom_margin)

        self.sl = QtWidgets.QSlider(Qt.Horizontal, self)
        self.sl.setMinimum(minimum)
        self.sl.setMaximum(maximum)
        self.sl.setValue(minimum)
        self.sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sl.setTickInterval(interval)
        self.sl.setSingleStep(1)
        self.layout.addWidget(self.sl)

    def paintEvent(self, e):
        super().paintEvent(e)

        style = self.sl.style()
        painter = QPainter(self)
        st_slider = QStyleOptionSlider()
        st_slider.initFrom(self.sl)
        st_slider.orientation = self.sl.orientation()

        length = style.pixelMetric(QStyle.PM_SliderLength, st_slider, self.sl)
        available = style.pixelMetric(QStyle.PM_SliderSpaceAvailable, st_slider, self.sl)

        for v, v_str in self.levels:
            rect = painter.drawText(QRect(), Qt.TextDontPrint, v_str)

            x_loc = QStyle.sliderPositionFromValue(self.sl.minimum(),
                                                   self.sl.maximum(),
                                                   v,
                                                   available) + length // 2

            left = x_loc - rect.width() // 2 + self.left_margin
            bottom = self.rect().bottom() - 10  # Adjust position of labels

            pos = QPoint(left, bottom)
            painter.drawText(pos, v_str)

class SliderWithText(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paintEvent(self, event):
        # Call the base class's paintEvent to render the slider
        super().paintEvent(event)
        
        # Create a QPainter object to draw the text
        painter = QPainter(self)

        # Set a readable font size
        painter.setFont(QFont("Arial", 10))

        # Create and initialize the style option
        option = QStyleOptionSlider()
        self.initStyleOption(option)

        # Get the position of the slider handle
        handle_rect = self.style().subControlRect(
            self.style().CC_Slider, option, self.style().SC_SliderHandle, self
        )
        handle_center = handle_rect.center()

        # Calculate percentage text
        percentage = self.value()
        text = f"{percentage}%"

        # Draw the text centered above the slider handle
        text_rect = painter.boundingRect(
            handle_center.x() - 20, handle_center.y() - 30, 40, 20, Qt.AlignCenter, text
        )
        painter.drawText(text_rect, Qt.AlignCenter, text)

        # End painting
        painter.end()
        
class SharedState(QObject):
    """Shared state for communicating between windows."""
    updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.selected_points = []

    def set_selected_points(self, points):
        """Update the selected points and notify all listeners."""
        self.selected_points = points
        self.updated.emit()


class SelectFromCollectionLasso:
    def __init__(self, ax, scatter, on_select_callback, is_3d=False):
        self.ax = ax
        self.scatter = scatter
        self.is_3d = is_3d
        self.on_select_callback = on_select_callback
        self.ind = set()
        self._initialize_data()
        self.lasso = None
        self.selector_enabled = False

    def _initialize_data(self):
        """Initialize scatter plot data."""
        if self.is_3d:
            # For 3D scatter, _offsets3d is a (3, N) tuple
            self.data = np.array(self.scatter._offsets3d).T
        else:
            # For 2D scatter, get_offsets() is shape (N, 2)
            self.data = self.scatter.get_offsets()

    def toggle_selection_mode(self):
        """Enable or disable lasso selection."""
        self.selector_enabled = not self.selector_enabled
        if self.selector_enabled:
            if self.is_3d:
                self.ax.disable_mouse_rotation()
            self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        else:
            if self.is_3d:
                self.ax.mouse_init()
            if self.lasso:
                self.lasso.disconnect_events()
                self.lasso = None

    def on_select(self, verts):
        """Handle lasso selection."""
        path = Path(verts)
        if self.is_3d:
            projected = self.project_3d_to_2d(self.data)
        else:
            projected = self.data
        selected_indices = np.nonzero(path.contains_points(projected))[0]
        self.ind = set(selected_indices)
        self.on_select_callback(list(self.ind))

    def project_3d_to_2d(self, points):
        """Project 3D points to 2D for selection logic."""
        trans = self.ax.get_proj()
        points_4d = np.column_stack((points, np.ones(points.shape[0])))
        projected = np.dot(points_4d, trans.T)
        projected[:, :2] /= projected[:, 3, None]
        return projected[:, :2]


class AnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.session = None
        self.viewers = []
        self.shared_state = SharedState()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Session Analysis GUI')

        window_width = 300  # Desired width
        window_height = 600  # Desired height
        self.resize(window_width, window_height)  # Adjust the size of the window
        
        # Center the window on the screen
        self.show()  # Ensure the window is initialized before measuring geometry
        frame_geometry = self.frameGeometry()  # Includes window decorations
        screen_geometry = QApplication.desktop().screenGeometry()
        
        # Calculate center
        x = (screen_geometry.width() - frame_geometry.width()) // 2 - (400 + 150 + 25)
        y = (screen_geometry.height() - frame_geometry.height()) // 2
        self.move(x, y)
        
        self.layout = QVBoxLayout()

        self.load_file_button = QPushButton('Load .pkl File', self)
        self.load_file_button.clicked.connect(self.load_pkl_file)
        self.layout.addWidget(self.load_file_button)

        self.file_label = QLabel('Selected File: None', self)
        self.layout.addWidget(self.file_label)

        self.region_label = QLabel('Select Brain Regions:')
        self.layout.addWidget(self.region_label)
        self.region_list = QListWidget(self)
        self.region_list.setSelectionMode(QListWidget.MultiSelection)
        self.layout.addWidget(self.region_list)

        self.embedding_label = QLabel('Select Embedding Method:')
        self.layout.addWidget(self.embedding_label)
        self.radio_group = QButtonGroup(self)

        self.pca_radio = QRadioButton('PCA', self)
        self.radio_group.addButton(self.pca_radio)
        self.layout.addWidget(self.pca_radio)

        self.fa_radio = QRadioButton('FA', self)
        self.radio_group.addButton(self.fa_radio)
        self.layout.addWidget(self.fa_radio)

        self.lem_radio = QRadioButton('Laplacian', self)
        self.radio_group.addButton(self.lem_radio)
        self.layout.addWidget(self.lem_radio)

        # Archetype Method
        self.archetype_label = QLabel('Select Archetype Method:')
        self.layout.addWidget(self.archetype_label)
        self.archetype_dropdown = QComboBox(self)
        self.archetype_dropdown.addItems(['None', 'ConvexHull', 'nfindr', 'SISAL', 'alpha_shape'])
        self.layout.addWidget(self.archetype_dropdown)
        
        self.analyse_button = QPushButton('Proceed with Analysis', self)
        self.analyse_button.clicked.connect(self.proceed_with_analysis)
        self.layout.addWidget(self.analyse_button)
    
        self.setLayout(self.layout)

    def load_pkl_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select .pkl File', '', 'Pickle Files (*.pkl)')
        if file_path:
            self.file_label.setText(f'Selected File: {file_path}')
            try:
                with open(file_path, 'rb') as f:
                    self.session = pkl.load(f)

                if hasattr(self.session, 'unit_channels') and 'structure_acronym' in self.session.unit_channels:
                    brain_regions = sorted(self.session.unit_channels['structure_acronym'].unique())
                    self.region_list.clear()
                    self.region_list.addItems(brain_regions)
                else:
                    QMessageBox.warning(self, 'Error', 'No brain region information found.')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load file: {e}')

    def proceed_with_analysis(self):
        if not self.session:
            QMessageBox.warning(self, 'Error', 'No session loaded.')
            return

        selected_regions = [item.text() for item in self.region_list.selectedItems()]
        if not selected_regions:
            QMessageBox.warning(self, 'Error', 'No brain regions selected.')
            return

        selected_method = None
        if self.pca_radio.isChecked():
            selected_method = 'PCA'
        elif self.fa_radio.isChecked():
            selected_method = 'FA'
        elif self.lem_radio.isChecked():
            selected_method = 'Laplacian'

        if not selected_method:
            QMessageBox.warning(self, 'Error', 'No embedding method selected.')
            return

        neuron_ids = self.session.unit_channels[self.session.unit_channels['structure_acronym'].isin(selected_regions)].index
        filtered_matrix = self.session.neuron_x_time_matrix.loc[neuron_ids]

        smoothed = gaussian_filter1d(filtered_matrix.values, sigma=3.5, axis=1)
        z_scored_data = StandardScaler().fit_transform(smoothed.T).T

        embeddings = None
        if selected_method == 'PCA':
            embeddings = PCA().fit_transform(z_scored_data.T)
        elif selected_method == 'FA':
            embeddings = FactorAnalysis().fit_transform(z_scored_data.T)
        elif selected_method == 'Laplacian':
            n_neighbors = max(1, int(z_scored_data.shape[1] * 0.005))
            adjacency_matrix = kneighbors_graph(z_scored_data.T, n_neighbors=n_neighbors,
                                                mode='connectivity', include_self=False).toarray()
            degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
            laplacian_rw = np.eye(adjacency_matrix.shape[0]) - np.linalg.inv(degree_matrix) @ adjacency_matrix
            _, eigenvectors = eigsh(laplacian_rw, k=4, which='SM')
            embeddings = eigenvectors[:, 1:]

        session_id = "Session_01"
        brain_regions = selected_regions
        algorithm = selected_method

        # We assume self.session.discrete_attributes includes all attributes
        # attrs = self.session.discrete_attributes
        attrs = self.session.continuous_attributes
        
        self.archetype_method = self.archetype_dropdown.currentText()
        viewer = EmbeddingWindow(embeddings, attrs,
                                 session_id, brain_regions, algorithm,
                                 self.shared_state, self.archetype_method)
        self.viewers.append(viewer)
        viewer.show()

class EmbeddingWindow(QWidget):
    def __init__(
        self,
        embeddings,
        attributes,
        session_id,
        brain_regions,
        algorithm,
        shared_state,
        archetype_method,
        alpha_other=0.3
    ):
        super().__init__()

        self.embeddings = embeddings
        self._scaled_embeddings = embeddings.copy()
        self.attributes = attributes  # DataFrame (n_points, n_attributes)
        self.session_id = session_id
        self.brain_regions = brain_regions
        self.algorithm = algorithm
        self.shared_state = shared_state
        self.archetype_method = archetype_method
        self.alpha_other = alpha_other
        self.selector = None
        self.lasso_active = False
        self.highlighted_indices = []

        # Caches for convex hull and SISAL (top usage)
        self.cached_convex_hull = None
        self.cached_sisal_hull = None

        # If embeddings are very small, scale them
        self._scale_embedding = False
        self._scale_factor = 100
        if np.max(self.embeddings) < 1:
            self._scaled_embeddings *= self._scale_factor
            self._scale_embedding = True

        # -- For the TOP logic --
        self.closest_indices_per_archetype = None    # used by top
        self.means_matrix = None                     # shape = (n_attrs, n_archetypes) [top usage]
        self.bin_matrix = None                       # shape = (n_attrs, n_bins) [top usage]
        self.last_valid_attribute = None             # for the top

        # -- For the BOTTOM-LEFT logic --
        self.last_valid_attribute_left = None
        self.closest_indices_per_archetype_left = None
        self.means_matrix_left = None
        self.bin_matrix_left = None

        # -- For the BOTTOM-RIGHT logic --
        self.last_valid_attribute_right = None
        self.closest_indices_per_archetype_right = None
        self.means_matrix_right = None
        self.bin_matrix_right = None

        # Connect to shared state for multi-window sync
        self.shared_state.updated.connect(self.sync_highlighted_points)

        self.initUI()

    # -------------------------------------------------------------------------
    #  INIT UI
    # -------------------------------------------------------------------------
    def initUI(self):
        self.setWindowTitle('3D Embedding Viewer with EXACT Schematic Layout')
        self.showMaximized()

        grid_layout = QGridLayout()
        self.setLayout(grid_layout)

        # 1) 3D figure
        self.figure_3d = Figure()
        self.canvas_3d = FigureCanvas(self.figure_3d)
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)

        threeD_layout = QVBoxLayout()
        threeD_layout.addWidget(self.canvas_3d)
        threeD_layout.addWidget(self.toolbar_3d)
        threeD_widget = QWidget()
        threeD_widget.setLayout(threeD_layout)
        grid_layout.addWidget(threeD_widget, 0, 0)

        # 2) 2D figure (XY, XZ, YZ)
        self.figure_2d = Figure()
        self.canvas_2d = FigureCanvas(self.figure_2d)
        twoD_layout = QVBoxLayout()
        twoD_layout.addWidget(self.canvas_2d)
        twoD_widget = QWidget()
        twoD_widget.setLayout(twoD_layout)
        grid_layout.addWidget(twoD_widget, 0, 1)

        # 3) Bottom-left heatmap
        self.figure_btm_left = Figure()
        self.canvas_btm_left = FigureCanvas(self.figure_btm_left)
        grid_layout.addWidget(self.canvas_btm_left, 1, 0)

        # 4) Bottom-right heatmap
        self.figure_btm_right = Figure()
        self.canvas_btm_right = FigureCanvas(self.figure_btm_right)
        grid_layout.addWidget(self.canvas_btm_right, 1, 1)

        # Expand columns equally
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        # --------------------------------------------------------------------
        # Right Column: Top + Bottom Controls
        # --------------------------------------------------------------------
        controls_column = QVBoxLayout()

        # ----- Top Controls (for the 3D & top-2D plots) -----
        top_controls_layout = QVBoxLayout()
        top_controls_label = QLabel("Controls (Top Plots):")
        top_controls_layout.addWidget(top_controls_label)

        # (A) attribute_dropdown
        self.attribute_dropdown = QComboBox(self)
        self.attribute_dropdown.addItems(
            [
                'None',
                'highlighted points',
                'points_closest_to_archetypes',
                'points_between_archetypes',
            ]
            + list(self.attributes.columns)
        )
        self.attribute_dropdown.currentIndexChanged.connect(self.update_plot)
        top_controls_layout.addWidget(self.attribute_dropdown)

        # (B) archetype_selection
        self.archetype_selection = QComboBox(self)
        self.archetype_selection.setVisible(False)
        self.archetype_selection.currentIndexChanged.connect(self.update_plot)
        top_controls_layout.addWidget(self.archetype_selection)

        # (C) Toggle Lasso
        self.toggle_button = QPushButton("Toggle Lasso Selection", self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.toggle_lasso_selection)
        top_controls_layout.addWidget(self.toggle_button)

        # (D) Save points
        self.save_button = QPushButton("Save Selected Points", self)
        self.save_button.clicked.connect(self.save_selected_points)
        top_controls_layout.addWidget(self.save_button)

        # (E) Top slider + label
        slider_row = QHBoxLayout()
        self.percentage_slider = QSlider(Qt.Horizontal, self)
        self.percentage_slider.setRange(1, 50)
        self.percentage_slider.setValue(5)
        self.percentage_slider.setTickPosition(QSlider.NoTicks)
        self.percentage_slider.setVisible(False)
        self.percentage_slider.valueChanged.connect(self.update_percentage)
        slider_row.addWidget(self.percentage_slider)

        self.slider_value_label = QLabel(f"{self.percentage_slider.value()}%", self)
        self.slider_value_label.setVisible(False)
        slider_row.addWidget(self.slider_value_label)
        top_controls_layout.addLayout(slider_row)

        # (F) Bin spinbox (top)
        bins_row = QHBoxLayout()
        self.bins_label = QLabel("Num Bins:")
        self.bins_label.setVisible(False)
        self.bin_spinbox = QSpinBox(self)
        self.bin_spinbox.setRange(1, 50)
        self.bin_spinbox.setValue(2)
        self.bin_spinbox.setVisible(False)
        self.bin_spinbox.valueChanged.connect(self.update_plot)
        bins_row.addWidget(self.bins_label)
        bins_row.addWidget(self.bin_spinbox)
        top_controls_layout.addLayout(bins_row)

        top_controls_layout.addStretch()
        top_controls_widget = QWidget()
        top_controls_widget.setLayout(top_controls_layout)
        controls_column.addWidget(top_controls_widget)

        # --------------------------------------------------------------------
        # Bottom Plot Axes & Controls
        # --------------------------------------------------------------------
        self.x_left_dropdown = QComboBox(self)
        self.x_left_dropdown.addItems(["attribute", "archetype", "bin"])
        self.x_left_dropdown.setCurrentText("attribute")
        self.x_left_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)

        self.y_left_dropdown = QComboBox(self)
        self.y_left_dropdown.addItems(["attribute", "archetype", "bin"])
        self.y_left_dropdown.setCurrentText("archetype")
        self.y_left_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)

        self.x_right_dropdown = QComboBox(self)
        self.x_right_dropdown.addItems(["attribute", "archetype", "bin"])
        self.x_right_dropdown.setCurrentText("bin")
        self.x_right_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)

        self.y_right_dropdown = QComboBox(self)
        self.y_right_dropdown.addItems(["attribute", "archetype", "bin"])
        self.y_right_dropdown.setCurrentText("attribute")
        self.y_right_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)

        bottom_controls_layout = QVBoxLayout()
        bottom_controls_label = QLabel("Controls (Bottom Plots):")
        bottom_controls_layout.addWidget(bottom_controls_label)

        # Left Plot Axes
        left_plot_label = QLabel("Left Plot Axes:")
        bottom_controls_layout.addWidget(left_plot_label)
        bottom_controls_layout.addWidget(self.x_left_dropdown)
        bottom_controls_layout.addWidget(self.y_left_dropdown)

        # Right Plot Axes
        right_plot_label = QLabel("Right Plot Axes:")
        bottom_controls_layout.addWidget(right_plot_label)
        bottom_controls_layout.addWidget(self.x_right_dropdown)
        bottom_controls_layout.addWidget(self.y_right_dropdown)

        # BOTTOM LEFT
        self.bottom_left_attr_dropdown = QComboBox(self)
        self.bottom_left_attr_dropdown.addItems(
            ["None", "highlighted points", "points_closest_to_archetypes", "points_between_archetypes"]
            + list(self.attributes.columns)
        )
        self.bottom_left_attr_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)
        bottom_controls_layout.addWidget(QLabel("Bottom-Left Color Attribute:"))
        bottom_controls_layout.addWidget(self.bottom_left_attr_dropdown)

        self.bottom_left_percentage_slider = QSlider(Qt.Horizontal, self)
        self.bottom_left_percentage_slider.setRange(1, 50)
        self.bottom_left_percentage_slider.setValue(5)
        self.bottom_left_percentage_slider.setTickPosition(QSlider.NoTicks)
        self.bottom_left_percentage_slider.valueChanged.connect(self.update_btm_left_heatmap)
        bottom_controls_layout.addWidget(self.bottom_left_percentage_slider)

        self.bottom_left_slider_label = QLabel("5%", self)
        bottom_controls_layout.addWidget(self.bottom_left_slider_label)

        self.bottom_left_bin_label = QLabel("Num Bins (Left):")
        bottom_controls_layout.addWidget(self.bottom_left_bin_label)

        self.bottom_left_bin_spinbox = QSpinBox(self)
        self.bottom_left_bin_spinbox.setRange(1, 50)
        self.bottom_left_bin_spinbox.setValue(2)
        self.bottom_left_bin_spinbox.valueChanged.connect(self.update_btm_left_heatmap)
        bottom_controls_layout.addWidget(self.bottom_left_bin_spinbox)

        # BOTTOM RIGHT
        self.bottom_right_attr_dropdown = QComboBox(self)
        self.bottom_right_attr_dropdown.addItems(
            ["None", "highlighted points", "points_closest_to_archetypes", "points_between_archetypes"]
            + list(self.attributes.columns)
        )
        self.bottom_right_attr_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)
        bottom_controls_layout.addWidget(QLabel("Bottom-Right Color Attribute:"))
        bottom_controls_layout.addWidget(self.bottom_right_attr_dropdown)

        self.bottom_right_percentage_slider = QSlider(Qt.Horizontal, self)
        self.bottom_right_percentage_slider.setRange(1, 50)
        self.bottom_right_percentage_slider.setValue(5)
        self.bottom_right_percentage_slider.setTickPosition(QSlider.NoTicks)
        self.bottom_right_percentage_slider.valueChanged.connect(self.update_btm_right_heatmap)
        bottom_controls_layout.addWidget(self.bottom_right_percentage_slider)

        self.bottom_right_slider_label = QLabel("5%", self)
        bottom_controls_layout.addWidget(self.bottom_right_slider_label)

        self.bottom_right_bin_label = QLabel("Num Bins (Right):")
        bottom_controls_layout.addWidget(self.bottom_right_bin_label)

        self.bottom_right_bin_spinbox = QSpinBox(self)
        self.bottom_right_bin_spinbox.setRange(1, 50)
        self.bottom_right_bin_spinbox.setValue(2)
        self.bottom_right_bin_spinbox.valueChanged.connect(self.update_btm_right_heatmap)
        bottom_controls_layout.addWidget(self.bottom_right_bin_spinbox)

        bottom_controls_layout.addStretch()
        bottom_controls_widget = QWidget()
        bottom_controls_widget.setLayout(bottom_controls_layout)
        controls_column.addWidget(bottom_controls_widget)

        # Finalize the right column
        controls_widget = QWidget()
        controls_widget.setLayout(controls_column)
        grid_layout.addWidget(controls_widget, 0, 2, 2, 1)

        # Uniform widths
        button_width = max(
            self.toggle_button.sizeHint().width(),
            self.save_button.sizeHint().width()
        )
        # Original top widgets:
        for w in [
            self.attribute_dropdown,
            self.archetype_selection,
            self.toggle_button,
            self.save_button,
            self.percentage_slider,
            self.slider_value_label,
            self.bin_spinbox,
            self.bins_label,
            self.x_left_dropdown,
            self.y_left_dropdown,
            self.x_right_dropdown,
            self.y_right_dropdown,
        ]:
            w.setFixedWidth(button_width)

        # Bottom-left set
        for w in [
            self.bottom_left_attr_dropdown,
            self.bottom_left_percentage_slider,
            self.bottom_left_slider_label,
            self.bottom_left_bin_label,
            self.bottom_left_bin_spinbox,
        ]:
            w.setFixedWidth(button_width)

        # Bottom-right set
        for w in [
            self.bottom_right_attr_dropdown,
            self.bottom_right_percentage_slider,
            self.bottom_right_slider_label,
            self.bottom_right_bin_label,
            self.bottom_right_bin_spinbox,
        ]:
            w.setFixedWidth(button_width)

        # Initial plot (top)
        self.plot_embedding()

    # -------------------------------------------------------------------------
    #  SYNCHRONIZE HIGHLIGHTED POINTS
    # -------------------------------------------------------------------------
    def sync_highlighted_points(self):
        self.highlighted_indices = self.shared_state.selected_points
        if self.attribute_dropdown.currentText() == 'highlighted points':
            self.plot_embedding(color_attribute='highlighted points')

    # -------------------------------------------------------------------------
    #  MASTER UPDATE (TOP PLOTS)
    # -------------------------------------------------------------------------
    def update_plot(self):
        selected_attr = self.attribute_dropdown.currentText()

        if selected_attr in self.attributes.columns:
            self.last_valid_attribute = selected_attr

        if self.last_valid_attribute is None and len(self.attributes.columns) > 0:
            self.last_valid_attribute = self.attributes.columns[0]

        if selected_attr in ["points_between_archetypes", "points_closest_to_archetypes"]:
            self.percentage_slider.setVisible(True)
            self.slider_value_label.setVisible(True)
            self.bin_spinbox.setVisible(True)
            self.bins_label.setVisible(True)

            if selected_attr == "points_between_archetypes":
                self.archetype_selection.setVisible(True)
                if not self.archetype_selection.count():
                    self.populate_archetype_selection()
            else:
                self.archetype_selection.setVisible(False)
        else:
            self.percentage_slider.setVisible(False)
            self.slider_value_label.setVisible(False)
            self.bin_spinbox.setVisible(False)
            self.bins_label.setVisible(False)
            self.archetype_selection.setVisible(False)

        # Re-plot (top)
        self.plot_embedding(color_attribute=selected_attr)

    def update_percentage(self):
        val = self.percentage_slider.value()
        self.slider_value_label.setText(f"{val}%")
        if not self.slider_value_label.isVisible():
            self.slider_value_label.setVisible(True)
        self.alpha_other = val / 100
        self.plot_embedding(color_attribute=self.attribute_dropdown.currentText())

    def populate_archetype_selection(self):
        self.ensure_archetypes_computed()
        if self.archetype_method == "ConvexHull" and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == "SISAL" and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            arch_points = None

        if arch_points is None:
            return

        self.archetype_selection.clear()
        pairs = []
        for i in range(len(arch_points)):
            for j in range(i + 1, len(arch_points)):
                pairs.append(f"{i + 1} - {j + 1}")
        self.archetype_selection.addItems(pairs)

    # -------------------------------------------------------------------------
    #  TOP PLOT + THEN calls bottom heatmap updates
    # -------------------------------------------------------------------------
    def plot_embedding(self, color_attribute=None):
        # Original 3D axis logic
        if not self.figure_3d.axes:
            self.figure_3d.clear()
            self.ax = self.figure_3d.add_subplot(111, projection='3d')
        else:
            self.ax = self.figure_3d.axes[0]

        n_points = self.embeddings.shape[0]
        colors = np.array(["k"] * n_points)

        # 1) "highlighted points"
        if color_attribute == "highlighted points":
            colors = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other))
            for idx in self.shared_state.selected_points:
                if idx < n_points:
                    colors[idx] = (1, 0, 0, 1)

        elif color_attribute == "points_closest_to_archetypes":
            pass

        elif color_attribute == "points_between_archetypes":
            pass

        elif color_attribute and color_attribute != "None":
            import matplotlib
            vals = self.attributes[color_attribute]
            norm = matplotlib.colors.Normalize(vals.min(), vals.max())
            colors = plt.cm.viridis_r(norm(vals))

        # Always compute top X% if recognized (for top usage)
        self.compute_top_xpct_indices()  # modifies self.closest_indices_per_archetype

        # Points_closest_to_archetypes (color logic)
        if color_attribute == "points_closest_to_archetypes" and self.closest_indices_per_archetype:
            colors = self.cumulative_bins_closest_archetypes()

        # Points_between_archetypes
        elif color_attribute == "points_between_archetypes":
            self.ensure_archetypes_computed()
            arch_points = None
            if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
                arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
            elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
                arch_points = self.cached_sisal_hull.T

            if arch_points is not None:
                selected_index = self.archetype_selection.currentIndex()
                if selected_index != -1:
                    pairs = [
                        (i, j)
                        for i in range(len(arch_points))
                        for j in range(i + 1, len(arch_points))
                    ]
                    if selected_index < len(pairs):
                        a, b = pairs[selected_index]
                        arch_a = arch_points[a]
                        arch_b = arch_points[b]
                        n_intermediate = self.bin_spinbox.value()
                        dist = np.linalg.norm(
                            self.embeddings[:, None, :3]
                            - np.linspace(arch_a, arch_b, n_intermediate)[None, :, :],
                            axis=2,
                        )
                        x_pct = self.percentage_slider.value()
                        n_closest = max(1, int((x_pct / 100) * n_points))
                        closest_indices = np.argsort(dist, axis=0)[:n_closest]
                        cmap = plt.get_cmap('viridis')
                        icolors = cmap(np.linspace(0, 1, n_intermediate))
                        colarray = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other), dtype=float)
                        for i_col, idxs_i in enumerate(closest_indices.T):
                            colarray[idxs_i, :3] = icolors[i_col, :3]
                            colarray[idxs_i, 3] = 1.0
                        colors = colarray

        # Recompute or update standard top-level heatmaps
        self.means_matrix = self.compute_attribute_archetype_means()
        self.bin_matrix = self.compute_bin_attribute_means()

        # Now update both bottom heatmaps (still the old approach for top)
        self.update_btm_left_heatmap()
        self.update_btm_right_heatmap()

        # Draw 3D scatter
        self.ax.clear()
        self.scatter = self.ax.scatter(
            self._scaled_embeddings[:, 0],
            self._scaled_embeddings[:, 1],
            self._scaled_embeddings[:, 2],
            c=colors, marker='.'
        )
        self.ax.set_title(
            f"Session: {self.session_id}\n"
            f"Regions: {', '.join(sorted(self.brain_regions))}\n"
            f"Algorithm: {self.algorithm}",
            fontsize=12
        )
        self.ax.set_xlabel('Dimension 1')
        self.ax.set_ylabel('Dimension 2')
        self.ax.set_zlabel('Dimension 3')

        # 2D XY, XZ, YZ
        self.figure_2d.clear()
        self.ax_xy = self.figure_2d.add_subplot(311, aspect='equal')
        self.ax_xz = self.figure_2d.add_subplot(312, aspect='equal')
        self.ax_yz = self.figure_2d.add_subplot(313, aspect='equal')

        self.scatter_xy = self.ax_xy.scatter(
            self.embeddings[:, 0], self.embeddings[:, 1], c=colors, marker='.'
        )
        self.scatter_xz = self.ax_xz.scatter(
            self.embeddings[:, 0], self.embeddings[:, 2], c=colors, marker='.'
        )
        self.scatter_yz = self.ax_yz.scatter(
            self.embeddings[:, 1], self.embeddings[:, 2], c=colors, marker='.'
        )

        # Axis buffer
        for ax_obj, (xx, yy) in zip(
            [self.ax_xy, self.ax_xz, self.ax_yz],
            [
                (self.embeddings[:, 0], self.embeddings[:, 1]),
                (self.embeddings[:, 0], self.embeddings[:, 2]),
                (self.embeddings[:, 1], self.embeddings[:, 2])
            ]
        ):
            x_min, x_max = xx.min(), xx.max()
            y_min, y_max = yy.min(), yy.max()
            buffer_x = (x_max - x_min) * 0.2
            buffer_y = (y_max - y_min) * 0.2
            ax_obj.set_xlim(x_min - buffer_x, x_max + buffer_x)
            ax_obj.set_ylim(y_min - buffer_y, y_max + buffer_y)
            ax_obj.set_aspect('equal', adjustable='datalim')

        self.ax_xy.set_title("1-2 Dim")
        self.ax_xz.set_title("1-3 Dim")
        self.ax_yz.set_title("2-3 Dim")

        for ax_obj in [self.ax_xy, self.ax_xz, self.ax_yz]:
            ax_obj.spines['top'].set_visible(False)
            ax_obj.spines['right'].set_visible(False)
            ax_obj.spines['left'].set_visible(False)
            ax_obj.spines['bottom'].set_visible(False)
            ax_obj.set_xticks([])
            ax_obj.set_yticks([])

        self.canvas_2d.draw_idle()
        self.canvas_3d.draw_idle()

        # If hull lines
        if self.archetype_method == 'ConvexHull':
            self.add_convex_hull(self.ax, fig_type='3d')
            self.add_convex_hull(self.ax_xy, fig_type='2d', projection=(0, 1))
            self.add_convex_hull(self.ax_xz, fig_type='2d', projection=(0, 2))
            self.add_convex_hull(self.ax_yz, fig_type='2d', projection=(1, 2))
        if self.archetype_method == 'SISAL':
            self.add_sisal_hull(self.ax, fig_type='3d')
            self.add_sisal_hull(self.ax_xy, fig_type='2d', projection=(0, 1))
            self.add_sisal_hull(self.ax_xz, fig_type='2d', projection=(0, 2))
            self.add_sisal_hull(self.ax_yz, fig_type='2d', projection=(1, 2))

        # Lasso reinit
        if hasattr(self, 'selector') and self.selector and self.selector.selector_enabled:
            self.selector.toggle_selection_mode()
        if hasattr(self, 'lasso_xy') and self.lasso_xy and self.lasso_xy.selector_enabled:
            self.lasso_xy.toggle_selection_mode()
        if hasattr(self, 'lasso_xz') and self.lasso_xz and self.lasso_xz.selector_enabled:
            self.lasso_xz.toggle_selection_mode()
        if hasattr(self, 'lasso_yz') and self.lasso_yz and self.lasso_yz.selector_enabled:
            self.lasso_yz.toggle_selection_mode()

        self.selector = SelectFromCollectionLasso(self.ax, self.scatter, self.on_select_3d, is_3d=True)
        self.lasso_xy = SelectFromCollectionLasso(self.ax_xy, self.scatter_xy, self.on_select_2d)
        self.lasso_xz = SelectFromCollectionLasso(self.ax_xz, self.scatter_xz, self.on_select_2d)
        self.lasso_yz = SelectFromCollectionLasso(self.ax_yz, self.scatter_yz, self.on_select_2d)
        if self.lasso_active:
            self.selector.toggle_selection_mode()
            self.lasso_xy.toggle_selection_mode()
            self.lasso_xz.toggle_selection_mode()
            self.lasso_yz.toggle_selection_mode()
        
        
    def compute_bin_attribute_means(self):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None
    
        arch_a = arch_points[0]
        arch_b = arch_points[1]
        n_intermediate = self.bin_spinbox.value()
        inter_pts = np.linspace(arch_a, arch_b, n_intermediate)
    
        n_points = self.embeddings.shape[0]
        x_pct = self.percentage_slider.value()
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :], axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]
    
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, n_intermediate), dtype=float)
        for bin_i in range(n_intermediate):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_attribute_archetype_means(self):
        if not self.closest_indices_per_archetype:
            return None
        n_archetypes = len(self.closest_indices_per_archetype)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M
    
   
    
    # -------------------------------------------------------------------------
    #  BOTTOM-LEFT HEATMAP
    # -------------------------------------------------------------------------
    def update_btm_left_heatmap(self):
        """Recompute & redraw ONLY the bottom-left heatmap using the bottom-left slider/spinbox."""
        selected_attr = self.bottom_left_attr_dropdown.currentText()
        if selected_attr in self.attributes.columns:
            self.last_valid_attribute_left = selected_attr
        elif not self.last_valid_attribute_left and len(self.attributes.columns) > 0:
            self.last_valid_attribute_left = self.attributes.columns[0]

        # Grab bottom-left slider/spinbox
        x_pct = self.bottom_left_percentage_slider.value()
        self.bottom_left_slider_label.setText(f"{x_pct}%")
        n_bins = self.bottom_left_bin_spinbox.value()

        # 1) Compute a new set of 'closest_indices_per_archetype_left' based on x_pct
        self.compute_bottom_left_xpct_indices(x_pct)

        # 2) Build means_matrix_left / bin_matrix_left from that
        self.means_matrix_left = self.compute_attribute_archetype_means_bottom_left()
        self.bin_matrix_left = self.compute_bin_attribute_means_bottom_left(x_pct, n_bins)

        # Now do the bottom-left pcolormesh
        dim_x = self.x_left_dropdown.currentText()
        dim_y = self.y_left_dropdown.currentText()

        self.figure_btm_left.clear()
        ax = self.figure_btm_left.add_subplot(111)
        ax.set_title(f"Bottom-Left Heatmap ({selected_attr})", fontsize=10)

        matrix, x_labels, y_labels = self.get_heatmap_data_bottom_left(dim_x, dim_y)
        if matrix is None:
            self.canvas_btm_left.draw_idle()
            return

        pcm = ax.pcolormesh(
            matrix, cmap="viridis_r", edgecolors="k", linewidth=1, shading="flat"
        )
        cbar = self.figure_btm_left.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Value", rotation=270, labelpad=15)

        ax.set_xticks(np.arange(len(y_labels)) + 0.5)
        ax.set_xticklabels(y_labels)
        ax.set_yticks(np.arange(len(x_labels)) + 0.5)
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(dim_y.capitalize())
        ax.set_ylabel(dim_x.capitalize())
        ax.invert_yaxis()

        self.canvas_btm_left.draw_idle()

    # -------------------------------------------------------------------------
    #  BOTTOM-RIGHT HEATMAP
    # -------------------------------------------------------------------------
    def update_btm_right_heatmap(self):
        """Recompute & redraw ONLY the bottom-right heatmap using the bottom-right slider/spinbox."""
        selected_attr = self.bottom_right_attr_dropdown.currentText()
        if selected_attr in self.attributes.columns:
            self.last_valid_attribute_right = selected_attr
        elif not self.last_valid_attribute_right and len(self.attributes.columns) > 0:
            self.last_valid_attribute_right = self.attributes.columns[0]

        # Grab bottom-right slider/spinbox
        x_pct = self.bottom_right_percentage_slider.value()
        self.bottom_right_slider_label.setText(f"{x_pct}%")
        n_bins = self.bottom_right_bin_spinbox.value()

        # 1) Compute a new set of 'closest_indices_per_archetype_right' based on x_pct
        self.compute_bottom_right_xpct_indices(x_pct)

        # 2) Build means_matrix_right / bin_matrix_right
        self.means_matrix_right = self.compute_attribute_archetype_means_bottom_right()
        self.bin_matrix_right = self.compute_bin_attribute_means_bottom_right(x_pct, n_bins)

        # Now do the bottom-right pcolormesh
        dim_x = self.x_right_dropdown.currentText()
        dim_y = self.y_right_dropdown.currentText()

        self.figure_btm_right.clear()
        ax = self.figure_btm_right.add_subplot(111)
        ax.set_title(f"Bottom-Right Heatmap ({selected_attr})", fontsize=10)

        matrix, x_labels, y_labels = self.get_heatmap_data_bottom_right(dim_x, dim_y)
        if matrix is None:
            self.canvas_btm_right.draw_idle()
            return

        pcm = ax.pcolormesh(
            matrix, cmap="viridis_r", edgecolors="k", linewidth=1, shading="flat"
        )
        cbar = self.figure_btm_right.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Value", rotation=270, labelpad=15)

        ax.set_xticks(np.arange(len(y_labels)) + 0.5)
        ax.set_xticklabels(y_labels)
        ax.set_yticks(np.arange(len(x_labels)) + 0.5)
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(dim_y.capitalize())
        ax.set_ylabel(dim_x.capitalize())
        ax.invert_yaxis()

        self.canvas_btm_right.draw_idle()

    # -------------------------------------------------------------------------
    #  BOTTOM LEFT: get_heatmap_data_bottom_left
    # -------------------------------------------------------------------------
    def get_heatmap_data_bottom_left(self, dim_x, dim_y):
        """
        Same concept as get_heatmap_data, but uses self.means_matrix_left / self.bin_matrix_left
        and self.last_valid_attribute_left instead of top's variables.
        """
        combos = {}

        # 1) attribute x archetype
        if self.means_matrix_left is not None:
            n_archetypes = self.means_matrix_left.shape[1]
            arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "archetype")] = (self.means_matrix_left, attrs, arch_labels)
            combos[("archetype", "attribute")] = (self.means_matrix_left.T, arch_labels, attrs)

        # 2) attribute x bin
        if self.bin_matrix_left is not None:
            n_bins = self.bin_matrix_left.shape[1]
            bin_labels = [f"Bin {i+1}" for i in range(n_bins)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "bin")] = (self.bin_matrix_left, attrs, bin_labels)
            combos[("bin", "attribute")] = (self.bin_matrix_left.T, bin_labels, attrs)

        # 3) If user chooses (archetype, bin) or (bin, archetype), do bottom-left cumulative approach
        if (dim_x, dim_y) in [("archetype", "bin"), ("bin", "archetype")]:
            chosen_attr = self.last_valid_attribute_left
            if not chosen_attr or chosen_attr not in self.attributes.columns:
                chosen_attr = self.attributes.columns[0]  # fallback
            A = self.compute_archetype_bin_attribute_means_cumulative_bottom_left(chosen_attr)
            if A is not None:
                n_archetypes = A.shape[0]
                n_bins = A.shape[1]
                arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
                bin_labels = [f"Bin {j+1}" for j in range(n_bins)]
                if (dim_x, dim_y) == ("archetype", "bin"):
                    return A, arch_labels, bin_labels
                else:
                    return A.T, bin_labels, arch_labels
            return None, None, None

        return combos.get((dim_x, dim_y), (None, None, None))

    # -------------------------------------------------------------------------
    #  BOTTOM RIGHT: get_heatmap_data_bottom_right
    # -------------------------------------------------------------------------
    def get_heatmap_data_bottom_right(self, dim_x, dim_y):
        """
        Same concept as get_heatmap_data_bottom_left, but for right side.
        """
        combos = {}

        if self.means_matrix_right is not None:
            n_archetypes = self.means_matrix_right.shape[1]
            arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "archetype")] = (self.means_matrix_right, attrs, arch_labels)
            combos[("archetype", "attribute")] = (self.means_matrix_right.T, arch_labels, attrs)

        if self.bin_matrix_right is not None:
            n_bins = self.bin_matrix_right.shape[1]
            bin_labels = [f"Bin {i+1}" for i in range(n_bins)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "bin")] = (self.bin_matrix_right, attrs, bin_labels)
            combos[("bin", "attribute")] = (self.bin_matrix_right.T, bin_labels, attrs)

        if (dim_x, dim_y) in [("archetype", "bin"), ("bin", "archetype")]:
            chosen_attr = self.last_valid_attribute_right
            if not chosen_attr or chosen_attr not in self.attributes.columns:
                chosen_attr = self.attributes.columns[0]
            A = self.compute_archetype_bin_attribute_means_cumulative_bottom_right(chosen_attr)
            if A is not None:
                n_archetypes = A.shape[0]
                n_bins = A.shape[1]
                arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
                bin_labels = [f"Bin {j+1}" for j in range(n_bins)]
                if (dim_x, dim_y) == ("archetype", "bin"):
                    return A, arch_labels, bin_labels
                else:
                    return A.T, bin_labels, arch_labels
            return None, None, None

        return combos.get((dim_x, dim_y), (None, None, None))

    # -------------------------------------------------------------------------
    #  BOTTOM LEFT: separate compute logic
    # -------------------------------------------------------------------------
    def compute_bottom_left_xpct_indices(self, x_pct):
        """
        We'll define a new 'closest_indices_per_archetype_left' using x_pct
        and ignoring self.percentage_slider from the top.
        """
        self.closest_indices_per_archetype_left = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        arch_pts_3d = None
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        if arch_pts_3d is None or len(arch_pts_3d) == 0:
            return

        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )  # shape (n_points, n_archetypes)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        n_archetypes = arch_pts_3d.shape[0]
        self.closest_indices_per_archetype_left = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def compute_attribute_archetype_means_bottom_left(self):
        if not self.closest_indices_per_archetype_left:
            return None
        n_archetypes = len(self.closest_indices_per_archetype_left)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype_left):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        # row-wise normalize
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M

    def compute_bin_attribute_means_bottom_left(self, x_pct, n_bins):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None

        arch_a = arch_points[0]
        arch_b = arch_points[1]
        inter_pts = np.linspace(arch_a, arch_b, n_bins)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :], axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, n_bins), dtype=float)
        for bin_i in range(n_bins):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_archetype_bin_attribute_means_cumulative_bottom_left(self, chosen_attr):
        if self.closest_indices_per_archetype_left is None:
            return None
        if chosen_attr not in self.attributes.columns:
            return None

        n_archetypes = len(self.closest_indices_per_archetype_left)
        if n_archetypes == 0:
            return None

        n_points = self.embeddings.shape[0]
        x_pct = self.bottom_left_percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bottom_left_bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        M = np.full((n_archetypes, n_bins), np.nan, dtype=float)

        for j in range(n_archetypes):
            idxs_j = self.closest_indices_per_archetype_left[j]
            if len(idxs_j) == 0:
                continue
            # Distances for shading logic
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) > 0:
                    val = self.attributes.iloc[bin_points][chosen_attr].mean()
                    M[j, i_bin] = val
        return M

    # -------------------------------------------------------------------------
    #  BOTTOM RIGHT: separate compute logic
    # -------------------------------------------------------------------------
    def compute_bottom_right_xpct_indices(self, x_pct):
        self.closest_indices_per_archetype_right = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        arch_pts_3d = None
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        if arch_pts_3d is None or len(arch_pts_3d) == 0:
            return

        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )
        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        n_archetypes = arch_pts_3d.shape[0]
        self.closest_indices_per_archetype_right = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def compute_attribute_archetype_means_bottom_right(self):
        if not self.closest_indices_per_archetype_right:
            return None
        n_archetypes = len(self.closest_indices_per_archetype_right)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype_right):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M

    def compute_bin_attribute_means_bottom_right(self, x_pct, n_bins):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None

        arch_a = arch_points[0]
        arch_b = arch_points[1]
        inter_pts = np.linspace(arch_a, arch_b, n_bins)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :], axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, n_bins), dtype=float)
        for bin_i in range(n_bins):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_archetype_bin_attribute_means_cumulative_bottom_right(self, chosen_attr):
        if self.closest_indices_per_archetype_right is None:
            return None
        if chosen_attr not in self.attributes.columns:
            return None

        n_archetypes = len(self.closest_indices_per_archetype_right)
        if n_archetypes == 0:
            return None

        n_points = self.embeddings.shape[0]
        x_pct = self.bottom_right_percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bottom_right_bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        M = np.full((n_archetypes, n_bins), np.nan, dtype=float)

        for j in range(n_archetypes):
            idxs_j = self.closest_indices_per_archetype_right[j]
            if len(idxs_j) == 0:
                continue
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) > 0:
                    val = self.attributes.iloc[bin_points][chosen_attr].mean()
                    M[j, i_bin] = val
        return M

    # -------------------------------------------------------------------------
    #  ensure_archetypes_computed, hull drawing, lasso, etc. remain the same
    # -------------------------------------------------------------------------
    def ensure_archetypes_computed(self):
        if self.archetype_method == "ConvexHull" and self.cached_convex_hull is None:
            self.cached_convex_hull = ConvexHull(self.embeddings[:, :3])
        elif self.archetype_method == "SISAL" and self.cached_sisal_hull is None:
            ArchsMin, _ = findMinSimplex(10, self.embeddings, 1, 4)
            self.cached_sisal_hull = ArchsMin[:, np.argsort(ArchsMin[0, :])]

    # -------------------------------------------------------------------------
    #  CONVEX / SISAL hulls
    # -------------------------------------------------------------------------
    def add_convex_hull(self, ax, fig_type='3d', projection=None):
        try:
            if self.cached_convex_hull is None:
                self.cached_convex_hull = ConvexHull(self.embeddings[:, :3])
            hull = self.cached_convex_hull
            if fig_type == '3d':
                vertices_3d = self.embeddings[hull.vertices, :3]
                edges = []
                for simplex in hull.simplices:
                    edges.append(self.embeddings[simplex, :3])
                edge_collection = Line3DCollection(edges, colors='k', linewidths=0.5)
                ax.add_collection3d(edge_collection)
                ax.scatter(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2],
                           c='b', s=10, label='Archetypes')
                for i, (x, y, z) in enumerate(vertices_3d):
                    ax.text(x, y, z, str(i + 1), color='red', fontsize=10)
            elif fig_type == '2d':
                if projection is None:
                    return
                vertices_2d = self.embeddings[hull.vertices, projection]
                edges = []
                for simplex in hull.simplices:
                    edge = self.embeddings[simplex, projection]
                    edges.append(edge)
                    ax.plot(*edge.T, c='k', linewidth=0.5)
                ax.scatter(vertices_2d[:, 0], vertices_2d[:, 1],
                           c='b', s=10, label='Archetypes', zorder=5)
                for i, (xx, yy) in enumerate(vertices_2d):
                    ax.text(xx, yy, str(i + 1), color='red', fontsize=10, zorder=6)
                self.adjust_axis_limits(ax, self.embeddings[:, projection], vertices_2d)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error in adding ConvexHull: {e}')

    def add_sisal_hull(self, ax, fig_type='3d', projection=None):
        try:
            if self.cached_sisal_hull is None:
                ArchsMin, VolArchReal = findMinSimplex(10, self.embeddings, 1, 4)
                ArchsOrder = np.argsort(ArchsMin[0, :])
                self.cached_sisal_hull = ArchsMin[:, ArchsOrder]
            ArchsMin = self.cached_sisal_hull
            NArchetypes = ArchsMin.shape[1]
            if fig_type == '3d':
                ArchsMin_3d = ArchsMin[:3, :]
                edges = []
                for i in range(NArchetypes):
                    for j in range(i + 1, NArchetypes):
                        edge = np.array([ArchsMin_3d[:, i], ArchsMin_3d[:, j]])
                        edges.append(edge)
                edge_collection = Line3DCollection(edges, colors='k', linewidths=0.5)
                ax.add_collection3d(edge_collection)
                ax.scatter(ArchsMin_3d[0, :], ArchsMin_3d[1, :], ArchsMin_3d[2, :],
                           c='b', s=10, label='Archetypes')
                for i, (x, y, z) in enumerate(ArchsMin_3d.T):
                    ax.text(x, y, z, str(i + 1), color='red', fontsize=10)
            elif fig_type == '2d':
                if projection is None:
                    return
                ArchsMin_2d = ArchsMin[list(projection), :]
                edges = []
                for i in range(NArchetypes):
                    for j in range(i + 1, NArchetypes):
                        edge = np.array([ArchsMin_2d[:, i], ArchsMin_2d[:, j]])
                        edges.append(edge)
                        ax.plot(*edge.T, c='k', linewidth=0.5)
                ax.scatter(ArchsMin_2d[0, :], ArchsMin_2d[1, :],
                           c='b', s=10, label='Archetypes', zorder=5)
                for i, (xx, yy) in enumerate(ArchsMin_2d.T):
                    ax.text(xx, yy, str(i + 1), color='red', fontsize=10, zorder=6)
                scatter_points = self.embeddings[:, projection]
                self.adjust_axis_limits(ax, scatter_points, ArchsMin_2d.T)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error in adding SISAL hull: {e}')

    def adjust_axis_limits(self, ax, scatter_points, polytope_points):
        all_points = np.vstack([scatter_points, polytope_points])
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        buffer_x = (x_max - x_min) * 0.1
        buffer_y = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
        ax.set_ylim(y_min - buffer_y, y_max + buffer_y)

    def invalidate_cache(self):
        self.cached_convex_hull = None
        self.cached_sisal_hull = None

    # -------------------------------------------------------------------------
    #  LASSO
    # -------------------------------------------------------------------------
    def on_select_3d(self, indices):
        self.highlighted_indices = list(indices)

    def on_select_2d(self, indices):
        self.highlighted_indices = list(indices)

    def toggle_lasso_selection(self):
        if self.lasso_active:
            self.lasso_active = False
            if self.selector and self.selector.selector_enabled:
                self.selector.toggle_selection_mode()
            if self.lasso_xy and self.lasso_xy.selector_enabled:
                self.lasso_xy.toggle_selection_mode()
            if self.lasso_xz and self.lasso_xz.selector_enabled:
                self.lasso_xz.toggle_selection_mode()
            if self.lasso_yz and self.lasso_yz.selector_enabled:
                self.lasso_yz.toggle_selection_mode()
            self.canvas_3d.draw_idle()
            self.canvas_2d.draw_idle()
        else:
            self.lasso_active = True
            if self.selector:
                self.selector.toggle_selection_mode()
            if self.lasso_xy:
                self.lasso_xy.toggle_selection_mode()
            if self.lasso_xz:
                self.lasso_xz.toggle_selection_mode()
            if self.lasso_yz:
                self.lasso_yz.toggle_selection_mode()

    def save_selected_points(self):
        self.selected_points = list(self.highlighted_indices)
        self.shared_state.set_selected_points(self.selected_points)
        print(f"Selected points saved: {self.selected_points}")
        if self.attribute_dropdown.currentText() == 'highlighted points':
            self.plot_embedding(color_attribute='highlighted points')

    # -------------------------------------------------------------------------
    #  "points_closest_to_archetypes" color logic
    # -------------------------------------------------------------------------
    def compute_top_xpct_indices(self):
        """
        For each archetype j, define top X% by distance => closest_indices_per_archetype[j].
        We do this for potential usage in color or in the heatmaps.
        """
        self.closest_indices_per_archetype = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        else:
            return
        n_archetypes = arch_pts_3d.shape[0]
        if n_archetypes == 0:
            return
        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )  # shape (n_points, n_archetypes)

        x_pct = self.percentage_slider.value()
        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        self.closest_indices_per_archetype = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def cumulative_bins_closest_archetypes(self):
        """
        This sets the scatter color per chunk for each archetype in 'points_closest_to_archetypes'.
        Returns RGBA array of shape (n_points, 4).
        """
        import matplotlib
        import colorsys
        n_points = self.embeddings.shape[0]
        colors = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other), dtype=float)

        n_archetypes = len(self.closest_indices_per_archetype)
        if n_archetypes == 0:
            return colors

        base_cmap = matplotlib.colormaps.get_cmap("viridis")
        base_colors = base_cmap(np.linspace(0, 1, n_archetypes))

        x_pct = self.percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        for j, idxs_j in enumerate(self.closest_indices_per_archetype):
            if len(idxs_j) == 0:
                continue
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            base_r, base_g, base_b, _ = base_colors[j]
            h, l, s = colorsys.rgb_to_hls(base_r, base_g, base_b)

            min_lightness = 0.3
            if l < min_lightness:
                l = min_lightness

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) == 0:
                    break

                fraction = (i_bin + 1) / (n_bins + 1)
                lightness_increment = fraction * 0.3
                new_l = min(l + lightness_increment, 1.0)

                shaded_r, shaded_g, shaded_b = colorsys.hls_to_rgb(h, new_l, s)
                colors[bin_points, :3] = (shaded_r, shaded_g, shaded_b)
                colors[bin_points, 3] = 1.0

        return colors

    def get_archetype_coords(self, j):
        """Return the j-th archetype's 3D coords from either convex hull or SISAL."""
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull is not None:
            arch_points_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
            if j < len(arch_points_3d):
                return arch_points_3d[j]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points_3d = self.cached_sisal_hull.T[:, :3]
            if j < len(arch_points_3d):
                return arch_points_3d[j]
        return np.array([0, 0, 0], dtype=float)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = AnalysisGUI()
    main_window.show()
    sys.exit(app.exec_())

class LabeledSlider(QtWidgets.QWidget):
    def __init__(self, minimum, maximum, interval=1, parent=None):
        super().__init__(parent)

        levels = range(minimum, maximum + interval, interval)
        self.levels = list(zip(levels, map(str, levels)))

        self.left_margin = 10
        self.top_margin = 10
        self.right_margin = 10
        self.bottom_margin = 20  # Adjust bottom margin to fit labels

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(self.left_margin, self.top_margin, self.right_margin, self.bottom_margin)

        self.sl = QtWidgets.QSlider(Qt.Horizontal, self)
        self.sl.setMinimum(minimum)
        self.sl.setMaximum(maximum)
        self.sl.setValue(minimum)
        self.sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sl.setTickInterval(interval)
        self.sl.setSingleStep(1)
        self.layout.addWidget(self.sl)

    def paintEvent(self, e):
        super().paintEvent(e)

        style = self.sl.style()
        painter = QPainter(self)
        st_slider = QStyleOptionSlider()
        st_slider.initFrom(self.sl)
        st_slider.orientation = self.sl.orientation()

        length = style.pixelMetric(QStyle.PM_SliderLength, st_slider, self.sl)
        available = style.pixelMetric(QStyle.PM_SliderSpaceAvailable, st_slider, self.sl)

        for v, v_str in self.levels:
            rect = painter.drawText(QRect(), Qt.TextDontPrint, v_str)

            x_loc = QStyle.sliderPositionFromValue(self.sl.minimum(),
                                                   self.sl.maximum(),
                                                   v,
                                                   available) + length // 2

            left = x_loc - rect.width() // 2 + self.left_margin
            bottom = self.rect().bottom() - 10  # Adjust position of labels

            pos = QPoint(left, bottom)
            painter.drawText(pos, v_str)

class SliderWithText(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paintEvent(self, event):
        # Call the base class's paintEvent to render the slider
        super().paintEvent(event)
        
        # Create a QPainter object to draw the text
        painter = QPainter(self)

        # Set a readable font size
        painter.setFont(QFont("Arial", 10))

        # Create and initialize the style option
        option = QStyleOptionSlider()
        self.initStyleOption(option)

        # Get the position of the slider handle
        handle_rect = self.style().subControlRect(
            self.style().CC_Slider, option, self.style().SC_SliderHandle, self
        )
        handle_center = handle_rect.center()

        # Calculate percentage text
        percentage = self.value()
        text = f"{percentage}%"

        # Draw the text centered above the slider handle
        text_rect = painter.boundingRect(
            handle_center.x() - 20, handle_center.y() - 30, 40, 20, Qt.AlignCenter, text
        )
        painter.drawText(text_rect, Qt.AlignCenter, text)

        # End painting
        painter.end()
        
class SharedState(QObject):
    """Shared state for communicating between windows."""
    updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.selected_points = []

    def set_selected_points(self, points):
        """Update the selected points and notify all listeners."""
        self.selected_points = points
        self.updated.emit()


class SelectFromCollectionLasso:
    def __init__(self, ax, scatter, on_select_callback, is_3d=False):
        self.ax = ax
        self.scatter = scatter
        self.is_3d = is_3d
        self.on_select_callback = on_select_callback
        self.ind = set()
        self._initialize_data()
        self.lasso = None
        self.selector_enabled = False

    def _initialize_data(self):
        """Initialize scatter plot data."""
        if self.is_3d:
            # For 3D scatter, _offsets3d is a (3, N) tuple
            self.data = np.array(self.scatter._offsets3d).T
        else:
            # For 2D scatter, get_offsets() is shape (N, 2)
            self.data = self.scatter.get_offsets()

    def toggle_selection_mode(self):
        """Enable or disable lasso selection."""
        self.selector_enabled = not self.selector_enabled
        if self.selector_enabled:
            if self.is_3d:
                self.ax.disable_mouse_rotation()
            self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        else:
            if self.is_3d:
                self.ax.mouse_init()
            if self.lasso:
                self.lasso.disconnect_events()
                self.lasso = None

    def on_select(self, verts):
        """Handle lasso selection."""
        path = Path(verts)
        if self.is_3d:
            projected = self.project_3d_to_2d(self.data)
        else:
            projected = self.data
        selected_indices = np.nonzero(path.contains_points(projected))[0]
        self.ind = set(selected_indices)
        self.on_select_callback(list(self.ind))

    def project_3d_to_2d(self, points):
        """Project 3D points to 2D for selection logic."""
        trans = self.ax.get_proj()
        points_4d = np.column_stack((points, np.ones(points.shape[0])))
        projected = np.dot(points_4d, trans.T)
        projected[:, :2] /= projected[:, 3, None]
        return projected[:, :2]


class AnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.session = None
        self.viewers = []
        self.shared_state = SharedState()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Session Analysis GUI')

        window_width = 300  # Desired width
        window_height = 600  # Desired height
        self.resize(window_width, window_height)  # Adjust the size of the window
        
        # Center the window on the screen
        self.show()  # Ensure the window is initialized before measuring geometry
        frame_geometry = self.frameGeometry()  # Includes window decorations
        screen_geometry = QApplication.desktop().screenGeometry()
        
        # Calculate center
        x = (screen_geometry.width() - frame_geometry.width()) // 2 - (400 + 150 + 25)
        y = (screen_geometry.height() - frame_geometry.height()) // 2
        self.move(x, y)
        
        self.layout = QVBoxLayout()

        self.load_file_button = QPushButton('Load .pkl File', self)
        self.load_file_button.clicked.connect(self.load_pkl_file)
        self.layout.addWidget(self.load_file_button)

        self.file_label = QLabel('Selected File: None', self)
        self.layout.addWidget(self.file_label)

        self.region_label = QLabel('Select Brain Regions:')
        self.layout.addWidget(self.region_label)
        self.region_list = QListWidget(self)
        self.region_list.setSelectionMode(QListWidget.MultiSelection)
        self.layout.addWidget(self.region_list)

        self.embedding_label = QLabel('Select Embedding Method:')
        self.layout.addWidget(self.embedding_label)
        self.radio_group = QButtonGroup(self)

        self.pca_radio = QRadioButton('PCA', self)
        self.radio_group.addButton(self.pca_radio)
        self.layout.addWidget(self.pca_radio)

        self.fa_radio = QRadioButton('FA', self)
        self.radio_group.addButton(self.fa_radio)
        self.layout.addWidget(self.fa_radio)

        self.lem_radio = QRadioButton('Laplacian', self)
        self.radio_group.addButton(self.lem_radio)
        self.layout.addWidget(self.lem_radio)

        # Archetype Method
        self.archetype_label = QLabel('Select Archetype Method:')
        self.layout.addWidget(self.archetype_label)
        self.archetype_dropdown = QComboBox(self)
        self.archetype_dropdown.addItems(['None', 'ConvexHull', 'nfindr', 'SISAL', 'alpha_shape'])
        self.layout.addWidget(self.archetype_dropdown)
        
        self.analyse_button = QPushButton('Proceed with Analysis', self)
        self.analyse_button.clicked.connect(self.proceed_with_analysis)
        self.layout.addWidget(self.analyse_button)
    
        self.setLayout(self.layout)

    def load_pkl_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select .pkl File', '', 'Pickle Files (*.pkl)')
        if file_path:
            self.file_label.setText(f'Selected File: {file_path}')
            try:
                with open(file_path, 'rb') as f:
                    self.session = pkl.load(f)

                if hasattr(self.session, 'unit_channels') and 'structure_acronym' in self.session.unit_channels:
                    brain_regions = sorted(self.session.unit_channels['structure_acronym'].unique())
                    self.region_list.clear()
                    self.region_list.addItems(brain_regions)
                else:
                    QMessageBox.warning(self, 'Error', 'No brain region information found.')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load file: {e}')

    def proceed_with_analysis(self):
        if not self.session:
            QMessageBox.warning(self, 'Error', 'No session loaded.')
            return

        selected_regions = [item.text() for item in self.region_list.selectedItems()]
        if not selected_regions:
            QMessageBox.warning(self, 'Error', 'No brain regions selected.')
            return

        selected_method = None
        if self.pca_radio.isChecked():
            selected_method = 'PCA'
        elif self.fa_radio.isChecked():
            selected_method = 'FA'
        elif self.lem_radio.isChecked():
            selected_method = 'Laplacian'

        if not selected_method:
            QMessageBox.warning(self, 'Error', 'No embedding method selected.')
            return

        neuron_ids = self.session.unit_channels[self.session.unit_channels['structure_acronym'].isin(selected_regions)].index
        filtered_matrix = self.session.neuron_x_time_matrix.loc[neuron_ids]

        smoothed = gaussian_filter1d(filtered_matrix.values, sigma=3.5, axis=1)
        z_scored_data = StandardScaler().fit_transform(smoothed.T).T

        embeddings = None
        if selected_method == 'PCA':
            embeddings = PCA().fit_transform(z_scored_data.T)
        elif selected_method == 'FA':
            embeddings = FactorAnalysis().fit_transform(z_scored_data.T)
        elif selected_method == 'Laplacian':
            n_neighbors = max(1, int(z_scored_data.shape[1] * 0.005))
            adjacency_matrix = kneighbors_graph(z_scored_data.T, n_neighbors=n_neighbors,
                                                mode='connectivity', include_self=False).toarray()
            degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
            laplacian_rw = np.eye(adjacency_matrix.shape[0]) - np.linalg.inv(degree_matrix) @ adjacency_matrix
            _, eigenvectors = eigsh(laplacian_rw, k=4, which='SM')
            embeddings = eigenvectors[:, 1:]

        session_id = "Session_01"
        brain_regions = selected_regions
        algorithm = selected_method

        # We assume self.session.discrete_attributes includes all attributes
        # attrs = self.session.discrete_attributes
        attrs = self.session.continuous_attributes
        
        self.archetype_method = self.archetype_dropdown.currentText()
        viewer = EmbeddingWindow(embeddings, attrs,
                                 session_id, brain_regions, algorithm,
                                 self.shared_state, self.archetype_method)
        self.viewers.append(viewer)
        viewer.show()

class EmbeddingWindow(QWidget):
    def __init__(
        self,
        embeddings,
        attributes,
        session_id,
        brain_regions,
        algorithm,
        shared_state,
        archetype_method,
        alpha_other=0.3
    ):
        super().__init__()

        self.embeddings = embeddings
        self._scaled_embeddings = embeddings.copy()
        self.attributes = attributes  # DataFrame (n_points, n_attributes)
        self.session_id = session_id
        self.brain_regions = brain_regions
        self.algorithm = algorithm
        self.shared_state = shared_state
        self.archetype_method = archetype_method
        self.alpha_other = alpha_other
        self.selector = None
        self.lasso_active = False
        self.highlighted_indices = []

        # Caches for convex hull and SISAL (top usage)
        self.cached_convex_hull = None
        self.cached_sisal_hull = None

        # If embeddings are very small, scale them
        self._scale_embedding = False
        self._scale_factor = 100
        if np.max(self.embeddings) < 1:
            self._scaled_embeddings *= self._scale_factor
            self._scale_embedding = True

        # -- For the TOP logic --
        self.closest_indices_per_archetype = None    # used by top
        self.means_matrix = None                     # shape = (n_attrs, n_archetypes) [top usage]
        self.bin_matrix = None                       # shape = (n_attrs, n_bins) [top usage]
        self.last_valid_attribute = None             # for the top

        # -- For the BOTTOM-LEFT logic --
        self.last_valid_attribute_left = None
        self.closest_indices_per_archetype_left = None
        self.means_matrix_left = None
        self.bin_matrix_left = None

        # -- For the BOTTOM-RIGHT logic --
        self.last_valid_attribute_right = None
        self.closest_indices_per_archetype_right = None
        self.means_matrix_right = None
        self.bin_matrix_right = None

        # Connect to shared state for multi-window sync
        self.shared_state.updated.connect(self.sync_highlighted_points)

        self.initUI()

    # -------------------------------------------------------------------------
    #  INIT UI
    # -------------------------------------------------------------------------
    def initUI(self):
        self.setWindowTitle('3D Embedding Viewer with EXACT Schematic Layout')
        self.showMaximized()

        grid_layout = QGridLayout()
        self.setLayout(grid_layout)

        # 1) 3D figure
        self.figure_3d = Figure()
        self.canvas_3d = FigureCanvas(self.figure_3d)
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)

        threeD_layout = QVBoxLayout()
        threeD_layout.addWidget(self.canvas_3d)
        threeD_layout.addWidget(self.toolbar_3d)
        threeD_widget = QWidget()
        threeD_widget.setLayout(threeD_layout)
        grid_layout.addWidget(threeD_widget, 0, 0)

        # 2) 2D figure (XY, XZ, YZ)
        self.figure_2d = Figure()
        self.canvas_2d = FigureCanvas(self.figure_2d)
        twoD_layout = QVBoxLayout()
        twoD_layout.addWidget(self.canvas_2d)
        twoD_widget = QWidget()
        twoD_widget.setLayout(twoD_layout)
        grid_layout.addWidget(twoD_widget, 0, 1)

        # 3) Bottom-left heatmap
        self.figure_btm_left = Figure()
        self.canvas_btm_left = FigureCanvas(self.figure_btm_left)
        grid_layout.addWidget(self.canvas_btm_left, 1, 0)

        # 4) Bottom-right heatmap
        self.figure_btm_right = Figure()
        self.canvas_btm_right = FigureCanvas(self.figure_btm_right)
        grid_layout.addWidget(self.canvas_btm_right, 1, 1)

        # Expand columns equally
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        # --------------------------------------------------------------------
        # Right Column: Top + Bottom Controls
        # --------------------------------------------------------------------
        controls_column = QVBoxLayout()

        # ----- Top Controls (for the 3D & top-2D plots) -----
        top_controls_layout = QVBoxLayout()
        top_controls_label = QLabel("Controls (Top Plots):")
        top_controls_layout.addWidget(top_controls_label)

        # (A) attribute_dropdown
        self.attribute_dropdown = QComboBox(self)
        self.attribute_dropdown.addItems(
            [
                'None',
                'highlighted points',
                'points_closest_to_archetypes',
                'points_between_archetypes',
            ]
            + list(self.attributes.columns)
        )
        self.attribute_dropdown.currentIndexChanged.connect(self.update_plot)
        top_controls_layout.addWidget(self.attribute_dropdown)

        # (B) archetype_selection
        self.archetype_selection = QComboBox(self)
        self.archetype_selection.setVisible(False)
        self.archetype_selection.currentIndexChanged.connect(self.update_plot)
        top_controls_layout.addWidget(self.archetype_selection)

        # (C) Toggle Lasso
        self.toggle_button = QPushButton("Toggle Lasso Selection", self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.toggle_lasso_selection)
        top_controls_layout.addWidget(self.toggle_button)

        # (D) Save points
        self.save_button = QPushButton("Save Selected Points", self)
        self.save_button.clicked.connect(self.save_selected_points)
        top_controls_layout.addWidget(self.save_button)

        # (E) Top slider + label
        slider_row = QHBoxLayout()
        self.percentage_slider = QSlider(Qt.Horizontal, self)
        self.percentage_slider.setRange(1, 50)
        self.percentage_slider.setValue(5)
        self.percentage_slider.setTickPosition(QSlider.NoTicks)
        self.percentage_slider.setVisible(False)
        self.percentage_slider.valueChanged.connect(self.update_percentage)
        slider_row.addWidget(self.percentage_slider)

        self.slider_value_label = QLabel(f"{self.percentage_slider.value()}%", self)
        self.slider_value_label.setVisible(False)
        slider_row.addWidget(self.slider_value_label)
        top_controls_layout.addLayout(slider_row)

        # (F) Bin spinbox (top)
        bins_row = QHBoxLayout()
        self.bins_label = QLabel("Num Bins:")
        self.bins_label.setVisible(False)
        self.bin_spinbox = QSpinBox(self)
        self.bin_spinbox.setRange(1, 50)
        self.bin_spinbox.setValue(2)
        self.bin_spinbox.setVisible(False)
        self.bin_spinbox.valueChanged.connect(self.update_plot)
        bins_row.addWidget(self.bins_label)
        bins_row.addWidget(self.bin_spinbox)
        top_controls_layout.addLayout(bins_row)

        top_controls_layout.addStretch()
        top_controls_widget = QWidget()
        top_controls_widget.setLayout(top_controls_layout)
        controls_column.addWidget(top_controls_widget)

        # --------------------------------------------------------------------
        # Bottom Plot Axes & Controls
        # --------------------------------------------------------------------
        self.x_left_dropdown = QComboBox(self)
        self.x_left_dropdown.addItems(["attribute", "archetype", "bin"])
        self.x_left_dropdown.setCurrentText("attribute")
        self.x_left_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)

        self.y_left_dropdown = QComboBox(self)
        self.y_left_dropdown.addItems(["attribute", "archetype", "bin"])
        self.y_left_dropdown.setCurrentText("archetype")
        self.y_left_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)

        self.x_right_dropdown = QComboBox(self)
        self.x_right_dropdown.addItems(["attribute", "archetype", "bin"])
        self.x_right_dropdown.setCurrentText("bin")
        self.x_right_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)

        self.y_right_dropdown = QComboBox(self)
        self.y_right_dropdown.addItems(["attribute", "archetype", "bin"])
        self.y_right_dropdown.setCurrentText("attribute")
        self.y_right_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)

        bottom_controls_layout = QVBoxLayout()
        bottom_controls_label = QLabel("Controls (Bottom Plots):")
        bottom_controls_layout.addWidget(bottom_controls_label)

        # Left Plot Axes
        left_plot_label = QLabel("Left Plot Axes:")
        bottom_controls_layout.addWidget(left_plot_label)
        bottom_controls_layout.addWidget(self.x_left_dropdown)
        bottom_controls_layout.addWidget(self.y_left_dropdown)

        # Right Plot Axes
        right_plot_label = QLabel("Right Plot Axes:")
        bottom_controls_layout.addWidget(right_plot_label)
        bottom_controls_layout.addWidget(self.x_right_dropdown)
        bottom_controls_layout.addWidget(self.y_right_dropdown)

        # BOTTOM LEFT
        self.bottom_left_attr_dropdown = QComboBox(self)
        self.bottom_left_attr_dropdown.addItems(
            ["None", "highlighted points", "points_closest_to_archetypes", "points_between_archetypes"]
            + list(self.attributes.columns)
        )
        self.bottom_left_attr_dropdown.currentIndexChanged.connect(self.update_btm_left_heatmap)
        bottom_controls_layout.addWidget(QLabel("Bottom-Left Color Attribute:"))
        bottom_controls_layout.addWidget(self.bottom_left_attr_dropdown)

        self.bottom_left_percentage_slider = QSlider(Qt.Horizontal, self)
        self.bottom_left_percentage_slider.setRange(1, 50)
        self.bottom_left_percentage_slider.setValue(5)
        self.bottom_left_percentage_slider.setTickPosition(QSlider.NoTicks)
        self.bottom_left_percentage_slider.valueChanged.connect(self.update_btm_left_heatmap)
        bottom_controls_layout.addWidget(self.bottom_left_percentage_slider)

        self.bottom_left_slider_label = QLabel("5%", self)
        bottom_controls_layout.addWidget(self.bottom_left_slider_label)

        self.bottom_left_bin_label = QLabel("Num Bins (Left):")
        bottom_controls_layout.addWidget(self.bottom_left_bin_label)

        self.bottom_left_bin_spinbox = QSpinBox(self)
        self.bottom_left_bin_spinbox.setRange(1, 50)
        self.bottom_left_bin_spinbox.setValue(2)
        self.bottom_left_bin_spinbox.valueChanged.connect(self.update_btm_left_heatmap)
        bottom_controls_layout.addWidget(self.bottom_left_bin_spinbox)

        # BOTTOM RIGHT
        self.bottom_right_attr_dropdown = QComboBox(self)
        self.bottom_right_attr_dropdown.addItems(
            ["None", "highlighted points", "points_closest_to_archetypes", "points_between_archetypes"]
            + list(self.attributes.columns)
        )
        self.bottom_right_attr_dropdown.currentIndexChanged.connect(self.update_btm_right_heatmap)
        bottom_controls_layout.addWidget(QLabel("Bottom-Right Color Attribute:"))
        bottom_controls_layout.addWidget(self.bottom_right_attr_dropdown)

        self.bottom_right_percentage_slider = QSlider(Qt.Horizontal, self)
        self.bottom_right_percentage_slider.setRange(1, 50)
        self.bottom_right_percentage_slider.setValue(5)
        self.bottom_right_percentage_slider.setTickPosition(QSlider.NoTicks)
        self.bottom_right_percentage_slider.valueChanged.connect(self.update_btm_right_heatmap)
        bottom_controls_layout.addWidget(self.bottom_right_percentage_slider)

        self.bottom_right_slider_label = QLabel("5%", self)
        bottom_controls_layout.addWidget(self.bottom_right_slider_label)

        self.bottom_right_bin_label = QLabel("Num Bins (Right):")
        bottom_controls_layout.addWidget(self.bottom_right_bin_label)

        self.bottom_right_bin_spinbox = QSpinBox(self)
        self.bottom_right_bin_spinbox.setRange(1, 50)
        self.bottom_right_bin_spinbox.setValue(2)
        self.bottom_right_bin_spinbox.valueChanged.connect(self.update_btm_right_heatmap)
        bottom_controls_layout.addWidget(self.bottom_right_bin_spinbox)

        bottom_controls_layout.addStretch()
        bottom_controls_widget = QWidget()
        bottom_controls_widget.setLayout(bottom_controls_layout)
        controls_column.addWidget(bottom_controls_widget)

        # Finalize the right column
        controls_widget = QWidget()
        controls_widget.setLayout(controls_column)
        grid_layout.addWidget(controls_widget, 0, 2, 2, 1)

        # Uniform widths
        button_width = max(
            self.toggle_button.sizeHint().width(),
            self.save_button.sizeHint().width()
        )
        # Original top widgets:
        for w in [
            self.attribute_dropdown,
            self.archetype_selection,
            self.toggle_button,
            self.save_button,
            self.percentage_slider,
            self.slider_value_label,
            self.bin_spinbox,
            self.bins_label,
            self.x_left_dropdown,
            self.y_left_dropdown,
            self.x_right_dropdown,
            self.y_right_dropdown,
        ]:
            w.setFixedWidth(button_width)

        # Bottom-left set
        for w in [
            self.bottom_left_attr_dropdown,
            self.bottom_left_percentage_slider,
            self.bottom_left_slider_label,
            self.bottom_left_bin_label,
            self.bottom_left_bin_spinbox,
        ]:
            w.setFixedWidth(button_width)

        # Bottom-right set
        for w in [
            self.bottom_right_attr_dropdown,
            self.bottom_right_percentage_slider,
            self.bottom_right_slider_label,
            self.bottom_right_bin_label,
            self.bottom_right_bin_spinbox,
        ]:
            w.setFixedWidth(button_width)

        # Initial plot (top)
        self.plot_embedding()

    # -------------------------------------------------------------------------
    #  SYNCHRONIZE HIGHLIGHTED POINTS
    # -------------------------------------------------------------------------
    def sync_highlighted_points(self):
        self.highlighted_indices = self.shared_state.selected_points
        if self.attribute_dropdown.currentText() == 'highlighted points':
            self.plot_embedding(color_attribute='highlighted points')

    # -------------------------------------------------------------------------
    #  MASTER UPDATE (TOP PLOTS)
    # -------------------------------------------------------------------------
    def update_plot(self):
        selected_attr = self.attribute_dropdown.currentText()

        if selected_attr in self.attributes.columns:
            self.last_valid_attribute = selected_attr

        if self.last_valid_attribute is None and len(self.attributes.columns) > 0:
            self.last_valid_attribute = self.attributes.columns[0]

        if selected_attr in ["points_between_archetypes", "points_closest_to_archetypes"]:
            self.percentage_slider.setVisible(True)
            self.slider_value_label.setVisible(True)
            self.bin_spinbox.setVisible(True)
            self.bins_label.setVisible(True)

            if selected_attr == "points_between_archetypes":
                self.archetype_selection.setVisible(True)
                if not self.archetype_selection.count():
                    self.populate_archetype_selection()
            else:
                self.archetype_selection.setVisible(False)
        else:
            self.percentage_slider.setVisible(False)
            self.slider_value_label.setVisible(False)
            self.bin_spinbox.setVisible(False)
            self.bins_label.setVisible(False)
            self.archetype_selection.setVisible(False)

        # Re-plot (top)
        self.plot_embedding(color_attribute=selected_attr)

    def update_percentage(self):
        val = self.percentage_slider.value()
        self.slider_value_label.setText(f"{val}%")
        if not self.slider_value_label.isVisible():
            self.slider_value_label.setVisible(True)
        self.alpha_other = val / 100
        self.plot_embedding(color_attribute=self.attribute_dropdown.currentText())

    def populate_archetype_selection(self):
        self.ensure_archetypes_computed()
        if self.archetype_method == "ConvexHull" and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == "SISAL" and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            arch_points = None

        if arch_points is None:
            return

        self.archetype_selection.clear()
        pairs = []
        for i in range(len(arch_points)):
            for j in range(i + 1, len(arch_points)):
                pairs.append(f"{i + 1} - {j + 1}")
        self.archetype_selection.addItems(pairs)

    # -------------------------------------------------------------------------
    #  TOP PLOT + THEN calls bottom heatmap updates
    # -------------------------------------------------------------------------
    def plot_embedding(self, color_attribute=None):
        # Original 3D axis logic
        if not self.figure_3d.axes:
            self.figure_3d.clear()
            self.ax = self.figure_3d.add_subplot(111, projection='3d')
        else:
            self.ax = self.figure_3d.axes[0]

        n_points = self.embeddings.shape[0]
        colors = np.array(["k"] * n_points)

        # 1) "highlighted points"
        if color_attribute == "highlighted points":
            colors = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other))
            for idx in self.shared_state.selected_points:
                if idx < n_points:
                    colors[idx] = (1, 0, 0, 1)

        elif color_attribute == "points_closest_to_archetypes":
            pass

        elif color_attribute == "points_between_archetypes":
            pass

        elif color_attribute and color_attribute != "None":
            import matplotlib
            vals = self.attributes[color_attribute]
            norm = matplotlib.colors.Normalize(vals.min(), vals.max())
            colors = plt.cm.viridis_r(norm(vals))

        # Always compute top X% if recognized (for top usage)
        self.compute_top_xpct_indices()  # modifies self.closest_indices_per_archetype

        # Points_closest_to_archetypes (color logic)
        if color_attribute == "points_closest_to_archetypes" and self.closest_indices_per_archetype:
            colors = self.cumulative_bins_closest_archetypes()

        # Points_between_archetypes
        elif color_attribute == "points_between_archetypes":
            self.ensure_archetypes_computed()
            arch_points = None
            if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
                arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
            elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
                arch_points = self.cached_sisal_hull.T

            if arch_points is not None:
                selected_index = self.archetype_selection.currentIndex()
                if selected_index != -1:
                    pairs = [
                        (i, j)
                        for i in range(len(arch_points))
                        for j in range(i + 1, len(arch_points))
                    ]
                    if selected_index < len(pairs):
                        a, b = pairs[selected_index]
                        arch_a = arch_points[a]
                        arch_b = arch_points[b]
                        n_intermediate = self.bin_spinbox.value()
                        dist = np.linalg.norm(
                            self.embeddings[:, None, :3]
                            - np.linspace(arch_a, arch_b, n_intermediate)[None, :, :],
                            axis=2,
                        )
                        x_pct = self.percentage_slider.value()
                        n_closest = max(1, int((x_pct / 100) * n_points))
                        closest_indices = np.argsort(dist, axis=0)[:n_closest]
                        cmap = plt.get_cmap('viridis')
                        icolors = cmap(np.linspace(0, 1, n_intermediate))
                        colarray = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other), dtype=float)
                        for i_col, idxs_i in enumerate(closest_indices.T):
                            colarray[idxs_i, :3] = icolors[i_col, :3]
                            colarray[idxs_i, 3] = 1.0
                        colors = colarray

        # Recompute or update standard top-level heatmaps
        self.means_matrix = self.compute_attribute_archetype_means()
        self.bin_matrix = self.compute_bin_attribute_means()

        # Now update both bottom heatmaps (still the old approach for top)
        self.update_btm_left_heatmap()
        self.update_btm_right_heatmap()

        # Draw 3D scatter
        self.ax.clear()
        self.scatter = self.ax.scatter(
            self._scaled_embeddings[:, 0],
            self._scaled_embeddings[:, 1],
            self._scaled_embeddings[:, 2],
            c=colors, marker='.'
        )
        self.ax.set_title(
            f"Session: {self.session_id}\n"
            f"Regions: {', '.join(sorted(self.brain_regions))}\n"
            f"Algorithm: {self.algorithm}",
            fontsize=12
        )
        self.ax.set_xlabel('Dimension 1')
        self.ax.set_ylabel('Dimension 2')
        self.ax.set_zlabel('Dimension 3')

        # 2D XY, XZ, YZ
        self.figure_2d.clear()
        self.ax_xy = self.figure_2d.add_subplot(311, aspect='equal')
        self.ax_xz = self.figure_2d.add_subplot(312, aspect='equal')
        self.ax_yz = self.figure_2d.add_subplot(313, aspect='equal')

        self.scatter_xy = self.ax_xy.scatter(
            self.embeddings[:, 0], self.embeddings[:, 1], c=colors, marker='.'
        )
        self.scatter_xz = self.ax_xz.scatter(
            self.embeddings[:, 0], self.embeddings[:, 2], c=colors, marker='.'
        )
        self.scatter_yz = self.ax_yz.scatter(
            self.embeddings[:, 1], self.embeddings[:, 2], c=colors, marker='.'
        )

        # Axis buffer
        for ax_obj, (xx, yy) in zip(
            [self.ax_xy, self.ax_xz, self.ax_yz],
            [
                (self.embeddings[:, 0], self.embeddings[:, 1]),
                (self.embeddings[:, 0], self.embeddings[:, 2]),
                (self.embeddings[:, 1], self.embeddings[:, 2])
            ]
        ):
            x_min, x_max = xx.min(), xx.max()
            y_min, y_max = yy.min(), yy.max()
            buffer_x = (x_max - x_min) * 0.2
            buffer_y = (y_max - y_min) * 0.2
            ax_obj.set_xlim(x_min - buffer_x, x_max + buffer_x)
            ax_obj.set_ylim(y_min - buffer_y, y_max + buffer_y)
            ax_obj.set_aspect('equal', adjustable='datalim')

        self.ax_xy.set_title("1-2 Dim")
        self.ax_xz.set_title("1-3 Dim")
        self.ax_yz.set_title("2-3 Dim")

        for ax_obj in [self.ax_xy, self.ax_xz, self.ax_yz]:
            ax_obj.spines['top'].set_visible(False)
            ax_obj.spines['right'].set_visible(False)
            ax_obj.spines['left'].set_visible(False)
            ax_obj.spines['bottom'].set_visible(False)
            ax_obj.set_xticks([])
            ax_obj.set_yticks([])

        self.canvas_2d.draw_idle()
        self.canvas_3d.draw_idle()

        # If hull lines
        if self.archetype_method == 'ConvexHull':
            self.add_convex_hull(self.ax, fig_type='3d')
            self.add_convex_hull(self.ax_xy, fig_type='2d', projection=(0, 1))
            self.add_convex_hull(self.ax_xz, fig_type='2d', projection=(0, 2))
            self.add_convex_hull(self.ax_yz, fig_type='2d', projection=(1, 2))
        if self.archetype_method == 'SISAL':
            self.add_sisal_hull(self.ax, fig_type='3d')
            self.add_sisal_hull(self.ax_xy, fig_type='2d', projection=(0, 1))
            self.add_sisal_hull(self.ax_xz, fig_type='2d', projection=(0, 2))
            self.add_sisal_hull(self.ax_yz, fig_type='2d', projection=(1, 2))

        # Lasso reinit
        if hasattr(self, 'selector') and self.selector and self.selector.selector_enabled:
            self.selector.toggle_selection_mode()
        if hasattr(self, 'lasso_xy') and self.lasso_xy and self.lasso_xy.selector_enabled:
            self.lasso_xy.toggle_selection_mode()
        if hasattr(self, 'lasso_xz') and self.lasso_xz and self.lasso_xz.selector_enabled:
            self.lasso_xz.toggle_selection_mode()
        if hasattr(self, 'lasso_yz') and self.lasso_yz and self.lasso_yz.selector_enabled:
            self.lasso_yz.toggle_selection_mode()

        self.selector = SelectFromCollectionLasso(self.ax, self.scatter, self.on_select_3d, is_3d=True)
        self.lasso_xy = SelectFromCollectionLasso(self.ax_xy, self.scatter_xy, self.on_select_2d)
        self.lasso_xz = SelectFromCollectionLasso(self.ax_xz, self.scatter_xz, self.on_select_2d)
        self.lasso_yz = SelectFromCollectionLasso(self.ax_yz, self.scatter_yz, self.on_select_2d)
        if self.lasso_active:
            self.selector.toggle_selection_mode()
            self.lasso_xy.toggle_selection_mode()
            self.lasso_xz.toggle_selection_mode()
            self.lasso_yz.toggle_selection_mode()
        
        
    def compute_bin_attribute_means(self):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None
    
        arch_a = arch_points[0]
        arch_b = arch_points[1]
        n_intermediate = self.bin_spinbox.value()
        inter_pts = np.linspace(arch_a, arch_b, n_intermediate)
    
        n_points = self.embeddings.shape[0]
        x_pct = self.percentage_slider.value()
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :], axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]
    
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, n_intermediate), dtype=float)
        for bin_i in range(n_intermediate):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_attribute_archetype_means(self):
        if not self.closest_indices_per_archetype:
            return None
        n_archetypes = len(self.closest_indices_per_archetype)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M
    
   
    
    # -------------------------------------------------------------------------
    #  BOTTOM-LEFT HEATMAP
    # -------------------------------------------------------------------------
    def update_btm_left_heatmap(self):
        """Recompute & redraw ONLY the bottom-left heatmap using the bottom-left slider/spinbox."""
        selected_attr = self.bottom_left_attr_dropdown.currentText()
        if selected_attr in self.attributes.columns:
            self.last_valid_attribute_left = selected_attr
        elif not self.last_valid_attribute_left and len(self.attributes.columns) > 0:
            self.last_valid_attribute_left = self.attributes.columns[0]

        # Grab bottom-left slider/spinbox
        x_pct = self.bottom_left_percentage_slider.value()
        self.bottom_left_slider_label.setText(f"{x_pct}%")
        n_bins = self.bottom_left_bin_spinbox.value()

        # 1) Compute a new set of 'closest_indices_per_archetype_left' based on x_pct
        self.compute_bottom_left_xpct_indices(x_pct)

        # 2) Build means_matrix_left / bin_matrix_left from that
        self.means_matrix_left = self.compute_attribute_archetype_means_bottom_left()
        self.bin_matrix_left = self.compute_bin_attribute_means_bottom_left(x_pct, n_bins)

        # Now do the bottom-left pcolormesh
        dim_x = self.x_left_dropdown.currentText()
        dim_y = self.y_left_dropdown.currentText()

        self.figure_btm_left.clear()
        ax = self.figure_btm_left.add_subplot(111)
        ax.set_title(f"Bottom-Left Heatmap ({selected_attr})", fontsize=10)

        matrix, x_labels, y_labels = self.get_heatmap_data_bottom_left(dim_x, dim_y)
        if matrix is None:
            self.canvas_btm_left.draw_idle()
            return

        pcm = ax.pcolormesh(
            matrix, cmap="viridis_r", edgecolors="k", linewidth=1, shading="flat"
        )
        cbar = self.figure_btm_left.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Value", rotation=270, labelpad=15)

        ax.set_xticks(np.arange(len(y_labels)) + 0.5)
        ax.set_xticklabels(y_labels)
        ax.set_yticks(np.arange(len(x_labels)) + 0.5)
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(dim_y.capitalize())
        ax.set_ylabel(dim_x.capitalize())
        ax.invert_yaxis()

        self.canvas_btm_left.draw_idle()

    # -------------------------------------------------------------------------
    #  BOTTOM-RIGHT HEATMAP
    # -------------------------------------------------------------------------
    def update_btm_right_heatmap(self):
        """Recompute & redraw ONLY the bottom-right heatmap using the bottom-right slider/spinbox."""
        selected_attr = self.bottom_right_attr_dropdown.currentText()
        if selected_attr in self.attributes.columns:
            self.last_valid_attribute_right = selected_attr
        elif not self.last_valid_attribute_right and len(self.attributes.columns) > 0:
            self.last_valid_attribute_right = self.attributes.columns[0]

        # Grab bottom-right slider/spinbox
        x_pct = self.bottom_right_percentage_slider.value()
        self.bottom_right_slider_label.setText(f"{x_pct}%")
        n_bins = self.bottom_right_bin_spinbox.value()

        # 1) Compute a new set of 'closest_indices_per_archetype_right' based on x_pct
        self.compute_bottom_right_xpct_indices(x_pct)

        # 2) Build means_matrix_right / bin_matrix_right
        self.means_matrix_right = self.compute_attribute_archetype_means_bottom_right()
        self.bin_matrix_right = self.compute_bin_attribute_means_bottom_right(x_pct, n_bins)

        # Now do the bottom-right pcolormesh
        dim_x = self.x_right_dropdown.currentText()
        dim_y = self.y_right_dropdown.currentText()

        self.figure_btm_right.clear()
        ax = self.figure_btm_right.add_subplot(111)
        ax.set_title(f"Bottom-Right Heatmap ({selected_attr})", fontsize=10)

        matrix, x_labels, y_labels = self.get_heatmap_data_bottom_right(dim_x, dim_y)
        if matrix is None:
            self.canvas_btm_right.draw_idle()
            return

        pcm = ax.pcolormesh(
            matrix, cmap="viridis_r", edgecolors="k", linewidth=1, shading="flat"
        )
        cbar = self.figure_btm_right.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Value", rotation=270, labelpad=15)

        ax.set_xticks(np.arange(len(y_labels)) + 0.5)
        ax.set_xticklabels(y_labels)
        ax.set_yticks(np.arange(len(x_labels)) + 0.5)
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(dim_y.capitalize())
        ax.set_ylabel(dim_x.capitalize())
        ax.invert_yaxis()

        self.canvas_btm_right.draw_idle()

    # -------------------------------------------------------------------------
    #  BOTTOM LEFT: get_heatmap_data_bottom_left
    # -------------------------------------------------------------------------
    def get_heatmap_data_bottom_left(self, dim_x, dim_y):
        """
        Same concept as get_heatmap_data, but uses self.means_matrix_left / self.bin_matrix_left
        and self.last_valid_attribute_left instead of top's variables.
        """
        combos = {}

        # 1) attribute x archetype
        if self.means_matrix_left is not None:
            n_archetypes = self.means_matrix_left.shape[1]
            arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "archetype")] = (self.means_matrix_left, attrs, arch_labels)
            combos[("archetype", "attribute")] = (self.means_matrix_left.T, arch_labels, attrs)

        # 2) attribute x bin
        if self.bin_matrix_left is not None:
            n_bins = self.bin_matrix_left.shape[1]
            bin_labels = [f"Bin {i+1}" for i in range(n_bins)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "bin")] = (self.bin_matrix_left, attrs, bin_labels)
            combos[("bin", "attribute")] = (self.bin_matrix_left.T, bin_labels, attrs)

        # 3) If user chooses (archetype, bin) or (bin, archetype), do bottom-left cumulative approach
        if (dim_x, dim_y) in [("archetype", "bin"), ("bin", "archetype")]:
            chosen_attr = self.last_valid_attribute_left
            if not chosen_attr or chosen_attr not in self.attributes.columns:
                chosen_attr = self.attributes.columns[0]  # fallback
            A = self.compute_archetype_bin_attribute_means_cumulative_bottom_left(chosen_attr)
            if A is not None:
                n_archetypes = A.shape[0]
                n_bins = A.shape[1]
                arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
                bin_labels = [f"Bin {j+1}" for j in range(n_bins)]
                if (dim_x, dim_y) == ("archetype", "bin"):
                    return A, arch_labels, bin_labels
                else:
                    return A.T, bin_labels, arch_labels
            return None, None, None

        return combos.get((dim_x, dim_y), (None, None, None))

    # -------------------------------------------------------------------------
    #  BOTTOM RIGHT: get_heatmap_data_bottom_right
    # -------------------------------------------------------------------------
    def get_heatmap_data_bottom_right(self, dim_x, dim_y):
        """
        Same concept as get_heatmap_data_bottom_left, but for right side.
        """
        combos = {}

        if self.means_matrix_right is not None:
            n_archetypes = self.means_matrix_right.shape[1]
            arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "archetype")] = (self.means_matrix_right, attrs, arch_labels)
            combos[("archetype", "attribute")] = (self.means_matrix_right.T, arch_labels, attrs)

        if self.bin_matrix_right is not None:
            n_bins = self.bin_matrix_right.shape[1]
            bin_labels = [f"Bin {i+1}" for i in range(n_bins)]
            attrs = list(self.attributes.columns)
            combos[("attribute", "bin")] = (self.bin_matrix_right, attrs, bin_labels)
            combos[("bin", "attribute")] = (self.bin_matrix_right.T, bin_labels, attrs)

        if (dim_x, dim_y) in [("archetype", "bin"), ("bin", "archetype")]:
            chosen_attr = self.last_valid_attribute_right
            if not chosen_attr or chosen_attr not in self.attributes.columns:
                chosen_attr = self.attributes.columns[0]
            A = self.compute_archetype_bin_attribute_means_cumulative_bottom_right(chosen_attr)
            if A is not None:
                n_archetypes = A.shape[0]
                n_bins = A.shape[1]
                arch_labels = [f"A{i+1}" for i in range(n_archetypes)]
                bin_labels = [f"Bin {j+1}" for j in range(n_bins)]
                if (dim_x, dim_y) == ("archetype", "bin"):
                    return A, arch_labels, bin_labels
                else:
                    return A.T, bin_labels, arch_labels
            return None, None, None

        return combos.get((dim_x, dim_y), (None, None, None))

    # -------------------------------------------------------------------------
    #  BOTTOM LEFT: separate compute logic
    # -------------------------------------------------------------------------
    def compute_bottom_left_xpct_indices(self, x_pct):
        """
        We'll define a new 'closest_indices_per_archetype_left' using x_pct
        and ignoring self.percentage_slider from the top.
        """
        self.closest_indices_per_archetype_left = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        arch_pts_3d = None
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        if arch_pts_3d is None or len(arch_pts_3d) == 0:
            return

        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )  # shape (n_points, n_archetypes)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        n_archetypes = arch_pts_3d.shape[0]
        self.closest_indices_per_archetype_left = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def compute_attribute_archetype_means_bottom_left(self):
        if not self.closest_indices_per_archetype_left:
            return None
        n_archetypes = len(self.closest_indices_per_archetype_left)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype_left):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        # row-wise normalize
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M

    def compute_bin_attribute_means_bottom_left(self, x_pct, n_bins):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None

        arch_a = arch_points[0]
        arch_b = arch_points[1]
        inter_pts = np.linspace(arch_a, arch_b, n_bins)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :], axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, n_bins), dtype=float)
        for bin_i in range(n_bins):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_archetype_bin_attribute_means_cumulative_bottom_left(self, chosen_attr):
        if self.closest_indices_per_archetype_left is None:
            return None
        if chosen_attr not in self.attributes.columns:
            return None

        n_archetypes = len(self.closest_indices_per_archetype_left)
        if n_archetypes == 0:
            return None

        n_points = self.embeddings.shape[0]
        x_pct = self.bottom_left_percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bottom_left_bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        M = np.full((n_archetypes, n_bins), np.nan, dtype=float)

        for j in range(n_archetypes):
            idxs_j = self.closest_indices_per_archetype_left[j]
            if len(idxs_j) == 0:
                continue
            # Distances for shading logic
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) > 0:
                    val = self.attributes.iloc[bin_points][chosen_attr].mean()
                    M[j, i_bin] = val
        return M

    # -------------------------------------------------------------------------
    #  BOTTOM RIGHT: separate compute logic
    # -------------------------------------------------------------------------
    def compute_bottom_right_xpct_indices(self, x_pct):
        self.closest_indices_per_archetype_right = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        arch_pts_3d = None
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        if arch_pts_3d is None or len(arch_pts_3d) == 0:
            return

        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )
        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        n_archetypes = arch_pts_3d.shape[0]
        self.closest_indices_per_archetype_right = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def compute_attribute_archetype_means_bottom_right(self):
        if not self.closest_indices_per_archetype_right:
            return None
        n_archetypes = len(self.closest_indices_per_archetype_right)
        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        M = np.zeros((n_attrs, n_archetypes), dtype=float)
        for j, idxs in enumerate(self.closest_indices_per_archetype_right):
            if len(idxs) == 0:
                continue
            subset_df = self.attributes.iloc[idxs]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                M[i, j] = col_means.get(col, 0.0)
        for i in range(n_attrs):
            row_min = M[i, :].min()
            row_max = M[i, :].max()
            denom = max(row_max - row_min, 1e-9)
            M[i, :] = (M[i, :] - row_min) / denom
        return M

    def compute_bin_attribute_means_bottom_right(self, x_pct, n_bins):
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return None
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_points = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points = self.cached_sisal_hull.T
        else:
            return None
        if arch_points.shape[0] < 2:
            return None

        arch_a = arch_points[0]
        arch_b = arch_points[1]
        inter_pts = np.linspace(arch_a, arch_b, n_bins)

        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        dist = np.linalg.norm(
            self.embeddings[:, None, :3] - inter_pts[None, :, :], axis=2
        )
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        attr_names = self.attributes.columns
        n_attrs = len(attr_names)
        bin_matrix = np.zeros((n_attrs, n_bins), dtype=float)
        for bin_i in range(n_bins):
            idxs_bin = closest_indices[:, bin_i]
            if len(idxs_bin) == 0:
                continue
            subset_df = self.attributes.iloc[idxs_bin]
            col_means = subset_df.mean(numeric_only=True)
            for i, col in enumerate(attr_names):
                bin_matrix[i, bin_i] = col_means.get(col, 0.0)
        return bin_matrix

    def compute_archetype_bin_attribute_means_cumulative_bottom_right(self, chosen_attr):
        if self.closest_indices_per_archetype_right is None:
            return None
        if chosen_attr not in self.attributes.columns:
            return None

        n_archetypes = len(self.closest_indices_per_archetype_right)
        if n_archetypes == 0:
            return None

        n_points = self.embeddings.shape[0]
        x_pct = self.bottom_right_percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bottom_right_bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        M = np.full((n_archetypes, n_bins), np.nan, dtype=float)

        for j in range(n_archetypes):
            idxs_j = self.closest_indices_per_archetype_right[j]
            if len(idxs_j) == 0:
                continue
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) > 0:
                    val = self.attributes.iloc[bin_points][chosen_attr].mean()
                    M[j, i_bin] = val
        return M

    # -------------------------------------------------------------------------
    #  ensure_archetypes_computed, hull drawing, lasso, etc. remain the same
    # -------------------------------------------------------------------------
    def ensure_archetypes_computed(self):
        if self.archetype_method == "ConvexHull" and self.cached_convex_hull is None:
            self.cached_convex_hull = ConvexHull(self.embeddings[:, :3])
        elif self.archetype_method == "SISAL" and self.cached_sisal_hull is None:
            ArchsMin, _ = findMinSimplex(10, self.embeddings, 1, 4)
            self.cached_sisal_hull = ArchsMin[:, np.argsort(ArchsMin[0, :])]

    # -------------------------------------------------------------------------
    #  CONVEX / SISAL hulls
    # -------------------------------------------------------------------------
    def add_convex_hull(self, ax, fig_type='3d', projection=None):
        try:
            if self.cached_convex_hull is None:
                self.cached_convex_hull = ConvexHull(self.embeddings[:, :3])
            hull = self.cached_convex_hull
            if fig_type == '3d':
                vertices_3d = self.embeddings[hull.vertices, :3]
                edges = []
                for simplex in hull.simplices:
                    edges.append(self.embeddings[simplex, :3])
                edge_collection = Line3DCollection(edges, colors='k', linewidths=0.5)
                ax.add_collection3d(edge_collection)
                ax.scatter(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2],
                           c='b', s=10, label='Archetypes')
                for i, (x, y, z) in enumerate(vertices_3d):
                    ax.text(x, y, z, str(i + 1), color='red', fontsize=10)
            elif fig_type == '2d':
                if projection is None:
                    return
                vertices_2d = self.embeddings[hull.vertices, projection]
                edges = []
                for simplex in hull.simplices:
                    edge = self.embeddings[simplex, projection]
                    edges.append(edge)
                    ax.plot(*edge.T, c='k', linewidth=0.5)
                ax.scatter(vertices_2d[:, 0], vertices_2d[:, 1],
                           c='b', s=10, label='Archetypes', zorder=5)
                for i, (xx, yy) in enumerate(vertices_2d):
                    ax.text(xx, yy, str(i + 1), color='red', fontsize=10, zorder=6)
                self.adjust_axis_limits(ax, self.embeddings[:, projection], vertices_2d)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error in adding ConvexHull: {e}')

    def add_sisal_hull(self, ax, fig_type='3d', projection=None):
        try:
            if self.cached_sisal_hull is None:
                ArchsMin, VolArchReal = findMinSimplex(10, self.embeddings, 1, 4)
                ArchsOrder = np.argsort(ArchsMin[0, :])
                self.cached_sisal_hull = ArchsMin[:, ArchsOrder]
            ArchsMin = self.cached_sisal_hull
            NArchetypes = ArchsMin.shape[1]
            if fig_type == '3d':
                ArchsMin_3d = ArchsMin[:3, :]
                edges = []
                for i in range(NArchetypes):
                    for j in range(i + 1, NArchetypes):
                        edge = np.array([ArchsMin_3d[:, i], ArchsMin_3d[:, j]])
                        edges.append(edge)
                edge_collection = Line3DCollection(edges, colors='k', linewidths=0.5)
                ax.add_collection3d(edge_collection)
                ax.scatter(ArchsMin_3d[0, :], ArchsMin_3d[1, :], ArchsMin_3d[2, :],
                           c='b', s=10, label='Archetypes')
                for i, (x, y, z) in enumerate(ArchsMin_3d.T):
                    ax.text(x, y, z, str(i + 1), color='red', fontsize=10)
            elif fig_type == '2d':
                if projection is None:
                    return
                ArchsMin_2d = ArchsMin[list(projection), :]
                edges = []
                for i in range(NArchetypes):
                    for j in range(i + 1, NArchetypes):
                        edge = np.array([ArchsMin_2d[:, i], ArchsMin_2d[:, j]])
                        edges.append(edge)
                        ax.plot(*edge.T, c='k', linewidth=0.5)
                ax.scatter(ArchsMin_2d[0, :], ArchsMin_2d[1, :],
                           c='b', s=10, label='Archetypes', zorder=5)
                for i, (xx, yy) in enumerate(ArchsMin_2d.T):
                    ax.text(xx, yy, str(i + 1), color='red', fontsize=10, zorder=6)
                scatter_points = self.embeddings[:, projection]
                self.adjust_axis_limits(ax, scatter_points, ArchsMin_2d.T)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error in adding SISAL hull: {e}')

    def adjust_axis_limits(self, ax, scatter_points, polytope_points):
        all_points = np.vstack([scatter_points, polytope_points])
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        buffer_x = (x_max - x_min) * 0.1
        buffer_y = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
        ax.set_ylim(y_min - buffer_y, y_max + buffer_y)

    def invalidate_cache(self):
        self.cached_convex_hull = None
        self.cached_sisal_hull = None

    # -------------------------------------------------------------------------
    #  LASSO
    # -------------------------------------------------------------------------
    def on_select_3d(self, indices):
        self.highlighted_indices = list(indices)

    def on_select_2d(self, indices):
        self.highlighted_indices = list(indices)

    def toggle_lasso_selection(self):
        if self.lasso_active:
            self.lasso_active = False
            if self.selector and self.selector.selector_enabled:
                self.selector.toggle_selection_mode()
            if self.lasso_xy and self.lasso_xy.selector_enabled:
                self.lasso_xy.toggle_selection_mode()
            if self.lasso_xz and self.lasso_xz.selector_enabled:
                self.lasso_xz.toggle_selection_mode()
            if self.lasso_yz and self.lasso_yz.selector_enabled:
                self.lasso_yz.toggle_selection_mode()
            self.canvas_3d.draw_idle()
            self.canvas_2d.draw_idle()
        else:
            self.lasso_active = True
            if self.selector:
                self.selector.toggle_selection_mode()
            if self.lasso_xy:
                self.lasso_xy.toggle_selection_mode()
            if self.lasso_xz:
                self.lasso_xz.toggle_selection_mode()
            if self.lasso_yz:
                self.lasso_yz.toggle_selection_mode()

    def save_selected_points(self):
        self.selected_points = list(self.highlighted_indices)
        self.shared_state.set_selected_points(self.selected_points)
        print(f"Selected points saved: {self.selected_points}")
        if self.attribute_dropdown.currentText() == 'highlighted points':
            self.plot_embedding(color_attribute='highlighted points')

    # -------------------------------------------------------------------------
    #  "points_closest_to_archetypes" color logic
    # -------------------------------------------------------------------------
    def compute_top_xpct_indices(self):
        """
        For each archetype j, define top X% by distance => closest_indices_per_archetype[j].
        We do this for potential usage in color or in the heatmaps.
        """
        self.closest_indices_per_archetype = []
        if self.archetype_method not in ["ConvexHull", "SISAL"]:
            return
        self.ensure_archetypes_computed()
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull:
            arch_pts_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_pts_3d = self.cached_sisal_hull.T[:, :3]
        else:
            return
        n_archetypes = arch_pts_3d.shape[0]
        if n_archetypes == 0:
            return
        dist = np.linalg.norm(
            self.embeddings[:, :3, None] - arch_pts_3d.T[None, :, :],
            axis=1
        )  # shape (n_points, n_archetypes)

        x_pct = self.percentage_slider.value()
        n_points = self.embeddings.shape[0]
        n_closest = max(1, int((x_pct / 100) * n_points))
        closest_indices = np.argsort(dist, axis=0)[:n_closest]

        self.closest_indices_per_archetype = [
            closest_indices[:, col] for col in range(n_archetypes)
        ]

    def cumulative_bins_closest_archetypes(self):
        """
        This sets the scatter color per chunk for each archetype in 'points_closest_to_archetypes'.
        Returns RGBA array of shape (n_points, 4).
        """
        import matplotlib
        import colorsys
        n_points = self.embeddings.shape[0]
        colors = np.full((n_points, 4), (0.5, 0.5, 0.5, self.alpha_other), dtype=float)

        n_archetypes = len(self.closest_indices_per_archetype)
        if n_archetypes == 0:
            return colors

        base_cmap = matplotlib.colormaps.get_cmap("viridis")
        base_colors = base_cmap(np.linspace(0, 1, n_archetypes))

        x_pct = self.percentage_slider.value()
        chunk_size = int((x_pct / 100) * n_points)
        n_bins = self.bin_spinbox.value()
        max_points = min(n_points, n_bins * chunk_size)

        for j, idxs_j in enumerate(self.closest_indices_per_archetype):
            if len(idxs_j) == 0:
                continue
            arch_coords = self.get_archetype_coords(j)
            dist_j = np.linalg.norm(self.embeddings[:, :3] - arch_coords, axis=1)
            sorted_all = np.argsort(dist_j)

            base_r, base_g, base_b, _ = base_colors[j]
            h, l, s = colorsys.rgb_to_hls(base_r, base_g, base_b)

            min_lightness = 0.3
            if l < min_lightness:
                l = min_lightness

            for i_bin in range(n_bins):
                start = i_bin * chunk_size
                end = (i_bin + 1) * chunk_size
                if start >= max_points:
                    break
                if end > max_points:
                    end = max_points
                bin_points = sorted_all[start:end]
                if len(bin_points) == 0:
                    break

                fraction = (i_bin + 1) / (n_bins + 1)
                lightness_increment = fraction * 0.3
                new_l = min(l + lightness_increment, 1.0)

                shaded_r, shaded_g, shaded_b = colorsys.hls_to_rgb(h, new_l, s)
                colors[bin_points, :3] = (shaded_r, shaded_g, shaded_b)
                colors[bin_points, 3] = 1.0

        return colors

    def get_archetype_coords(self, j):
        """Return the j-th archetype's 3D coords from either convex hull or SISAL."""
        if self.archetype_method == 'ConvexHull' and self.cached_convex_hull is not None:
            arch_points_3d = self.embeddings[self.cached_convex_hull.vertices, :3]
            if j < len(arch_points_3d):
                return arch_points_3d[j]
        elif self.archetype_method == 'SISAL' and self.cached_sisal_hull is not None:
            arch_points_3d = self.cached_sisal_hull.T[:, :3]
            if j < len(arch_points_3d):
                return arch_points_3d[j]
        return np.array([0, 0, 0], dtype=float)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = AnalysisGUI()
    main_window.show()
    sys.exit(app.exec_())