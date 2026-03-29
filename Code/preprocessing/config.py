from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Constantes globales

TARGET_HEIGHT: int = 32   # altura fija de todas las líneas normalizadas (px)
PAD_VALUE:   float = 1.0  # valor de relleno en [0,1] (fondo blanco)
MIN_WIDTH:     int = 16   # ancho mínimo para evitar imágenes degeneradas


# Métricas de imagen (resultado de analyze())

@dataclass
class ImageMetrics:
    contrast:          float
    mean_luminance:    float
    dark_background:   bool
    estimated_text_h:  float
    estimated_n_lines: int
    needs_clahe:       bool
    needs_bilateral:   bool
    best_channel:      str
    H:                 int
    W:                 int


# Configuración del pipeline

@dataclass
class PipelineConfig:
    # Binarización
    sauvola_window:   int   = 51
    sauvola_k:        float = 0.25
    morph_open:       int   = 0
    morph_close:      int   = 0
    use_clahe:        bool  = False
    clahe_clip:       float = 3.0
    clahe_tile:       int   = 16
    use_blue_channel: bool  = False
    invert_binary:    bool  = False
    use_bilateral:    bool  = False
    bilateral_d:      int   = 9
    bilateral_sc:     float = 75.0
    bilateral_ss:     float = 75.0

    # Deskew global
    deskew:           bool  = True
    deskew_max_angle: float = 15.0

    # Deskew por bloque
    deskew_blocks:          bool  = False
    deskew_block_max_angle: float = 15.0

    # Segmentación de líneas
    min_line_height:   int  = 20
    min_line_width:    int  = 20
    line_merge_gap:    int  = 4
    projection_smooth: int  = 3
    use_savgol:        bool = False
    savgol_window:     int  = 25
    savgol_polyorder:  int  = 3
    line_h_dilation:   int  = 0
    line_v_dilation:   int  = 0

    # Expansión de caja
    expand_to_ink:       bool  = True
    expand_max_frac:     float = 0.80
    expand_no_ink_gap:   int   = 8
    expand_min_ink_frac: float = 0.003

    # Enderezado per-línea
    straighten_lines:  bool = True
    straighten_poly:   int  = 2
    straighten_slices: int  = 0

    # Normalización
    target_height: int = TARGET_HEIGHT
    trim_margin:   int = 2

    # Detección de bloques
    detect_text_blocks:  bool  = True
    block_col_min_depth: float = 0.28
    block_min_h_gap:     int   = 0
    block_h_thr_frac:    float = 0.04
    block_max_cols:      int   = 4
    header_zone_frac:    float = 1.0 / 6.0

    # Filtrado post-binarización
    global_floor_pct:              float = 0.0
    min_component_area:            int   = 0
    use_adaptive_component_filter: bool  = False
    use_remove_bg:                 bool  = False

    # Fusión de bloques
    para_split_factor: float = 2.5
    block_merge_gap:   int   = 0

    # Control
    debug:     bool = False
    debug_dir: str  = "debug_pipeline"


# Resultado del pipeline

@dataclass
class PipelineResult:
    lines:       list[np.ndarray]
    line_boxes:  list[tuple[int, int, int, int]]
    block_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    binary:      Optional[np.ndarray]            = None
    n_lines:     int                             = 0
    warnings:    list[str]                       = field(default_factory=list)
    config_used: Optional[PipelineConfig]        = None

    def __post_init__(self):
        self.n_lines = len(self.lines)