from dataclasses import dataclass, field
from typing import Optional

import numpy as np


TARGET_HEIGHT: int = 64    # altura fija de normalizacion (divisible entre 16)
PAD_VALUE:   float = 1.0   # fondo blanco en espacio [0,1]
MIN_WIDTH:     int = 16    # ancho minimo para evitar tensores degenerados


@dataclass
class ImageMetrics:
    # Metricas calculadas sobre la imagen completa, usadas por auto_config.
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


@dataclass
class PipelineConfig:
    # Configuracion completa del pipeline multi-linea.

    # Binarizacion
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
    deskew_blocks:             bool  = False
    deskew_block_max_angle:    float = 15.0
    block_min_h_for_skew:      int   = 60
    block_residual_threshold:  float = 0.5

    # Segmentacion de lineas
    min_line_height:   int  = 14
    min_line_width:    int  = 20
    line_merge_gap:    int  = 1
    projection_smooth: int  = 3
    use_savgol:        bool = False
    savgol_window:     int  = 25
    savgol_polyorder:  int  = 3
    line_h_dilation:   int  = 0
    line_v_dilation:   int  = 0

    # Expansion de caja
    expand_to_ink:       bool  = True
    expand_max_frac:     float = 0.80
    expand_no_ink_gap:   int   = 6
    expand_min_ink_frac: float = 0.002

    # Enderezado per-linea
    straighten_lines:  bool = True
    straighten_poly:   int  = 2
    straighten_slices: int  = 0

    # Normalizacion
    target_height: int = TARGET_HEIGHT
    trim_margin:   int = 2

    # Enmascaramiento de encuadernacion / borde del libro: pinta a blanco
    # las franjas oscuras laterales antes de detectar lineas.
    mask_binding:        bool  = True
    binding_max_frac:    float = 0.15
    binding_density_thr: float = 0.30

    # Limpieza per-linea: elimina componentes cuyo centroide Y cae fuera de
    # la banda principal del strip (descenders/ascenders de lineas vecinas).
    trim_orphans_per_line: bool = True

    # Deteccion de bloques
    detect_text_blocks:  bool  = True
    block_col_min_depth: float = 0.50
    block_min_h_gap:     int   = 0
    block_h_thr_frac:    float = 0.04
    block_max_cols:      int   = 4
    header_zone_frac:    float = 1.0 / 6.0

    # Filtrado post-binarizacion
    global_floor_pct:              float = 0.0
    min_component_area:            int   = 0
    use_adaptive_component_filter: bool  = False
    use_remove_bg:                 bool  = False
    use_oriented_crop:             bool  = True

    # Fusion de bloques
    para_split_factor: float = 2.5
    block_merge_gap:   int   = 0

    # Debug
    debug:     bool = False
    debug_dir: str  = "debug_pipeline"


@dataclass
class PipelineResult:
    # Salida del pipeline completo: lineas normalizadas, crops uint8 y bboxes.
    lines:          list[np.ndarray]
    line_boxes:     list[tuple[int, int, int, int]]
    # Crops uint8 de cada linea ya procesada (rotada, straightened y limpiada
    # de tinta huerfana de lineas adyacentes). Listos para guardar como JPG.
    line_crops:     list[np.ndarray]                            = field(default_factory=list)
    block_boxes:    list[tuple[int, int, int, int]]             = field(default_factory=list)
    oriented_boxes: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    binary:         Optional[np.ndarray]                        = None
    deskew_angle:   float                                       = 0.0
    n_lines:        int                                         = 0
    warnings:       list[str]                                   = field(default_factory=list)
    config_used:    Optional[PipelineConfig]                    = None

    def __post_init__(self):
        # Sincroniza n_lines con la longitud real de la lista.
        self.n_lines = len(self.lines)
