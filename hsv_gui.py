import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QFrame, QPushButton
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor


# ──────────────────────────────────────────────
# Helper: numpy BGR → QPixmap
# ──────────────────────────────────────────────
def bgr_to_pixmap(img, target_w=None, target_h=None):
    if img is None:
        return QPixmap()
    if len(img.shape) == 2:                          # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if target_w and target_h:
        img = cv2.resize(img, (target_w, target_h))
    h, w, ch = img.shape
    qimg = QImage(img.data, w, h, ch * w, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg)


# ──────────────────────────────────────────────
# Labeled slider widget
# ──────────────────────────────────────────────
class LabeledSlider(QWidget):
    def __init__(self, label, lo, hi, val, suffix="", scale=1, parent=None):
        super().__init__(parent)
        self.scale = scale
        self.suffix = suffix

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(8)

        lbl = QLabel(label)
        lbl.setFixedWidth(110)
        lbl.setStyleSheet("color:#a0b4c8; font-size:11px;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(val)
        self.slider.setFixedHeight(18)

        self.val_lbl = QLabel(f"{val / scale:.2f}{suffix}")
        self.val_lbl.setFixedWidth(52)
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val_lbl.setStyleSheet("color:#e0eaf4; font-size:11px; font-weight:600;")

        self.slider.valueChanged.connect(self._on_change)

        row.addWidget(lbl)
        row.addWidget(self.slider)
        row.addWidget(self.val_lbl)

    def _on_change(self, v):
        self.val_lbl.setText(f"{v / self.scale:.2f}{self.suffix}")

    def value(self):
        return self.slider.value() / self.scale


# ──────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────
class GrassGUI(QMainWindow):
    PANEL_W = 320
    PANEL_H = 240

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌿 Grass Coverage Analyzer")
        self.setStyleSheet(DARK_STYLE)

        # ── video ──
        self.cap = cv2.VideoCapture("grass.mp4")
        self.smooth_coverage = 0.0
        self.smooth_spread   = 0.0

        # ── build UI ──
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # title bar
        title = QLabel("🌿  Grass Coverage Analyzer")
        title.setStyleSheet("font-size:16px; font-weight:700; color:#5dde8a; letter-spacing:1px;")
        root.addWidget(title)

        body = QHBoxLayout()
        root.addLayout(body)

        # ── left: 4 video panels ──
        panels_box = QGroupBox("Video Feeds")
        panels_box.setStyleSheet(GROUP_STYLE)
        grid = QGridLayout(panels_box)
        grid.setSpacing(6)

        def make_panel(label_text):
            frame = QFrame()
            frame.setStyleSheet("background:#0a1520; border:1px solid #1e3a52; border-radius:4px;")
            vl = QVBoxLayout(frame)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(0)
            cap_lbl = QLabel(label_text)
            cap_lbl.setAlignment(Qt.AlignCenter)
            cap_lbl.setStyleSheet("color:#4a7a9b; font-size:10px; padding:2px; background:#0d1e2e;")
            img_lbl = QLabel()
            img_lbl.setFixedSize(self.PANEL_W, self.PANEL_H)
            img_lbl.setAlignment(Qt.AlignCenter)
            vl.addWidget(cap_lbl)
            vl.addWidget(img_lbl)
            return frame, img_lbl

        f1, self.lbl_orig   = make_panel("RAW VIDEO")
        f2, self.lbl_binary = make_panel("BINARY MASK")
        f3, self.lbl_mask   = make_panel("SEGMENTED OVERLAY")
        f4, self.lbl_roi    = make_panel("ROI ONLY")

        grid.addWidget(f1, 0, 0)
        grid.addWidget(f2, 0, 1)
        grid.addWidget(f3, 1, 0)
        grid.addWidget(f4, 1, 1)
        body.addWidget(panels_box)

        # ── right: controls + metrics ──
        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        body.addLayout(right_col)

        # parameters
        param_box = QGroupBox("Parameters")
        param_box.setStyleSheet(GROUP_STYLE)
        param_box.setFixedWidth(340)
        pv = QVBoxLayout(param_box)
        pv.setSpacing(4)

        self.sl_alpha    = LabeledSlider("Smoothing α",    1, 99,  30, suffix="", scale=100)
        self.sl_min_area = LabeledSlider("Min Contour",   50, 2000, 300, suffix=" px", scale=1)
        self.sl_roi_top  = LabeledSlider("ROI Top %",      0,  80,  30, suffix="%", scale=1)
        self.sl_blur     = LabeledSlider("Blur Kernel",    1,  15,   5, suffix="", scale=1)
        self.sl_h_lo     = LabeledSlider("Hue Low",        0, 179,  35, suffix="°", scale=1)
        self.sl_h_hi     = LabeledSlider("Hue High",       0, 179,  85, suffix="°", scale=1)
        self.sl_s_lo     = LabeledSlider("Sat Low",        0, 255,  40, suffix="", scale=1)
        self.sl_v_lo     = LabeledSlider("Val Low",        0, 255,  40, suffix="", scale=1)
        self.sl_morph    = LabeledSlider("Morph Kernel",   1,  15,   5, suffix="", scale=1)

        for sl in [self.sl_alpha, self.sl_min_area, self.sl_roi_top,
                   self.sl_blur, self.sl_h_lo, self.sl_h_hi,
                   self.sl_s_lo, self.sl_v_lo, self.sl_morph]:
            pv.addWidget(sl)

        right_col.addWidget(param_box)

        # metrics
        met_box = QGroupBox("Live Metrics")
        met_box.setStyleSheet(GROUP_STYLE)
        met_box.setFixedWidth(340)
        mv = QVBoxLayout(met_box)
        mv.setSpacing(8)

        self.m_coverage = self._metric_label("Coverage", "0.00")
        self.m_spread   = self._metric_label("Spread",   "0.00")
        self.m_pump     = self._metric_label("Pump PWM", "30")
        self.m_nozzle   = self._metric_label("Nozzle",   "0%")

        for m in [self.m_coverage, self.m_spread, self.m_pump, self.m_nozzle]:
            mv.addLayout(m["row"])

        right_col.addWidget(met_box)

        # control buttons
        btn_row = QHBoxLayout()
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.setStyleSheet(BTN_STYLE)
        self.btn_pause.toggled.connect(self._toggle_pause)

        btn_reset = QPushButton("↺  Reset Smooth")
        btn_reset.setStyleSheet(BTN_STYLE)
        btn_reset.clicked.connect(self._reset_smooth)

        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(btn_reset)
        right_col.addLayout(btn_row)
        right_col.addStretch()

        # ── timer ──
        self.paused = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)

        self.adjustSize()

    # ── metric row builder ──
    def _metric_label(self, name, init):
        row_layout = QHBoxLayout()
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet("color:#6a8faa; font-size:11px;")
        val_lbl = QLabel(init)
        val_lbl.setStyleSheet("color:#5dde8a; font-size:14px; font-weight:700;")
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row_layout.addWidget(name_lbl)
        row_layout.addWidget(val_lbl)
        return {"row": row_layout, "val": val_lbl}

    def _toggle_pause(self, checked):
        self.paused = checked
        self.btn_pause.setText("▶  Resume" if checked else "⏸  Pause")

    def _reset_smooth(self):
        self.smooth_coverage = 0.0
        self.smooth_spread   = 0.0

    # ── main processing loop ──
    def process_frame(self):
        if self.paused:
            return
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame = cv2.resize(frame, (self.PANEL_W, self.PANEL_H))
        orig  = frame.copy()

        # ── read params ──
        alpha    = self.sl_alpha.value()
        min_area = int(self.sl_min_area.value())
        roi_pct  = int(self.sl_roi_top.value())
        blur_k   = int(self.sl_blur.value()) | 1   # must be odd
        h_lo     = int(self.sl_h_lo.value())
        h_hi     = int(self.sl_h_hi.value())
        s_lo     = int(self.sl_s_lo.value())
        v_lo     = int(self.sl_v_lo.value())
        morph_k  = int(self.sl_morph.value())

        # ── preprocess ──
        blur = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
        hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        v_mean = np.mean(hsv[:, :, 2])

        # adaptive HSV bounds (blend slider with auto-brightness logic)
        if v_mean < 80:
            lower = np.array([max(h_lo - 5, 0), max(s_lo - 10, 0), 30])
            upper = np.array([min(h_hi + 5, 179), 255, 255])
        elif v_mean > 180:
            lower = np.array([min(h_lo + 5, 179), min(s_lo + 20, 255), 60])
            upper = np.array([min(h_hi, 179), 255, 255])
        else:
            lower = np.array([h_lo, s_lo, v_lo])
            upper = np.array([h_hi, 255, 255])

        # ── segmentation ──
        mask = cv2.inRange(hsv, lower, upper)

        # ── morphology ──
        kernel = np.ones((morph_k, morph_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # ── contour filter ──
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
        mask = clean_mask

        # ── binary panel (white on black) ──
        binary_vis = mask.copy()

        # ── ROI ──
        h, w = mask.shape
        roi_y = int(h * roi_pct / 100)
        roi   = mask[roi_y:h, :]

        # ── ROI panel (full frame, grey out excluded top) ──
        roi_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if roi_y > 0:
            roi_vis[:roi_y, :] = (roi_vis[:roi_y, :] * 0.15).astype(np.uint8)
        cv2.rectangle(roi_vis, (0, roi_y), (w-1, h-1), (0, 100, 255), 1)

        # ── coverage & spread ──
        total_px  = roi.size
        grass_px  = cv2.countNonZero(roi)
        coverage  = grass_px / total_px if total_px > 0 else 0

        coords = np.column_stack(np.where(roi > 0))
        if coords.size > 0:
            x_coords   = coords[:, 1]
            spread_px  = x_coords.max() - x_coords.min()
            spread_ratio = spread_px / w
        else:
            spread_ratio = 0

        # ── smoothing ──
        self.smooth_coverage = alpha * coverage      + (1 - alpha) * self.smooth_coverage
        self.smooth_spread   = alpha * spread_ratio  + (1 - alpha) * self.smooth_spread

        # ── overlay panel ──
        overlay = orig.copy()
        overlay[mask > 0] = (0, 220, 80)
        display = cv2.addWeighted(orig, 0.65, overlay, 0.35, 0)

        # draw spread line
        if coords.size > 0:
            xmin = int(x_coords.min())
            xmax = int(x_coords.max())
            ymid = roi_y + (h - roi_y) // 2
            cv2.line(display, (xmin, ymid), (xmax, ymid), (0, 60, 255), 2)
            cv2.circle(display, (xmin, ymid), 3, (0, 60, 255), -1)
            cv2.circle(display, (xmax, ymid), 3, (0, 60, 255), -1)

        # ── control outputs ──
        pump_pwm    = int(30 + 70 * self.smooth_coverage)   # 30–100
        nozzle_pct  = int(self.smooth_spread * 100)          # 0–100 %

        # ── update panels ──
        self.lbl_orig.setPixmap(bgr_to_pixmap(orig, self.PANEL_W, self.PANEL_H))
        self.lbl_binary.setPixmap(bgr_to_pixmap(binary_vis, self.PANEL_W, self.PANEL_H))
        self.lbl_mask.setPixmap(bgr_to_pixmap(display, self.PANEL_W, self.PANEL_H))
        self.lbl_roi.setPixmap(bgr_to_pixmap(roi_vis, self.PANEL_W, self.PANEL_H))

        # ── update metrics ──
        self.m_coverage["val"].setText(f"{self.smooth_coverage:.3f}")
        self.m_spread["val"].setText(f"{self.smooth_spread:.3f}")
        self.m_pump["val"].setText(str(pump_pwm))
        self.m_nozzle["val"].setText(f"{nozzle_pct}%")

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


# ──────────────────────────────────────────────
# Style sheets
# ──────────────────────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0b1a27;
    color: #c8dce8;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #1e3a52;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #5dde8a;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}
QSlider::sub-page:horizontal {
    background: #2a7a4f;
    border-radius: 2px;
}
"""

GROUP_STYLE = """
QGroupBox {
    border: 1px solid #1e3a52;
    border-radius: 6px;
    margin-top: 14px;
    padding: 6px;
    color: #4a9aba;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
"""

BTN_STYLE = """
QPushButton {
    background: #112233;
    border: 1px solid #2a5070;
    color: #7ab8d4;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton:hover  { background: #1a3348; border-color: #5dde8a; color: #5dde8a; }
QPushButton:checked { background: #1a2e1a; border-color: #5dde8a; color: #5dde8a; }
"""


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GrassGUI()
    win.show()
    sys.exit(app.exec_())