"""WSI preview, sampling preview, and lightweight doctor annotation widgets."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pathodataforge.app.components import Card
from pathodataforge.app.state import AppState
from pathodataforge.core.sampling import build_sampling_overlay, sample_slide_coordinates
from pathodataforge.core.replay_verifier import (
    bind_annotation_relations,
    build_annotation_provenance_payload,
)
from pathodataforge.core.wsi_reader import WSIInfo, WSIReader


class AnnotationCanvas(QWidget):
    """Draw level-0 coordinate strokes over the currently rendered WSI view."""

    view_bounds_requested = Signal(object)
    zoom_changed = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("OverlayCanvas")
        self.setMinimumSize(620, 430)
        self.base_pixmap: QPixmap | None = None
        self.slide_size = (0, 0)
        self.view_bounds_level0 = (0, 0, 0, 0)
        self.strokes: list[dict[str, Any]] = []
        self.current_stroke: dict[str, Any] | None = None
        self.mode = "pan"
        self.brush_width = 10
        self.brush_color = QColor("#16A34A")
        self._pan_start: QPoint | None = None
        self._pan_origin_bounds = (0, 0, 0, 0)
        self._pan_offset = QPoint(0, 0)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_rendered_view(
        self,
        pixmap: QPixmap,
        slide_size: tuple[int, int],
        view_bounds_level0: tuple[int, int, int, int] | None = None,
        clear_annotations: bool = False,
    ) -> None:
        self.base_pixmap = pixmap
        self.slide_size = slide_size
        self.view_bounds_level0 = view_bounds_level0 or (0, 0, slide_size[0], slide_size[1])
        self._pan_start = None
        self._pan_offset = QPoint(0, 0)
        if clear_annotations:
            self.strokes.clear()
            self.current_stroke = None
        self._emit_zoom()
        self.update()

    def reset_view(self) -> None:
        if not self.slide_size[0] or not self.slide_size[1]:
            return
        self._pan_start = None
        self._pan_offset = QPoint(0, 0)
        self.view_bounds_requested.emit((0, 0, self.slide_size[0], self.slide_size[1]))

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif mode == "erase":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def set_brush_width(self, width: int) -> None:
        self.brush_width = max(1, int(width))

    def undo(self) -> None:
        if self.strokes:
            self.strokes.pop()
            self.update()

    def clear_annotations(self) -> None:
        self.strokes.clear()
        self.current_stroke = None
        self.update()

    def export_annotations(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for stroke in self.strokes:
            result.append(
                {
                    "label": stroke.get("label", "high_risk"),
                    "tool": stroke.get("tool", "brush"),
                    "color": stroke.get("color", "#16A34A"),
                    "width": stroke.get("width", self.brush_width),
                    "points_level0": stroke.get("points_level0", []),
                }
            )
        return result

    def save_composited_png(self, path: str | Path) -> None:
        if self.base_pixmap is None:
            return
        target = QPixmap(self.base_pixmap.size())
        target.fill(Qt.GlobalColor.transparent)
        painter = QPainter(target)
        painter.drawPixmap(0, 0, self.base_pixmap)
        self._draw_strokes(painter, QRect(0, 0, self.base_pixmap.width(), self.base_pixmap.height()))
        painter.end()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        target.save(str(path))

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#F8FAFC"))
        if self.base_pixmap is None:
            painter.setPen(QColor("#64748B"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "选择一张 WSI 后，这里会显示便于医生查看的预览图。\n滚轮缩放，左键拖动浏览，点击画笔后可标注高风险区域。",
            )
            painter.end()
            return
        target = self._image_rect()
        painter.drawPixmap(target, self.base_pixmap)
        self._draw_strokes(painter, target)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.base_pixmap is None or event.button() != Qt.MouseButton.LeftButton:
            return
        if self.mode == "pan":
            self._pan_start = event.position().toPoint()
            self._pan_origin_bounds = self.view_bounds_level0
            self._pan_offset = QPoint(0, 0)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self.mode == "erase":
            self._erase_near(event.position().toPoint())
            return
        point = self._widget_to_level0(event.position().toPoint())
        if point is None:
            return
        self.current_stroke = {
            "label": "high_risk",
            "tool": "brush",
            "color": self.brush_color.name(),
            "width": self.brush_width,
            "points_level0": [point],
        }
        self.strokes.append(self.current_stroke)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._pan_start is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self._pan_offset = event.position().toPoint() - self._pan_start
            self.update()
            return
        if self.current_stroke is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        point = self._widget_to_level0(event.position().toPoint())
        if point is None:
            return
        points = self.current_stroke.setdefault("points_level0", [])
        if not points or points[-1] != point:
            points.append(point)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._pan_start is not None:
            bounds = self._bounds_after_pan(self._pan_offset)
            self._pan_start = None
            self._pan_offset = QPoint(0, 0)
            if self.mode == "pan":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.view_bounds_requested.emit(bounds)
        self.current_stroke = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.base_pixmap is None or not self.slide_size[0] or not self.slide_size[1]:
            return
        cursor = event.position().toPoint()
        anchor = self._widget_to_level0(cursor)
        if anchor is None:
            anchor = self._view_center()
        delta = event.angleDelta().y()
        if delta == 0:
            return
        steps = delta / 120.0
        scale = 1.0 / (1.25**steps)
        self.view_bounds_requested.emit(self._zoomed_bounds(anchor, scale))
        event.accept()

    def _draw_strokes(self, painter: QPainter, target: QRect) -> None:
        view_x0, view_y0, view_x1, view_y1 = self.view_bounds_level0
        view_width = max(1, view_x1 - view_x0)
        view_height = max(1, view_y1 - view_y0)
        sx = target.width() / float(view_width)
        sy = target.height() / float(view_height)
        for stroke in self.strokes:
            points = stroke.get("points_level0", [])
            if len(points) < 2:
                continue
            pen = QPen(QColor(stroke.get("color", "#16A34A")))
            pen.setWidth(max(1, int(stroke.get("width", self.brush_width))))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            previous = None
            for x_level0, y_level0 in points:
                point = QPoint(
                    target.left() + int(round((float(x_level0) - view_x0) * sx)),
                    target.top() + int(round((float(y_level0) - view_y0) * sy)),
                )
                if previous is not None:
                    painter.drawLine(previous, point)
                previous = point

    def _image_rect(self) -> QRect:
        if self.base_pixmap is None:
            return self.rect()
        scaled = self.base_pixmap.size()
        scaled.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        left = (self.width() - scaled.width()) // 2 + self._pan_offset.x()
        top = (self.height() - scaled.height()) // 2 + self._pan_offset.y()
        return QRect(left, top, scaled.width(), scaled.height())

    def _widget_to_level0(self, point: QPoint) -> list[int] | None:
        target = self._image_rect()
        if not target.contains(point):
            return None
        view_x0, view_y0, view_x1, view_y1 = self.view_bounds_level0
        view_width = max(1, view_x1 - view_x0)
        view_height = max(1, view_y1 - view_y0)
        x = view_x0 + (point.x() - target.left()) / max(1, target.width()) * view_width
        y = view_y0 + (point.y() - target.top()) / max(1, target.height()) * view_height
        return [int(round(x)), int(round(y))]

    def _erase_near(self, point: QPoint) -> None:
        target = self._image_rect()
        if not target.contains(point):
            return
        threshold = max(8, self.brush_width * 2)
        view_x0, view_y0, view_x1, view_y1 = self.view_bounds_level0
        view_width = max(1, view_x1 - view_x0)
        view_height = max(1, view_y1 - view_y0)
        sx = target.width() / float(view_width)
        sy = target.height() / float(view_height)
        for index in range(len(self.strokes) - 1, -1, -1):
            stroke = self.strokes[index]
            for x_level0, y_level0 in stroke.get("points_level0", []):
                x = target.left() + int(round((float(x_level0) - view_x0) * sx))
                y = target.top() + int(round((float(y_level0) - view_y0) * sy))
                if abs(x - point.x()) <= threshold and abs(y - point.y()) <= threshold:
                    self.strokes.pop(index)
                    self.update()
                    return

    def _zoomed_bounds(self, anchor: list[int], scale: float) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.view_bounds_level0
        old_width = max(1, x1 - x0)
        old_height = max(1, y1 - y0)
        min_width = min(max(256, self.slide_size[0] // 200), self.slide_size[0])
        min_height = min(max(256, self.slide_size[1] // 200), self.slide_size[1])
        new_width = int(round(max(min_width, min(self.slide_size[0], old_width * scale))))
        new_height = int(round(max(min_height, min(self.slide_size[1], old_height * scale))))
        rel_x = (anchor[0] - x0) / old_width
        rel_y = (anchor[1] - y0) / old_height
        new_x0 = int(round(anchor[0] - rel_x * new_width))
        new_y0 = int(round(anchor[1] - rel_y * new_height))
        return self._clamp_bounds(new_x0, new_y0, new_width, new_height)

    def _bounds_after_pan(self, offset: QPoint) -> tuple[int, int, int, int]:
        target = self._image_rect()
        x0, y0, x1, y1 = self._pan_origin_bounds
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        dx = int(round(-offset.x() / max(1, target.width()) * width))
        dy = int(round(-offset.y() / max(1, target.height()) * height))
        return self._clamp_bounds(x0 + dx, y0 + dy, width, height)

    def _clamp_bounds(self, x0: int, y0: int, width: int, height: int) -> tuple[int, int, int, int]:
        slide_width, slide_height = self.slide_size
        width = max(1, min(width, slide_width))
        height = max(1, min(height, slide_height))
        x0 = max(0, min(x0, slide_width - width))
        y0 = max(0, min(y0, slide_height - height))
        return x0, y0, x0 + width, y0 + height

    def _view_center(self) -> list[int]:
        x0, y0, x1, y1 = self.view_bounds_level0
        return [int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))]

    def _emit_zoom(self) -> None:
        if not self.slide_size[0]:
            return
        x0, _, x1, _ = self.view_bounds_level0
        zoom = int(round(self.slide_size[0] / max(1, x1 - x0) * 100))
        self.zoom_changed.emit(max(100, zoom))


class WSIPreviewPanel(QWidget):
    """Preview WSI viewports, sampling overlays, and doctor annotations."""

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        self.current_slide_path = ""
        self.current_slide_info: WSIInfo | None = None
        self.current_view_bounds = (0, 0, 0, 0)
        self.last_sampling_overlay: Image.Image | None = None
        self.is_sampling_overlay = False
        self.slide_info_label = QLabel("尚未加载切片预览")
        self.slide_info_label.setObjectName("MutedText")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.canvas = AnnotationCanvas()
        self.canvas.view_bounds_requested.connect(self.render_view_bounds)
        self.canvas.zoom_changed.connect(self._sync_zoom_label)
        layout.addWidget(self.canvas, 1)

        controls_card = Card("WSI 预览与医生标注")
        slide_row = QHBoxLayout()
        self.slide_combo = QComboBox()
        self.slide_combo.setMinimumWidth(0)
        self.slide_combo.setMinimumContentsLength(24)
        self.slide_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.slide_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.slide_combo.currentIndexChanged.connect(self.load_selected_slide)
        load_button = QPushButton("加载预览")
        load_button.clicked.connect(self.load_selected_slide)
        self.reset_view_button = QPushButton("适配窗口")
        self.reset_view_button.clicked.connect(self.reset_view)
        slide_row.addWidget(QLabel("切片"))
        slide_row.addWidget(self.slide_combo, 1)
        slide_row.addWidget(load_button)
        slide_row.addWidget(self.reset_view_button)

        tool_row = QHBoxLayout()
        self.pan_button = QPushButton("浏览")
        self.pan_button.setCheckable(True)
        self.pan_button.setChecked(True)
        self.pan_button.clicked.connect(lambda: self._set_mode("pan"))
        self.brush_button = QPushButton("画笔")
        self.brush_button.setCheckable(True)
        self.brush_button.clicked.connect(lambda: self._set_mode("brush"))
        self.erase_button = QPushButton("橡皮")
        self.erase_button.setCheckable(True)
        self.erase_button.clicked.connect(lambda: self._set_mode("erase"))
        undo_button = QPushButton("撤销")
        undo_button.clicked.connect(lambda: self.canvas.undo())
        clear_button = QPushButton("清空")
        clear_button.clicked.connect(lambda: self.canvas.clear_annotations())
        save_button = QPushButton("保存标注")
        save_button.clicked.connect(self.save_annotations)
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(2, 36)
        self.brush_slider.setValue(10)
        self.brush_slider.setFixedWidth(110)
        self.brush_slider.valueChanged.connect(self.canvas_width_changed)
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(2, 36)
        self.brush_spin.setValue(10)
        self.brush_spin.setFixedWidth(68)
        self.brush_slider.valueChanged.connect(self.brush_spin.setValue)
        self.brush_spin.valueChanged.connect(self.brush_slider.setValue)
        self.zoom_label = QLabel("缩放 100%")
        self.zoom_label.setObjectName("MutedText")
        tool_row.addWidget(self.pan_button)
        tool_row.addWidget(self.brush_button)
        tool_row.addWidget(self.erase_button)
        tool_row.addWidget(QLabel("笔刷"))
        tool_row.addWidget(self.brush_slider)
        tool_row.addWidget(self.brush_spin)
        tool_row.addWidget(undo_button)
        tool_row.addWidget(clear_button)
        tool_row.addWidget(save_button)
        tool_row.addWidget(self.zoom_label)
        tool_row.addStretch(1)

        sampling_row = QHBoxLayout()
        self.generate_sampling_button = QPushButton("生成采样预览图")
        self.generate_sampling_button.clicked.connect(self.generate_sampling_preview)
        self.save_sampling_button = QPushButton("保存采样图")
        self.save_sampling_button.setEnabled(False)
        self.save_sampling_button.clicked.connect(self.save_sampling_preview)
        self.restore_wsi_button = QPushButton("恢复 WSI")
        self.restore_wsi_button.clicked.connect(self.restore_wsi_preview)
        sampling_row.addWidget(self.generate_sampling_button)
        sampling_row.addWidget(self.save_sampling_button)
        sampling_row.addWidget(self.restore_wsi_button)
        sampling_row.addStretch(1)

        controls_card.body.addLayout(slide_row)
        controls_card.body.addLayout(tool_row)
        controls_card.body.addLayout(sampling_row)
        controls_card.body.addWidget(self.slide_info_label)
        layout.addWidget(controls_card, 0)

    def refresh_slides(self) -> None:
        current = self.slide_combo.currentData()
        self.slide_combo.blockSignals(True)
        self.slide_combo.clear()
        for path in self.state.scanned_files:
            self.slide_combo.addItem(path.name, str(path))
            self.slide_combo.setItemData(self.slide_combo.count() - 1, str(path), Qt.ItemDataRole.ToolTipRole)
        if current:
            index = self.slide_combo.findData(current)
            if index >= 0:
                self.slide_combo.setCurrentIndex(index)
        self.slide_combo.blockSignals(False)
        if self.slide_combo.count() and not self.current_slide_path:
            self.load_selected_slide()

    def load_selected_slide(self) -> None:
        path = self.slide_combo.currentData()
        if not path:
            return
        self.current_slide_path = str(path)
        try:
            with WSIReader(path) as reader:
                self.current_slide_info = reader.info()
            full_bounds = (0, 0, self.current_slide_info.width, self.current_slide_info.height)
            self.render_view_bounds(full_bounds, clear_annotations=True)
        except Exception as exc:
            QMessageBox.critical(self, "预览失败", str(exc))

    def render_view_bounds(self, bounds: object, clear_annotations: bool = False) -> None:
        if not self.current_slide_path:
            return
        try:
            with WSIReader(self.current_slide_path) as reader:
                info = reader.info()
                bounds_tuple = self._clamp_bounds(tuple(int(v) for v in bounds), info)
                image = self._render_region(reader, info, bounds_tuple)
            pixmap = QPixmap.fromImage(ImageQt(image.convert("RGB")))
            self.current_slide_info = info
            self.current_view_bounds = bounds_tuple
            self.is_sampling_overlay = False
            self.canvas.set_rendered_view(pixmap, (info.width, info.height), bounds_tuple, clear_annotations)
            self._update_slide_info()
        except Exception as exc:
            QMessageBox.critical(self, "预览失败", str(exc))

    def reset_view(self) -> None:
        self.canvas.reset_view()

    def restore_wsi_preview(self) -> None:
        if self.current_slide_info is None:
            return
        self.render_view_bounds(self.current_view_bounds or (0, 0, self.current_slide_info.width, self.current_slide_info.height))

    def canvas_width_changed(self, value: int) -> None:
        self.canvas.set_brush_width(value)

    def generate_sampling_preview(self) -> None:
        if not self.current_slide_path or self.current_slide_info is None:
            QMessageBox.information(self, "无法生成", "请先加载一张 WSI 预览。")
            return
        self.state.sync_config_inputs()
        row = {
            "case_id": "preview",
            "slide_id": "interactive",
            "source_path": self.current_slide_path,
        }
        try:
            with tempfile.TemporaryDirectory(prefix="pathodataforge_sampling_") as tmpdir:
                coords = sample_slide_coordinates(
                    source_path=self.current_slide_path,
                    patch_config=self.state.config.get("patch", {}),
                    output_root=tmpdir,
                    row=row,
                    visualization_config={"save_overlay": False},
                    force=True,
                )
            with WSIReader(self.current_slide_path) as reader:
                overlay = build_sampling_overlay(
                    reader,
                    coords.ranges_level0,
                    downsample_factor=int(self.state.config.get("visualization", {}).get("downsample", 32)),
                    outline_rgb=tuple(self.state.config.get("visualization", {}).get("outline_rgb", [37, 99, 235])),
                    outline_width=max(1, int(self.state.config.get("visualization", {}).get("outline_width", 1))),
                    style=str(self.state.config.get("visualization", {}).get("style", "points")),
                    show_tissue_mask=bool(self.state.config.get("visualization", {}).get("show_tissue_mask", True)),
                    max_box_count=int(self.state.config.get("visualization", {}).get("max_box_count", 350)),
                )
                info = reader.info()
            self.last_sampling_overlay = overlay
            self.save_sampling_button.setEnabled(True)
            pixmap = QPixmap.fromImage(ImageQt(overlay.convert("RGB")))
            full_bounds = (0, 0, info.width, info.height)
            self.current_slide_info = info
            self.current_view_bounds = full_bounds
            self.is_sampling_overlay = True
            self.canvas.set_rendered_view(pixmap, (info.width, info.height), full_bounds, clear_annotations=False)
            self.slide_info_label.setText(
                f"采样预览：{coords.ranges_level0.shape[0]} 个候选 patch · "
                f"{Path(self.current_slide_path).name}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "采样预览失败", str(exc))

    def save_sampling_preview(self) -> None:
        if self.last_sampling_overlay is None or not self.current_slide_path:
            QMessageBox.information(self, "无法保存", "请先生成采样预览图。")
            return
        output_dir = Path(self.state.output_dir) / "reports" / "overlays"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{Path(self.current_slide_path).stem}_interactive_sampling_preview.jpg"
        self.last_sampling_overlay.save(path, quality=int(self.state.config.get("visualization", {}).get("jpeg_quality", 90)))
        QMessageBox.information(self, "采样图已保存", f"采样预览图已保存：{path}")

    def save_annotations(self) -> None:
        if not self.current_slide_path:
            QMessageBox.information(self, "无法保存", "请先加载一张 WSI 预览。")
            return
        annotations = self.canvas.export_annotations()
        if not annotations:
            QMessageBox.information(self, "没有标注", "当前没有可保存的画笔标注。")
            return
        stem = Path(self.current_slide_path).stem
        output_dir = Path(self.state.output_dir) / "reports" / "annotations"
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{stem}_doctor_annotations.json"
        png_path = output_dir / f"{stem}_doctor_annotations.png"
        with WSIReader(self.current_slide_path) as reader:
            source_info = reader.info()
        payload = build_annotation_provenance_payload(
            self.current_slide_path,
            (source_info.width, source_info.height),
            annotations,
        )
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.canvas.save_composited_png(png_path)
        manifest_path = Path(self.state.output_dir) / "metadata" / "patch_manifest.csv"
        bound_count = 0
        if manifest_path.exists():
            try:
                bound_count = bind_annotation_relations(manifest_path, json_path)
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "标注已保存但来源绑定失败",
                    f"JSON：{json_path}\nPNG：{png_path}\n{exc}",
                )
                return
        QMessageBox.information(
            self,
            "标注已保存",
            f"JSON：{json_path}\nPNG：{png_path}\n已声明 {bound_count} 个 patch 空间来源关系；交付时将重新计算。",
        )

    def _set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self.brush_button.setChecked(mode == "brush")
        self.erase_button.setChecked(mode == "erase")
        self.pan_button.setChecked(mode == "pan")

    def _sync_zoom_label(self, value: int) -> None:
        self.zoom_label.setText(f"缩放 {value}%")

    def _render_region(
        self,
        reader: WSIReader,
        info: WSIInfo,
        bounds: tuple[int, int, int, int],
    ) -> Image.Image:
        x0, y0, x1, y1 = bounds
        full_bounds = (0, 0, info.width, info.height)
        if bounds == full_bounds:
            return reader.thumbnail((1800, 1200)).convert("RGB")
        width0 = max(1, x1 - x0)
        height0 = max(1, y1 - y0)
        display_width = max(640, self.canvas.width())
        display_height = max(420, self.canvas.height())
        desired_downsample = max(width0 / display_width, height0 / display_height, 1.0)
        level = self._best_level(info, desired_downsample)
        downsample = info.level_downsamples[level]
        size = (
            max(1, int(math.ceil(width0 / downsample))),
            max(1, int(math.ceil(height0 / downsample))),
        )
        image = reader.read_region_level0(x0, y0, level, size).convert("RGB")
        max_edge = 2200
        if image.width > max_edge or image.height > max_edge:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return image

    def _best_level(self, info: WSIInfo, desired_downsample: float) -> int:
        candidates = [
            (index, downsample)
            for index, downsample in enumerate(info.level_downsamples)
            if downsample <= desired_downsample * 1.35
        ]
        if candidates:
            return max(candidates, key=lambda item: item[1])[0]
        return 0

    def _clamp_bounds(self, bounds: tuple[int, int, int, int], info: WSIInfo) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bounds
        width = max(1, min(info.width, x1 - x0))
        height = max(1, min(info.height, y1 - y0))
        x0 = max(0, min(x0, info.width - width))
        y0 = max(0, min(y0, info.height - height))
        return x0, y0, x0 + width, y0 + height

    def _update_slide_info(self) -> None:
        if self.current_slide_info is None or not self.current_slide_path:
            return
        info = self.current_slide_info
        x0, y0, x1, y1 = self.current_view_bounds
        self.slide_info_label.setText(
            f"{Path(self.current_slide_path).name} · {info.width} x {info.height} · "
            f"{info.level_count} 层 · {info.reader_backend} · 当前视野 "
            f"{x1 - x0} x {y1 - y0} px"
        )
