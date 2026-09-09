"""PySide6 main window for PathoTrust."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pathodataforge.app.components import NavButton, StatusChip, repolish
from pathodataforge.app.pages.feature_page import FeaturePage
from pathodataforge.app.pages.metadata_page import MetadataPage
from pathodataforge.app.pages.params_page import ParamsPage
from pathodataforge.app.pages.project_page import ProjectPage
from pathodataforge.app.pages.report_page import ReportPage
from pathodataforge.app.pages.run_page import RunPage
from pathodataforge.app.state import (
    STEP_DONE,
    STEP_ERROR,
    STEP_ORDER,
    STEP_TODO,
    STEP_WARN,
    AppState,
)
from pathodataforge.app.styles import APP_QSS, apply_light_theme
from pathodataforge.app.worker import ProcessingWorker
from pathodataforge.core.metadata_cleaner import clean_metadata
from pathodataforge.core.runtime import get_gpu_info, has_mps
from pathodataforge.core.scanner import read_metadata_fields, scan_wsi_folder
from pathodataforge.utils.config import deep_update
from pathodataforge.utils.io import ensure_directory, read_metadata_table


PAGE_LABELS = [
    ("项目导入", "import"),
    ("元信息匹配", "match"),
    ("WSI 预览与取样", "params"),
    ("特征提取", "feature"),
    ("运行与导出", "run"),
    ("结果报告", "report"),
]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            apply_light_theme(app)
        self.setWindowTitle("PathoTrust")
        self.resize(1400, 900)
        self.state = AppState()
        self.worker_thread: QThread | None = None
        self.worker: ProcessingWorker | None = None
        self._auto_matching = False

        self.setStyleSheet(APP_QSS)
        self._build_shell()
        self._connect_pages()
        self._load_local_project_defaults()
        self.refresh_hardware()
        self._refresh_shell()

    def _build_shell(self) -> None:
        root = QWidget()
        root.setObjectName("AppRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 14, 16, 10)
        root_layout.setSpacing(12)
        self.setCentralWidget(root)

        root_layout.addWidget(self._build_header())

        center = QHBoxLayout()
        center.setSpacing(12)
        center.addWidget(self._build_left_nav(), 0)

        self.stack = QStackedWidget()
        self.project_page = ProjectPage(self.state)
        self.metadata_page = MetadataPage(self.state)
        self.params_page = ParamsPage(self.state)
        self.feature_page = FeaturePage(self.state)
        self.run_page = RunPage(self.state)
        self.report_page = ReportPage(self.state)
        for page in [
            self.project_page,
            self.metadata_page,
            self.params_page,
            self.feature_page,
            self.run_page,
            self.report_page,
        ]:
            self.stack.addWidget(self._wrap_scroll(page))
        center.addWidget(self.stack, 1)
        root_layout.addLayout(center, 1)

        self._build_status_bar()

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("TopHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(12)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("iMoonLab-PathoDataForge")
        title.setObjectName("AppTitle")
        subtitle = QLabel("医院病理 WSI 预处理、质控、特征提取与数据集构建工具")
        subtitle.setObjectName("AppSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box, 1)

        self.cpu_chip = StatusChip("CPU：可用", "info")
        self.hardware_chip = StatusChip("设备：检测中", "info")
        self.model_chip = StatusChip("模型：ResNet50", "info")
        self.cache_chip = StatusChip("缓存：复用缓存", "success")
        for chip in [self.cpu_chip, self.hardware_chip, self.model_chip, self.cache_chip]:
            layout.addWidget(chip)
        return header

    def _build_left_nav(self) -> QFrame:
        nav = QFrame()
        nav.setObjectName("LeftNav")
        nav.setFixedWidth(250)
        layout = QVBoxLayout(nav)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        nav_title = QLabel("流程导航")
        nav_title.setObjectName("CardTitle")
        layout.addWidget(nav_title)
        self.nav_buttons: list[NavButton] = []
        for index, (label, _) in enumerate(PAGE_LABELS, start=1):
            button = NavButton(index, label)
            button.clicked.connect(lambda _=False, i=index - 1: self.set_page(i))
            self.nav_buttons.append(button)
            layout.addWidget(button)

        flow_title = QLabel("流程状态")
        flow_title.setObjectName("CardTitle")
        layout.addSpacing(10)
        layout.addWidget(flow_title)
        self.flow_labels: dict[str, QLabel] = {}
        for key, label in STEP_ORDER:
            item = QLabel(f"{label}：{STEP_TODO}")
            item.setObjectName("MutedText")
            item.setWordWrap(True)
            self.flow_labels[key] = item
            layout.addWidget(item)
        layout.addStretch(1)
        return nav

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        bar.setSizeGripEnabled(False)
        self.setStatusBar(bar)
        self.bottom_state_label = QLabel("当前状态：就绪")
        self.bottom_output_label = QLabel("输出目录：--")
        self.bottom_result_label = QLabel("最近结果：暂无结果")
        self.bottom_progress = QProgressBar()
        self.bottom_progress.setFixedWidth(180)
        self.bottom_progress.setRange(0, 100)
        bar.addWidget(self.bottom_state_label, 1)
        bar.addWidget(self.bottom_output_label, 2)
        bar.addWidget(self.bottom_result_label, 2)
        bar.addPermanentWidget(self.bottom_progress)

    def _connect_pages(self) -> None:
        self.project_page.scan_requested.connect(self.refresh_scan)
        self.project_page.metadata_requested.connect(self.load_metadata_fields)
        self.project_page.check_requested.connect(self.check_project)
        self.metadata_page.preview_requested.connect(self.preview_metadata_match)
        self.metadata_page.auto_guess_requested.connect(self.auto_guess_metadata_fields)
        self.metadata_page.export_requested.connect(self.export_match_report)
        self.params_page.config_loaded.connect(self.apply_loaded_config)
        self.feature_page.refresh_hardware_requested.connect(self.refresh_hardware)
        self.run_page.start_requested.connect(self.start_processing)
        self.state.changed.connect(self._refresh_shell)
        self.state.progress_changed.connect(self._handle_state_progress)

    def set_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.set_active(button_index == index)

    def refresh_scan(self) -> None:
        try:
            self.state.sync_config_inputs()
            if not self.state.wsi_dir:
                raise ValueError("请先选择 WSI 文件夹。")
            result = scan_wsi_folder(self.state.wsi_dir)
            self.state.scan_result = result
            self.state.scanned_files = result.files
            status = STEP_WARN if result.unsupported_count else STEP_DONE
            note = f"{result.image_file_count} 个文件"
            if result.unsupported_count:
                note += f"，{result.unsupported_count} 个不支持"
            self.state.update_step("import", status, note)
            self.project_page.set_scan_result(result)
            self.params_page.refresh_from_state()
            self.state.run_status = "已扫描"
            self._try_auto_match()
        except Exception as exc:
            self.state.update_step("import", STEP_ERROR, "扫描失败")
            QMessageBox.critical(self, "扫描失败", str(exc))

    def load_metadata_fields(self) -> None:
        try:
            if not self.state.metadata_file:
                raise ValueError("请先选择元信息文件。")
            fields = read_metadata_fields(self.state.metadata_file)
            table = read_metadata_table(self.state.metadata_file)
            self.state.metadata_fields = fields
            self.state.metadata_df = table
            self.auto_guess_metadata_fields(show_message=False)
            self.metadata_page.refresh_from_state()
            self.project_page.refresh_from_state()
            self.state.update_step("metadata", STEP_DONE, f"{len(table)} 条记录")
            self.state.run_status = "元信息已读取"
            self._try_auto_match()
        except Exception as exc:
            self.state.update_step("metadata", STEP_ERROR, "读取失败")
            QMessageBox.critical(self, "元信息读取失败", str(exc))

    def auto_guess_metadata_fields(self, show_message: bool = True) -> None:
        fields = self.state.metadata_fields
        if not fields:
            if show_message:
                QMessageBox.information(self, "无法猜测", "请先读取元信息字段。")
            return
        case_col = self._preferred(fields, "case_id", "patient_id", "patient", "病例ID")
        label_col = self._preferred(fields, "label", "diagnosis", "class", "target", "标签")
        filename_col = self._preferred(fields, "filename", "file_name", "slide_id", "submitter_id")
        self.state.set_metadata_columns(case_col, label_col, filename_col, self.state.case_id_regex)
        self.metadata_page.refresh_from_state()
        if show_message:
            QMessageBox.information(self, "自动猜测完成", "已根据字段名填入映射候选。")

    def check_project(self) -> None:
        if not self.state.scan_result:
            self.refresh_scan()
        if self.state.metadata_df is None:
            self.load_metadata_fields()
        issues = []
        if self.state.scan_result and self.state.scan_result.unsupported_count:
            issues.append(f"发现 {self.state.scan_result.unsupported_count} 个不支持文件。")
        if not self.state.scanned_files:
            issues.append("未扫描到支持的 WSI 文件。")
        if self.state.metadata_df is None:
            issues.append("元信息尚未读取。")
        if issues:
            QMessageBox.warning(self, "项目检查", "\n".join(issues))
        else:
            QMessageBox.information(self, "项目检查", "项目输入完整，可以继续预览匹配。")

    def preview_metadata_match(self, show_errors: bool = True) -> None:
        try:
            if not self.state.scanned_files:
                raise ValueError("请先扫描 WSI 文件夹。")
            if self.state.metadata_df is None:
                raise ValueError("请先读取元信息。")
            self.metadata_page._sync_columns()
            result = clean_metadata(
                self.state.scanned_files,
                self.state.metadata_file,
                self.state.selected_case_id_column,
                self.state.selected_label_column,
                self.state.selected_filename_column,
                case_id_regex=self.state.case_id_regex,
            )
            self.state.match_result = result
            self.metadata_page.set_match_result(result)
            total = result.matched_count + result.unmatched_count
            rate = result.matched_count / max(1, total)
            status = STEP_WARN if rate < 0.8 or result.unmatched_count else STEP_DONE
            self.state.update_step("match", status, f"{result.matched_count}/{total}")
            self.state.run_status = "匹配已预览"
        except Exception as exc:
            if show_errors:
                self.state.update_step("match", STEP_ERROR, "匹配失败")
                QMessageBox.critical(self, "匹配失败", str(exc))
            else:
                self.state.update_step("match", STEP_WARN, "等待手动检查")

    def _try_auto_match(self) -> None:
        if self._auto_matching:
            return
        if not self.state.scanned_files or self.state.metadata_df is None:
            return
        if not self.state.selected_case_id_column or not self.state.selected_label_column:
            return
        self._auto_matching = True
        try:
            self.preview_metadata_match(show_errors=False)
        finally:
            self._auto_matching = False

    def export_match_report(self) -> None:
        if not self.state.match_result:
            QMessageBox.information(self, "无法导出", "请先预览匹配。")
            return
        try:
            out_dir = ensure_directory(Path(self.state.output_dir) / "metadata")
            path = out_dir / "match_preview.csv"
            self.state.match_result.metadata.to_csv(path, index=False)
            QMessageBox.information(self, "导出完成", f"匹配报告已保存：{path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def apply_loaded_config(self, config: object) -> None:
        if not isinstance(config, dict):
            return
        self.state.config = deep_update(self.state.config, config)
        input_cfg = self.state.config.get("input", {})
        self.state.set_paths(
            str(input_cfg.get("wsi_dir", self.state.wsi_dir) or ""),
            str(input_cfg.get("metadata_file", self.state.metadata_file) or ""),
            str(input_cfg.get("output_dir", self.state.output_dir) or ""),
        )
        metadata_cfg = self.state.config.get("metadata", {})
        self.state.set_metadata_columns(
            str(metadata_cfg.get("case_id_column", "") or ""),
            str(metadata_cfg.get("label_column", "") or ""),
            str(metadata_cfg.get("filename_column", "") or ""),
            str(metadata_cfg.get("case_id_regex", "") or ""),
        )
        self.params_page.apply_config(self.state.config)
        self.feature_page.apply_config(self.state.config)
        self.project_page.refresh_from_state()
        self.metadata_page.refresh_from_state()
        self.params_page.refresh_from_state()
        self.state.update_step("params", STEP_DONE, "配置已加载")

    def refresh_hardware(self) -> None:
        devices = ["auto", "cpu"]
        lines = ["CPU：可用"]
        gpus = get_gpu_info()
        if gpus:
            for gpu in gpus:
                devices.append(f"cuda:{gpu.index}")
                lines.append(f"CUDA：cuda:{gpu.index} · {gpu.name} · {gpu.total_memory_gb:.2f} GB")
            summary = f"设备：CUDA {len(gpus)}"
        elif has_mps():
            devices.append("mps")
            lines.append("Apple MPS 可用，适合中小批量特征提取。")
            summary = "设备：MPS"
        else:
            lines.append("当前使用 CPU，特征提取可能较慢。")
            summary = "设备：CPU"
        self.state.hardware_status = "\n".join(lines)
        self.state.available_devices = devices
        self.feature_page.set_hardware_status(self.state.hardware_status, devices)
        self.hardware_chip.setText(summary)
        self.hardware_chip.set_kind("success" if len(devices) > 2 else "warn")

    def start_processing(self) -> None:
        if self.worker_thread is not None:
            return
        self.params_page.sync_to_state()
        self.feature_page.sync_to_state()
        self.state.sync_config_inputs()
        errors = self._validate_before_run()
        if errors:
            QMessageBox.warning(self, "无法开始处理", "\n".join(errors))
            self._focus_first_missing_step(errors)
            return

        config = deepcopy(self.state.config)
        self.state.run_status = "运行中"
        self.state.current_stage = "启动中"
        self.state.set_progress(0, "启动中")
        self.state.update_step("run", STEP_TODO, "运行中")
        self.run_page.prepare_run()
        self.set_page(4)

        self.worker_thread = QThread(self)
        self.worker = ProcessingWorker(config)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.handle_progress)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.handle_finished)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def handle_progress(self, value: int, message: str) -> None:
        text = self._localize_progress(message)
        self.state.set_progress(value, text)
        self.run_page.set_progress(value, text)

    def append_log(self, message: str) -> None:
        self.run_page.append_log(str(message))

    def handle_finished(self, summary: object) -> None:
        if isinstance(summary, dict):
            self.state.set_summary(summary)
            self.run_page.set_output_paths(self.state.output_paths)
            self.run_page.set_summary(summary)
            self.report_page.refresh_from_state()
            self.state.run_status = "完成"
            self.state.set_progress(100, "完成")
            self.state.update_step("run", STEP_DONE, "处理完成")
            self.state.update_step("report", STEP_DONE, "报告已生成")
        else:
            self.run_page.summary_text.setPlainText(str(summary))
        QMessageBox.information(self, "处理完成", "iMoonLab-PathoDataForge 处理完成。")

    def handle_error(self, message: str) -> None:
        self.state.run_status = "失败"
        self.state.set_progress(100, "失败")
        self.state.update_step("run", STEP_ERROR, "处理失败")
        self.run_page.append_log(message)
        QMessageBox.critical(self, "处理失败", message)

    def _cleanup_worker(self) -> None:
        self.run_page.finish_run_controls()
        if self.worker:
            self.worker.deleteLater()
        if self.worker_thread:
            self.worker_thread.deleteLater()
        self.worker = None
        self.worker_thread = None

    def _validate_before_run(self) -> list[str]:
        errors = self.state.validate_ready_to_run()
        quality = self.state.config.get("quality", {})
        if float(quality.get("brightness_min", 0)) >= float(quality.get("brightness_max", 255)):
            errors.append("最小亮度必须小于最大亮度。")
        return errors

    def _focus_first_missing_step(self, errors: list[str]) -> None:
        text = "\n".join(errors)
        if "扫描" in text or "WSI" in text or "输出目录" in text:
            self.set_page(0)
        elif "元信息" in text or "字段" in text or "匹配" in text:
            self.set_page(1)
        elif "比例" in text or "亮度" in text:
            self.set_page(3)

    def _handle_state_progress(self, value: int, message: str) -> None:
        self.bottom_progress.setValue(value)

    def _refresh_shell(self) -> None:
        self.project_page.wsi_row.set_path(self.state.wsi_dir)
        self.project_page.metadata_row.set_path(self.state.metadata_file)
        self.project_page.output_row.set_path(self.state.output_dir)
        self.model_chip.setText(f"模型：{self.state.model_name}")
        cache_kind = "warn" if self.state.cache_text == "覆盖重算" else "success"
        self.cache_chip.setText(f"缓存：{self.state.cache_text}")
        self.cache_chip.set_kind(cache_kind)
        self.bottom_state_label.setText(f"当前状态：{self.state.run_status}")
        self.bottom_output_label.setText(f"输出目录：{self.state.output_dir or '--'}")
        self.bottom_result_label.setText(f"最近结果：{self.state.last_result_text}")
        self.bottom_progress.setValue(self.state.progress)

        for page_index, button in enumerate(self.nav_buttons):
            _, step_key = PAGE_LABELS[page_index]
            status = self.state.step_status.get(step_key, STEP_TODO)
            note = self.state.step_notes.get(step_key, "")
            button.set_status(status, note)
            button.set_active(self.stack.currentIndex() == page_index)

        for key, label in self.flow_labels.items():
            title = next(name for step_key, name in STEP_ORDER if step_key == key)
            status = self.state.step_status.get(key, STEP_TODO)
            note = self.state.step_notes.get(key, "")
            label.setText(f"{title}：{status}{f' · {note}' if note else ''}")
            if status == STEP_DONE:
                label.setStyleSheet("color: #166534; font-weight: 700;")
            elif status == STEP_WARN:
                label.setStyleSheet("color: #92400E; font-weight: 700;")
            elif status == STEP_ERROR:
                label.setStyleSheet("color: #B91C1C; font-weight: 700;")
            else:
                label.setStyleSheet("color: #64748B;")
        repolish(self)

    def _load_local_project_defaults(self) -> None:
        root = Path.cwd()
        wsi_dir = root / "WSI"
        metadata = wsi_dir / "metadata_from_clinical.csv"
        output = root / "PathoDataForge_output"
        if wsi_dir.exists():
            self.state.wsi_dir = str(wsi_dir)
        if metadata.exists():
            self.state.metadata_file = str(metadata)
        self.state.output_dir = str(output)
        self.state.sync_config_inputs()
        self.project_page.refresh_from_state()
        if self.state.wsi_dir:
            try:
                self.refresh_scan()
            except Exception:
                pass
        if self.state.metadata_file:
            try:
                self.load_metadata_fields()
            except Exception:
                pass

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        widget.setObjectName("PageContent")
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.viewport().setObjectName("PageContent")
        area.setWidget(widget)
        area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return area

    @staticmethod
    def _preferred(fields: list[str], *candidates: str) -> str:
        lowered = {field.lower(): field for field in fields}
        normalized = {field.lower().replace(" ", "_").replace("-", "_"): field for field in fields}
        for candidate in candidates:
            key = candidate.lower()
            if key in lowered:
                return lowered[key]
            if key in normalized:
                return normalized[key]
        return fields[0] if fields else ""

    @staticmethod
    def _localize_progress(message: str) -> str:
        text = str(message or "")
        mapping = {
            "Starting": "启动中",
            "Scanned input folder": "已扫描输入文件",
            "Matched metadata": "已匹配元信息",
            "Sampling coordinates": "正在采样坐标",
            "Writing summary": "正在写入结果摘要",
            "Done": "完成",
            "Failed": "失败",
        }
        if text.startswith("Processing slide"):
            return text.replace("Processing slide", "正在处理切片")
        return mapping.get(text, text)


def run_app() -> None:
    app = QApplication([])
    apply_light_theme(app)
    window = MainWindow()
    window.show()
    app.exec()
