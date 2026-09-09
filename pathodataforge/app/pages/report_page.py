"""Result report page."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pathodataforge.app.components import Card, StatCard, icon_button, open_path
from pathodataforge.app.state import AppState
from pathodataforge.core.delivery import build_verified_delivery
from pathodataforge.core.replay_verifier import (
    canonical_json,
    resolve_annotation_by_wsi_content,
    sha256_file,
)
from pathodataforge.privacy_index import ensure_local_secret_key, run_privacy_index_dataframe


class ReportPage(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        title = QLabel("结果报告")
        title.setObjectName("PageTitle")
        subtitle = QLabel("查看输出文件、数据整理状态，并生成医院本地多维脱敏索引。")
        subtitle.setObjectName("MutedText")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        layout.addWidget(self._build_overview_card())
        layout.addWidget(self._build_stats_card())
        layout.addWidget(self._build_privacy_index_card())
        layout.addWidget(self._build_actions_card())
        layout.addStretch(1)

    def _build_overview_card(self) -> Card:
        card = Card("输出文件概览")
        self.file_lines: dict[str, QLineEdit] = {}
        for key, label in [
            ("metadata_cleaned", "metadata_cleaned.csv"),
            ("mapping", "mapping.csv"),
            ("manifest", "patch_manifest.csv"),
            ("coordinates", "坐标目录"),
            ("features", "特征目录"),
            ("summary", "summary.json"),
            ("overlays", "采样预览图目录"),
            ("annotations", "医生标注目录"),
            ("privacy_index", "多维脱敏索引目录"),
        ]:
            row = QHBoxLayout()
            title = QLabel(label)
            title.setMinimumWidth(150)
            line = QLineEdit()
            line.setReadOnly(True)
            button = QPushButton("打开")
            button.clicked.connect(lambda _=False, k=key: open_path(self.file_lines[k].text(), self))
            row.addWidget(title)
            row.addWidget(line, 1)
            row.addWidget(button)
            card.body.addLayout(row)
            self.file_lines[key] = line
        return card

    def _build_stats_card(self) -> Card:
        card = Card("结果统计")
        grid = QGridLayout()
        self.total_wsi_card = StatCard("总 WSI 数")
        self.total_case_card = StatCard("总病例数")
        self.total_patch_card = StatCard("总 patch 数")
        self.label_card = StatCard("标签数量")
        self.tissue_card = StatCard("平均组织比例")
        self.blur_card = StatCard("平均模糊分数")
        cards = [
            self.total_wsi_card,
            self.total_case_card,
            self.total_patch_card,
            self.label_card,
            self.tissue_card,
            self.blur_card,
        ]
        for index, item in enumerate(cards):
            grid.addWidget(item, index // 4, index % 4)
            grid.setColumnStretch(index % 4, 1)
        card.body.addLayout(grid)
        return card

    def _build_privacy_index_card(self) -> Card:
        card = Card("多维脱敏索引")
        desc = QLabel(
            "把患者、病例、样本、蜡块、切片、WSI、patch、医生标注等维度对齐，生成可交付的公开 manifest，"
            "并将可回溯私有映射表仅保存在医院本地。"
        )
        desc.setObjectName("MutedText")
        desc.setWordWrap(True)
        card.body.addWidget(desc)

        usage = QLabel(
            "使用方法：完成项目导入和元信息匹配后，可先在 WSI 预览页保存医生标注；"
            "然后点击“生成多维脱敏索引”。系统会自动使用本机医院密钥生成稳定匿名 ID。"
            "对外交付公开 manifest 和匿名数据即可，private_id_mapping.csv 只留在医院本地用于追溯。"
        )
        usage.setObjectName("MutedText")
        usage.setWordWrap(True)
        card.body.addWidget(usage)

        self.privacy_index_hint = QLabel("密钥状态：首次生成时自动创建本机密钥；后续自动复用，无需医生手动输入。")
        self.privacy_index_hint.setObjectName("MutedText")
        self.privacy_index_hint.setWordWrap(True)
        card.body.addWidget(self.privacy_index_hint)
        return card

    def _build_actions_card(self) -> Card:
        card = Card("快捷操作")
        row = QHBoxLayout()
        self.open_root_button = icon_button("打开输出目录")
        self.open_root_button.clicked.connect(lambda: open_path(self.state.output_paths.get("root", ""), self))
        self.open_summary_button = icon_button("打开 summary.json")
        self.open_summary_button.clicked.connect(lambda: open_path(self.state.output_paths.get("summary", ""), self))
        self.open_manifest_button = icon_button("打开 patch_manifest.csv")
        self.open_manifest_button.clicked.connect(lambda: open_path(self.state.output_paths.get("manifest", ""), self))
        self.open_features_button = icon_button("打开特征目录")
        self.open_features_button.clicked.connect(lambda: open_path(self.state.output_paths.get("features", ""), self))
        self.open_overlay_button = icon_button("打开采样预览图目录")
        self.open_overlay_button.clicked.connect(lambda: open_path(self.state.output_paths.get("overlays", ""), self))
        self.package_button = icon_button("生成脱敏数据包")
        self.package_button.clicked.connect(self.package_dataset)
        self.privacy_index_button = icon_button("生成多维脱敏索引")
        self.privacy_index_button.clicked.connect(self.build_privacy_index)
        self.privacy_index_button.setToolTip("系统自动使用本机医院密钥生成稳定匿名索引。")
        self.export_html_button = icon_button("导出 HTML 报告")
        self.export_html_button.setEnabled(False)
        self.export_html_button.setToolTip("后续版本支持。")
        for button in [
            self.open_root_button,
            self.open_summary_button,
            self.open_manifest_button,
            self.open_features_button,
            self.open_overlay_button,
            self.package_button,
            self.privacy_index_button,
            self.export_html_button,
        ]:
            row.addWidget(button)
        row.addStretch(1)
        card.body.addLayout(row)
        self.package_path_label = QLabel("脱敏数据包：尚未生成")
        self.package_path_label.setObjectName("MutedText")
        card.body.addWidget(self.package_path_label)
        self.privacy_index_label = QLabel("多维脱敏索引：尚未生成")
        self.privacy_index_label.setObjectName("MutedText")
        card.body.addWidget(self.privacy_index_label)
        return card

    def refresh_from_state(self) -> None:
        for key, line in self.file_lines.items():
            value = self.state.output_paths.get(key, "")
            line.setText(value)
            line.setToolTip(value)
        summary = self.state.last_summary or {}
        self._set_summary_stats(summary)

    def _set_summary_stats(self, summary: dict[str, Any]) -> None:
        input_cfg = summary.get("input", {})
        patches = summary.get("patches", {})
        self.total_wsi_card.set_value(input_cfg.get("scanned_files", 0))
        self.total_patch_card.set_value(patches.get("kept", 0), f"丢弃 {patches.get('discarded', 0)}")
        manifest_path = self.state.output_paths.get("manifest", "")
        if not manifest_path or not Path(manifest_path).exists():
            self.total_case_card.set_value("--")
            self.label_card.set_value("--")
            self.tissue_card.set_value("--")
            self.blur_card.set_value("--")
            return
        try:
            manifest = pd.read_csv(manifest_path)
        except Exception:
            return
        kept = manifest[manifest["keep"].astype(str).str.lower().isin(["true", "1"])] if "keep" in manifest else manifest
        self.total_case_card.set_value(kept["case_id"].nunique() if "case_id" in kept else "--")
        if "label" in kept:
            label_counts = kept["label"].fillna("").astype(str).value_counts().to_dict()
            self.label_card.set_value(", ".join(f"{key}:{value}" for key, value in label_counts.items()) or "--")
        if "tissue_ratio" in kept and not kept.empty:
            self.tissue_card.set_value(f"{kept['tissue_ratio'].mean():.3f}")
        if "blur_score" in kept and not kept.empty:
            self.blur_card.set_value(f"{kept['blur_score'].mean():.1f}")

    def package_dataset(self) -> None:
        root = self.state.output_paths.get("root", "")
        if not root:
            QMessageBox.information(self, "无法打包", "请先完成一次处理。")
            return
        root_path = Path(root)
        if not root_path.exists():
            QMessageBox.warning(self, "无法打包", f"输出目录不存在：{root_path}")
            return
        package_root = root_path / "deliverables" / f"PathoTrust_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            build_verified_delivery(
                root_path,
                package_root,
                source_search_roots=[self.state.wsi_dir] if self.state.wsi_dir else (),
            )
        except Exception as exc:
            QMessageBox.warning(self, "交付已阻止", str(exc))
            return
        self.package_path_label.setText(f"脱敏数据包：{package_root}")
        QMessageBox.information(self, "打包完成", f"脱敏数据包已生成：{package_root}")

    def build_privacy_index(self) -> None:
        if not self.state.match_result:
            QMessageBox.information(self, "无法生成", "请先完成元信息匹配。")
            return
        output_root = Path(self.state.output_dir) / "privacy_index"
        try:
            secret_key, key_path, created_key = ensure_local_secret_key()
            result = run_privacy_index_dataframe(
                self._privacy_index_metadata(),
                output_root,
                annotation_root=Path(self.state.output_dir) / "reports" / "annotations",
                secret_key=secret_key,
            )
        except Exception as exc:
            QMessageBox.critical(self, "生成失败", str(exc))
            return
        self.state.output_paths["privacy_index"] = str(output_root)
        self.file_lines["privacy_index"].setText(str(output_root))
        self.privacy_index_label.setText(
            f"多维脱敏索引：{result.manifest_csv}；私有映射：{result.private_mapping_csv}"
        )
        self.privacy_index_hint.setText(
            f"密钥状态：{'已创建' if created_key else '已复用'}本机医院密钥；路径 {key_path}。"
            "请由医院信息科或项目负责人备份，不随数据包外发。"
        )
        QMessageBox.information(
            self,
            "生成完成",
            "多维脱敏索引已生成。\n"
            f"公开 manifest：{result.manifest_csv}\n"
            f"私有映射表：{result.private_mapping_csv}\n\n"
            "系统已自动使用本机医院密钥；私有映射表和密钥只能保存在医院本地授权环境中。",
        )

    def _privacy_index_metadata(self) -> pd.DataFrame:
        if not self.state.match_result:
            return pd.DataFrame()
        data = self.state.match_result.metadata.copy()
        raw = self.state.metadata_df
        case_column = self.state.selected_case_id_column
        if raw is not None and case_column in raw.columns and "case_id" in data.columns:
            raw_copy = raw.copy()
            raw_copy["_case_join"] = raw_copy[case_column].astype(str).str.strip()
            data["_case_join"] = data["case_id"].astype(str).str.strip()
            extra_columns = [column for column in raw_copy.columns if column not in data.columns and column != "_case_join"]
            data = data.merge(raw_copy[["_case_join", *extra_columns]], on="_case_join", how="left")
            data = data.drop(columns=["_case_join"])

        def alias(*keywords: str) -> str:
            normalized_keywords = [keyword.lower().replace("_", "") for keyword in keywords]
            for column in data.columns:
                normalized = str(column).lower().replace("_", "").replace("-", "")
                if any(keyword in normalized for keyword in normalized_keywords):
                    return str(column)
            return ""

        def fill_from_alias(target: str, *keywords: str, fallback: str = "") -> None:
            if target in data.columns and data[target].astype(str).str.strip().any():
                return
            source = alias(target, *keywords)
            if source and source in data.columns:
                data[target] = data[source]
            else:
                data[target] = fallback

        fill_from_alias("real_patient_id", "patientid", "patientbarcode", "submitterid", fallback="")
        if "case_id" in data.columns:
            empty_real = data["real_patient_id"].astype(str).str.strip() == ""
            data.loc[empty_real, "real_patient_id"] = data.loc[empty_real, "case_id"].astype(str)
        fill_from_alias("patient_name", "patientname", "name", fallback="")
        fill_from_alias("hospital_id", "hospitalid", "mrn", "medicalrecord", fallback="")
        fill_from_alias("pathology_id", "pathologyid", "pathid", fallback="")
        fill_from_alias("specimen_id", "specimenid", "sampleid", "sample", fallback="")
        fill_from_alias("block_id", "blockid", "block", fallback="")
        fill_from_alias("stain_type", "stain", fallback="HE")
        fill_from_alias("tumor_site", "tumorsite", "primarysite", "organ", fallback="")
        fill_from_alias("diagnosis", "diagnosis", "primarydiagnosis", fallback="")
        fill_from_alias("age", "ageatindex", "ageatdiagnosis", "age", fallback="")
        fill_from_alias("scan_date", "scandate", "collectiondate", "diagnosisdate", fallback="")
        fill_from_alias("doctor_id", "doctorid", "pathologist", fallback="")
        if "diagnosis" in data.columns and "label" in data.columns:
            empty_diagnosis = data["diagnosis"].astype(str).str.strip() == ""
            data.loc[empty_diagnosis, "diagnosis"] = data.loc[empty_diagnosis, "label"].astype(str)
        if "pathology_id" in data.columns and "case_id" in data.columns:
            empty_pathology = data["pathology_id"].astype(str).str.strip() == ""
            data.loc[empty_pathology, "pathology_id"] = data.loc[empty_pathology, "case_id"].astype(str)
        if "specimen_id" in data.columns and "case_id" in data.columns:
            empty_specimen = data["specimen_id"].astype(str).str.strip() == ""
            data.loc[empty_specimen, "specimen_id"] = data.loc[empty_specimen, "case_id"].astype(str) + "_SPEC"
        if "block_id" in data.columns and "slide_id" in data.columns:
            empty_block = data["block_id"].astype(str).str.strip() == ""
            data.loc[empty_block, "block_id"] = data.loc[empty_block, "slide_id"].astype(str) + "_BLOCK"
        if "original_filename" in data.columns:
            data["wsi_filename"] = data["original_filename"]
        elif "source_path" in data.columns:
            data["wsi_filename"] = data["source_path"].map(lambda value: Path(str(value)).name)
        patch_manifest_path = Path(self.state.output_dir) / "metadata" / "patch_manifest.csv"
        patch_manifest = (
            pd.read_csv(patch_manifest_path).fillna("")
            if patch_manifest_path.exists()
            else pd.DataFrame()
        )

        def replay_summary(row: pd.Series) -> pd.Series:
            source_path = Path(str(row.get("source_path", "") or ""))
            if not source_path.exists() or patch_manifest.empty or "wsi_content_sha256" not in patch_manifest:
                return pd.Series(
                    {
                        "replay_profile_ids": "",
                        "replay_profiles": "[]",
                        "patch_verification_predicates": "[]",
                        "verified_patch_count": 0,
                        "patch_verification_status": "NOT_APPLICABLE",
                    }
                )
            digest = sha256_file(source_path)
            slide_rows = patch_manifest[
                patch_manifest["wsi_content_sha256"].astype(str) == digest
            ]
            kept_rows = slide_rows[
                slide_rows["keep"].astype(str).str.lower().isin(["true", "1"])
            ]
            if kept_rows.empty:
                status = "NOT_APPLICABLE"
            elif "verification_status" not in kept_rows:
                status = "UNSUPPORTED_PROFILE"
            elif kept_rows["verification_status"].astype(str).eq("VERIFIED").all():
                status = "VERIFIED"
            else:
                status = "|".join(sorted(set(kept_rows["verification_status"].astype(str))))
            profile_ids = "|".join(
                sorted({value for value in kept_rows.get("extraction_profile_id", pd.Series(dtype=str)).astype(str) if value})
            )
            profiles = []
            for value in sorted(
                {value for value in kept_rows.get("extraction_profile", pd.Series(dtype=str)).astype(str) if value}
            ):
                try:
                    profiles.append(json.loads(value))
                except json.JSONDecodeError:
                    profiles.append({"unsupported_raw_profile": value})
            predicates = []
            ordered_rows = (
                kept_rows.sort_values("patch_provenance_id")
                if "patch_provenance_id" in kept_rows
                else kept_rows
            )
            for _, patch_row in ordered_rows.iterrows():
                raw_predicates = str(patch_row.get("verification_predicates", "") or "")
                try:
                    parsed_predicates = json.loads(raw_predicates) if raw_predicates else {}
                except json.JSONDecodeError:
                    parsed_predicates = {"unsupported_raw_predicates": raw_predicates}
                predicates.append(
                    {
                        "patch_provenance_id": str(patch_row.get("patch_provenance_id", "")),
                        "predicates": parsed_predicates,
                    }
                )
            return pd.Series(
                {
                    "replay_profile_ids": profile_ids,
                    "replay_profiles": canonical_json(profiles),
                    "patch_verification_predicates": canonical_json(predicates),
                    "verified_patch_count": int(kept_rows.get("verification_status", pd.Series(dtype=str)).astype(str).eq("VERIFIED").sum()),
                    "patch_verification_status": status,
                }
            )

        replay_columns = data.apply(replay_summary, axis=1)
        for column in replay_columns.columns:
            data[column] = replay_columns[column]
        data["annotation_file"] = data.apply(self._annotation_file_for_row, axis=1)
        return data

    def _annotation_file_for_row(self, row: pd.Series) -> str:
        annotation_dir = Path(self.state.output_dir) / "reports" / "annotations"
        if not annotation_dir.exists():
            return ""
        source_path = Path(str(row.get("source_path", "") or ""))
        if not source_path.exists():
            return ""
        source_digest = sha256_file(source_path)
        match = resolve_annotation_by_wsi_content(annotation_dir, source_digest)
        return str(match) if match else ""
