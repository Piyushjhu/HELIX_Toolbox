# spall_analysis_gui.py
"""
SPADE GUI Application

Author: Piyush Wanchoo
Institution: Johns Hopkins University
Year: 2025
GitHub: https://github.com/Piyushjhu/SPADE

Graphical user interface for the Spall Analysis Toolkit (SPADE).
Provides an easy-to-use interface for processing velocity data and generating analysis plots.
"""

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox
)
import threading

class SpallAnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPADE")
        self.setGeometry(100, 100, 600, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Data directory
        self.data_dir_edit = QLineEdit()
        data_dir_btn = QPushButton("Select Raw Data Directory")
        data_dir_btn.clicked.connect(self.select_data_dir)
        layout.addWidget(QLabel("Raw Data Directory:"))
        layout.addWidget(self.data_dir_edit)
        layout.addWidget(data_dir_btn)

        # Output directory
        self.output_dir_edit = QLineEdit()
        output_dir_btn = QPushButton("Select Output Directory")
        output_dir_btn.clicked.connect(self.select_output_dir)
        layout.addWidget(QLabel("Output Directory:"))
        layout.addWidget(self.output_dir_edit)
        layout.addWidget(output_dir_btn)

        # Material properties
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0, 50000)
        self.density_spin.setValue(3000.0)
        self.density_spin.setSuffix(" kg/m³")
        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(0, 50000)
        self.velocity_spin.setValue(11000.0)
        self.velocity_spin.setSuffix(" m/s")
        layout.addWidget(QLabel("Material Density:"))
        layout.addWidget(self.density_spin)
        layout.addWidget(QLabel("Acoustic Velocity:"))
        layout.addWidget(self.velocity_spin)

        # Analysis model
        self.model_combo = QComboBox()
        self.model_combo.addItems(["hybrid_5_segment", "max_min"])
        layout.addWidget(QLabel("Analysis Model:"))
        layout.addWidget(self.model_combo)

        # Signal length
        self.signal_length_combo = QComboBox()
        self.signal_length_combo.addItems(["Full Signal (None)", "Custom..."])
        self.signal_length_combo.currentIndexChanged.connect(self.toggle_signal_length_spin)
        self.signal_length_spin = QDoubleSpinBox()
        self.signal_length_spin.setRange(0, 10000)
        self.signal_length_spin.setValue(20.0)
        self.signal_length_spin.setSuffix(" ns")
        self.signal_length_spin.setEnabled(False)
        layout.addWidget(QLabel("Signal Length (ns):"))
        layout.addWidget(self.signal_length_combo)
        layout.addWidget(self.signal_length_spin)

        # Filtering parameters
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0, 1)
        self.prominence_spin.setSingleStep(0.01)
        self.prominence_spin.setValue(0.01)
        self.prominence_spin.setDecimals(3)
        self.prominence_spin.setSuffix(" (fraction)")
        self.peak_dist_spin = QDoubleSpinBox()
        self.peak_dist_spin.setRange(0, 1000)
        self.peak_dist_spin.setValue(5.0)
        self.peak_dist_spin.setSuffix(" ns")
        layout.addWidget(QLabel("Prominence Factor:"))
        layout.addWidget(self.prominence_spin)
        layout.addWidget(QLabel("Peak Distance (ns):"))
        layout.addWidget(self.peak_dist_spin)

        # Output selection
        self.output_combo = QComboBox()
        self.output_combo.addItems(["core", "supplementary", "both"])
        layout.addWidget(QLabel("Output Selection:"))
        layout.addWidget(self.output_combo)

        # Run button
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_btn)

        # View Output Directory button
        self.view_output_btn = QPushButton("View Output Directory")
        self.view_output_btn.setEnabled(False)
        self.view_output_btn.clicked.connect(self.open_output_dir)
        layout.addWidget(self.view_output_btn)

        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def select_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Raw Data Directory")
        if dir_path:
            self.data_dir_edit.setText(dir_path)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def toggle_signal_length_spin(self):
        if self.signal_length_combo.currentText() == "Custom...":
            self.signal_length_spin.setEnabled(True)
        else:
            self.signal_length_spin.setEnabled(False)

    def run_analysis(self):
        # Disable button to prevent double-clicks
        self.run_btn.setEnabled(False)
        self.status_label.setText("Running analysis...")

        # Gather parameters
        if self.signal_length_combo.currentText() == "Full Signal (None)":
            signal_length_ns = None
        else:
            signal_length_ns = self.signal_length_spin.value()
        params = {
            "raw_data_dir": self.data_dir_edit.text(),
            "output_dir": self.output_dir_edit.text(),
            "density": self.density_spin.value(),
            "acoustic_velocity": self.velocity_spin.value(),
            "analysis_model": self.model_combo.currentText(),
            "signal_length_ns": signal_length_ns,
            "prominence_factor": self.prominence_spin.value(),
            "peak_distance_ns": self.peak_dist_spin.value(),
            "output_selection": self.output_combo.currentText()
        }

        # Run in a thread so GUI doesn't freeze
        threading.Thread(target=self._run_analysis_thread, args=(params,)).start()

    def open_output_dir(self):
        import os
        from PyQt5.QtCore import QUrl
        from PyQt5.QtGui import QDesktopServices
        output_dir = self.output_dir_edit.text()
        if output_dir and os.path.isdir(output_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))
        else:
            QMessageBox.warning(self, "Directory Not Found", "The output directory does not exist.")

    def _run_analysis_thread(self, params):
        try:
            import spall_analysis as sa
            import os

            # Set up processing options
            processing_options = {
                'density': params["density"],
                'acoustic_velocity': params["acoustic_velocity"],
                'analysis_model': params["analysis_model"],
                'signal_length_ns': params["signal_length_ns"],
                'plot_individual': True,
                'smooth_window': 101,
                'polyorder': 1,
                'prominence_factor': params["prominence_factor"],
                'peak_distance_ns': params["peak_distance_ns"],
            }

            # Output directories
            output_dir = params["output_dir"]
            core_plot_dir = os.path.join(output_dir, 'plots', 'core')
            os.makedirs(core_plot_dir, exist_ok=True)

            # Run the analysis
            summary_df = sa.process_velocity_files(
                input_folder=params["raw_data_dir"],
                file_pattern='*.csv',
                output_folder=core_plot_dir,
                save_summary_table=True,
                summary_table_name=os.path.join(output_dir, "tables", "summary_results.csv"),
                **processing_options
            )

            total_files = len(summary_df) if summary_df is not None else 0
            successful = 0
            if total_files > 0 and 'Processing Status' in summary_df.columns:
                successful = (summary_df['Processing Status'] == 'Success').sum()

            self.status_label.setText(f"Analysis complete! Files processed: {total_files}, Successful: {successful}")
            self.view_output_btn.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.view_output_btn.setEnabled(False)
        finally:
            self.run_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SpallAnalysisGUI()
    gui.show()
    sys.exit(app.exec_())
