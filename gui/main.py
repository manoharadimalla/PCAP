"""
Application entry point for RunwayMC GUI.

Creates the QApplication instance, sets metadata and styling,
initializes the MainWindow, and starts the event loop.
"""

from __future__ import annotations

import sys
import os

def main() -> None:
    # 1. Gracefully verify PySide6 installation
    try:
        from PySide6.QtCore import Qt, QCoreApplication
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print("\n" + "=" * 70, file=sys.stderr)
        print("ERROR: PySide6 is not installed or could not be loaded!", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("\nPlease install the required GUI dependencies via pip:", file=sys.stderr)
        print("    pip install -r gui/requirements.txt", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    # 2. Check matplotlib availability
    try:
        import matplotlib
        matplotlib.use("QtAgg")
    except ImportError as e:
        print(f"Warning: Matplotlib not found or QtAgg backend failed: {e}", file=sys.stderr)

    # 3. Create Qt Application
    QCoreApplication.setApplicationName("RunwayMC")
    QCoreApplication.setOrganizationName("RunwayMC")
    QCoreApplication.setOrganizationDomain("runwaymc.local")

    app = QApplication(sys.argv)
    app.setApplicationDisplayName("RunwayMC — Runway Scheduling")

    # Set icon if exists
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    icon_path = os.path.join(proj_root, "docs", "icon.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # 4. Initialize and show MainWindow
    from gui.main_window import MainWindow

    window = MainWindow()
    window.show()

    # 5. Exec
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
