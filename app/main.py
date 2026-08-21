"""
PySide6 Desktop Application Main Entry Point
"""

import sys
import os

# Add project root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication
from app.database.database import init_database
from app.ui.main_window import MainWindow


def main():
    init_database()

    app = QApplication(sys.argv)
    app.setApplicationName("Household AI — Security & Face Recognition")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
