"""
Custom PySide6 Dark QSS Theme Stylesheet
"""

DARK_THEME_QSS = """
QMainWindow {
    background-color: #0f172a;
    color: #f8fafc;
}

QWidget {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 13px;
}

/* Header */
#topHeader {
    background-color: #1e293b;
    border-bottom: 1px solid #334155;
    padding: 10px 20px;
}

#headerTitle {
    font-size: 18px;
    font-weight: bold;
    color: #818cf8;
}

/* Sidebar */
#sidebar {
    background-color: #1e293b;
    border-right: 1px solid #334155;
    min-width: 200px;
    max-width: 220px;
}

#navBtn {
    background-color: transparent;
    color: #94a3b8;
    border: none;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: left;
    font-size: 14px;
    font-weight: 500;
}

#navBtn:hover {
    background-color: #334155;
    color: #f8fafc;
}

#navBtn:checked {
    background-color: #4f46e5;
    color: #ffffff;
    font-weight: bold;
}

/* Cards */
.QFrame#card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
}

/* Buttons */
QPushButton {
    background-color: #4f46e5;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #6366f1;
}

QPushButton:pressed {
    background-color: #3730a3;
}

QPushButton#secondaryBtn {
    background-color: #334155;
    color: #f8fafc;
}

QPushButton#secondaryBtn:hover {
    background-color: #475569;
}

QPushButton#dangerBtn {
    background-color: #ef4444;
}

QPushButton#dangerBtn:hover {
    background-color: #f87171;
}

/* Tables */
QTableWidget {
    background-color: #1e293b;
    gridline-color: #334155;
    border: 1px solid #334155;
    border-radius: 8px;
}

QHeaderView::section {
    background-color: #0f172a;
    color: #94a3b8;
    padding: 8px;
    font-weight: bold;
    border: none;
    border-bottom: 1px solid #334155;
}

QTableWidget::item {
    padding: 8px;
    color: #f8fafc;
}

QTableWidget::item:selected {
    background-color: #3730a3;
}

/* LineEdit & SpinBox */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 12px;
    color: #f8fafc;
}

QLineEdit:focus, QComboBox:focus {
    border-color: #6366f1;
}

/* Progress Bar */
QProgressBar {
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    background-color: #0f172a;
    color: #f8fafc;
}

QProgressBar::chunk {
    background-color: #10b981;
    border-radius: 6px;
}
"""
