import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from PyQt5.QtCore import Qt

class SummarizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arabic Text Summarization")
        self.setGeometry(100, 100, 800, 600)

        # Create layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Input text
        self.input_label = QLabel("أدخل النص العربي:")
        self.input_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.input_label)

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("اكتب النص هنا...")
        self.layout.addWidget(self.input_text)

        # Summarize button
        self.summarize_button = QPushButton("تلخيص النص")
        self.summarize_button.clicked.connect(self.summarize)
        self.layout.addWidget(self.summarize_button)

        # Output summary
        self.output_label = QLabel("الملخص:")
        self.output_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.output_label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

    def summarize(self):
        self.output_text.setText("Test summary output")

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = SummarizationApp()
        window.show()
        print("GUI launched successfully")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = SummarizationApp()
        window.show()
        print("GUI launched successfully")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()