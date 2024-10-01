import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QStackedWidget, QProgressBar, QSizePolicy
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, QSize
import qtawesome as qta  # For FontAwesome icons


class Step1(QWidget):
    def __init__(self, parent=None):
        super(Step1, self).__init__(parent)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Step indicator with icon
        step_label = QLabel("Step 1 of 4")
        step_label.setFont(QFont('Arial', 14, QFont.Bold))  # Increased font size
        step_label.setAlignment(Qt.AlignCenter)

        title_layout = QHBoxLayout()
        step_icon = QLabel()
        step_icon.setPixmap(qta.icon('fa.user', color='white').pixmap(QSize(32, 32)))  # Larger icon
        title = QLabel("Contact Information")
        title.setFont(QFont('Arial', 18, QFont.Bold))  # Increased font size
        title.setAlignment(Qt.AlignCenter)

        title_layout.addWidget(step_icon)
        title_layout.addWidget(title, alignment=Qt.AlignCenter)

        # Form inputs
        form_layout = QVBoxLayout()
        form_layout.setAlignment(Qt.AlignCenter)

        self.first_name_input = QLineEdit()
        self.first_name_input.setPlaceholderText("First Name")
        self.first_name_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.last_name_input = QLineEdit()
        self.last_name_input.setPlaceholderText("Last Name")
        self.last_name_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email")
        self.email_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.company_input = QLineEdit()
        self.company_input.setPlaceholderText("Company Name")
        self.company_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        button_layout = QHBoxLayout()
        self.next_button = QPushButton('Next')
        self.next_button.setIcon(qta.icon('fa.arrow-right', color='white'))
        self.next_button.setFixedHeight(40)
        self.next_button.clicked.connect(self.go_to_step2)

        button_layout.addStretch()
        button_layout.addWidget(self.next_button)
        button_layout.addStretch()

        form_layout.addWidget(self.first_name_input)
        form_layout.addWidget(self.last_name_input)
        form_layout.addWidget(self.email_input)
        form_layout.addWidget(self.company_input)

        # Add layouts to main layout
        layout.addWidget(step_label)
        layout.addLayout(title_layout)
        layout.addLayout(form_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def go_to_step2(self):
        print("Step 1: Moving to Step 2")
        try:
            self.parentWidget().parentWidget().progress_bar.setValue(50)
            self.parentWidget().setCurrentIndex(1)
        except Exception as e:
            print(f"Error in transitioning to Step 2: {e}")


class Step2(QWidget):
    def __init__(self, parent=None):
        super(Step2, self).__init__(parent)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        step_label = QLabel("Step 2 of 4")
        step_label.setFont(QFont('Arial', 14, QFont.Bold))  # Increased font size
        step_label.setAlignment(Qt.AlignCenter)

        title_layout = QHBoxLayout()
        step_icon = QLabel()
        step_icon.setPixmap(qta.icon('fa.cog', color='white').pixmap(QSize(32, 32)))  # Larger icon
        title = QLabel("Processing Data")
        title.setFont(QFont('Arial', 18, QFont.Bold))  # Increased font size
        title.setAlignment(Qt.AlignCenter)

        title_layout.addWidget(step_icon)
        title_layout.addWidget(title, alignment=Qt.AlignCenter)

        info = QLabel("We are processing the data from the previous step...")
        info.setFont(QFont('Arial', 14))  # Increased font size
        info.setAlignment(Qt.AlignCenter)

        # Button layout for Previous and Next buttons
        button_layout = QHBoxLayout()

        self.previous_button = QPushButton('Previous')
        self.previous_button.setIcon(qta.icon('fa.arrow-left', color='white'))
        self.previous_button.setFixedHeight(40)
        self.previous_button.clicked.connect(self.go_to_step1)

        self.next_button = QPushButton('Next')
        self.next_button.setIcon(qta.icon('fa.arrow-right', color='white'))
        self.next_button.setFixedHeight(40)
        self.next_button.clicked.connect(self.go_to_step3)

        button_layout.addWidget(self.previous_button)
        button_layout.addStretch()
        button_layout.addWidget(self.next_button)

        layout.addWidget(step_label)
        layout.addLayout(title_layout)
        layout.addWidget(info)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def go_to_step1(self):
        print("Step 2: Moving to Step 1")
        try:
            self.parentWidget().parentWidget().progress_bar.setValue(25)
            self.parentWidget().setCurrentIndex(0)
        except Exception as e:
            print(f"Error in transitioning to Step 1: {e}")

    def go_to_step3(self):
        print("Step 2: Moving to Step 3")
        try:
            self.parentWidget().parentWidget().progress_bar.setValue(75)
            self.parentWidget().setCurrentIndex(2)
        except Exception as e:
            print(f"Error in transitioning to Step 3: {e}")


class Step3(QWidget):
    def __init__(self, parent=None):
        super(Step3, self).__init__(parent)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        step_label = QLabel("Step 3 of 4")
        step_label.setFont(QFont('Arial', 14, QFont.Bold))  # Increased font size
        step_label.setAlignment(Qt.AlignCenter)

        title_layout = QHBoxLayout()
        step_icon = QLabel()
        step_icon.setPixmap(qta.icon('fa.info-circle', color='white').pixmap(QSize(32, 32)))  # Larger icon
        title = QLabel("Results")
        title.setFont(QFont('Arial', 18, QFont.Bold))  # Increased font size
        title.setAlignment(Qt.AlignCenter)

        title_layout.addWidget(step_icon)
        title_layout.addWidget(title, alignment=Qt.AlignCenter)

        result_info = QLabel("Based on the data, here are your results...")
        result_info.setFont(QFont('Arial', 14))  # Increased font size
        result_info.setAlignment(Qt.AlignCenter)

        # Button layout for Previous and Next buttons
        button_layout = QHBoxLayout()

        self.previous_button = QPushButton('Previous')
        self.previous_button.setIcon(qta.icon('fa.arrow-left', color='white'))
        self.previous_button.setFixedHeight(40)
        self.previous_button.clicked.connect(self.go_to_step2)

        self.next_button = QPushButton('Next')
        self.next_button.setIcon(qta.icon('fa.arrow-right', color='white'))
        self.next_button.setFixedHeight(40)
        self.next_button.clicked.connect(self.go_to_step4)

        button_layout.addWidget(self.previous_button)
        button_layout.addStretch()
        button_layout.addWidget(self.next_button)

        layout.addWidget(step_label)
        layout.addLayout(title_layout)
        layout.addWidget(result_info)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def go_to_step2(self):
        print("Step 3: Moving to Step 2")
        try:
            self.parentWidget().parentWidget().progress_bar.setValue(50)
            self.parentWidget().setCurrentIndex(1)
        except Exception as e:
            print(f"Error in transitioning to Step 2: {e}")

    def go_to_step4(self):
        print("Step 3: Moving to Step 4")
        try:
            self.parentWidget().parentWidget().progress_bar.setValue(100)
            self.parentWidget().setCurrentIndex(3)
        except Exception as e:
            print(f"Error in transitioning to Step 4: {e}")


class Step4(QWidget):
    def __init__(self, parent=None):
        super(Step4, self).__init__(parent)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        step_label = QLabel("Step 4 of 4")
        step_label.setFont(QFont('Arial', 14, QFont.Bold))  # Increased font size
        step_label.setAlignment(Qt.AlignCenter)

        title_layout = QHBoxLayout()
        step_icon = QLabel()
        step_icon.setPixmap(qta.icon('fa.check-circle', color='white').pixmap(QSize(32, 32)))  # Larger icon
        title = QLabel("Final Step")
        title.setFont(QFont('Arial', 18, QFont.Bold))  # Increased font size
        title.setAlignment(Qt.AlignCenter)

        title_layout.addWidget(step_icon)
        title_layout.addWidget(title, alignment=Qt.AlignCenter)

        info = QLabel("Submit the final form to complete the process.")
        info.setFont(QFont('Arial', 14))  # Increased font size
        info.setAlignment(Qt.AlignCenter)

        submit_button = QPushButton('Submit')
        submit_button.setIcon(qta.icon('fa.check', color='white'))
        submit_button.setFixedHeight(40)
        submit_button.clicked.connect(self.submit_form)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(submit_button)
        button_layout.addStretch()

        layout.addWidget(step_label)
        layout.addLayout(title_layout)
        layout.addWidget(info)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def submit_form(self):
        print("Form Submitted!")


class MultiStepForm(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Multi-Step Form - Dark Theme")
        self.setGeometry(100, 100, 800, 600)

        # Stack of widgets for multi-step navigation
        self.stacked_widget = QStackedWidget()

        # Add each step to the stacked widget
        self.step1 = Step1(self)
        self.step2 = Step2(self)
        self.step3 = Step3(self)
        self.step4 = Step4(self)

        self.stacked_widget.addWidget(self.step1)
        self.stacked_widget.addWidget(self.step2)
        self.stacked_widget.addWidget(self.step3)
        self.stacked_widget.addWidget(self.step4)

        # Layout and progress bar
        layout = QVBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(25)  # Starts at 25%
        self.progress_bar.setFixedHeight(20)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)

        self.setStyleSheet(self.load_styles())

    def load_styles(self):
        return """
            QWidget {
                background-color: #2b2b2b;
                font-family: Arial;
                font-size: 14px;
                color: #e0e0e0;
            }
            QLabel {
                font-size: 14px;
                margin: 10px 0;
                color: #e0e0e0;
            }
            QLineEdit {
                padding: 12px;
                margin-bottom: 20px;
                border: 1px solid #555;
                border-radius: 5px;
                font-size: 16px;
                background-color: #3b3b3b;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                padding: 12px;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                background-color: #333;
            }
            QProgressBar::chunk {
                background-color: #007bff;
            }
        """


if __name__ == "__main__":
    app = QApplication(sys.argv)

    form = MultiStepForm()
    form.show()

    sys.exit(app.exec_())
