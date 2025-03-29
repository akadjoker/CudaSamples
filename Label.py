import os
import glob
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QMessageBox, QComboBox, QToolBar, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem
)
from PySide6.QtGui import QPixmap, QPen, QColor, QMouseEvent, QWheelEvent
from PySide6.QtCore import Qt, QRectF, QPointF

class ImageCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.image_item = None
        self.current_class = 0
        self.start_point = None
        self.rect_items = []
        self._panning = False
        self._pan_start = QPointF()

    def load_image(self, image_path):
        self.scene.clear()
        self.rect_items.clear()
        pixmap = QPixmap(image_path)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 0.85
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            self.start_point = self.mapToScene(event.position().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.LeftButton and self.start_point:
            end_point = self.mapToScene(event.position().toPoint())
            rect = QRectF(self.start_point, end_point).normalized()
            box = QGraphicsRectItem(rect)
            box.setPen(QPen(QColor(0, 255, 0), 2))
            box.setData(0, self.current_class)
            self.scene.addItem(box)
            self.rect_items.append(box)
            self.start_point = None
        super().mouseReleaseEvent(event)



    def set_class_index(self, class_index):
        self.current_class = class_index

    def undo_last_box(self):
        if self.rect_items:
            last = self.rect_items.pop()
            self.scene.removeItem(last)

    def get_normalized_boxes(self):
        boxes = []
        if not self.image_item:
            return boxes
        img_rect = self.image_item.boundingRect()
        for item in self.rect_items:
            rect = item.rect()
            x = (rect.x() + rect.width() / 2) / img_rect.width()
            y = (rect.y() + rect.height() / 2) / img_rect.height()
            w = rect.width() / img_rect.width()
            h = rect.height() / img_rect.height()
            cls = item.data(0)
            boxes.append((cls, x, y, w, h))
        return boxes

class Annotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotator YOLO Zoom & Pan")
        self.image_index = 0
        self.image_paths = []
        self.class_list = []

        self.canvas = ImageCanvas()

        self.combo_classes = QComboBox()
        self.load_class_list("classes.txt")
        self.combo_classes.currentIndexChanged.connect(self.update_class)

        self.btn_load = QPushButton("📂 Carregar pasta")
        self.btn_save = QPushButton("💾 Guardar anotação")
        self.btn_next = QPushButton("➡ Próxima imagem")
        self.btn_undo = QPushButton("↩ Undo último box")

        self.toolbar = QToolBar()
        self.toolbar.addWidget(QLabel("📦 Classe:"))
        self.toolbar.addWidget(self.combo_classes)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.btn_load)
        self.toolbar.addWidget(self.btn_save)
        self.toolbar.addWidget(self.btn_undo)
        self.toolbar.addWidget(self.btn_next)
        self.addToolBar(self.toolbar)

        

        self.btn_load.clicked.connect(self.load_folder)
        self.btn_save.clicked.connect(self.save_annotation)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_undo.clicked.connect(self.canvas.undo_last_box)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)


        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_class_list(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.class_list = [line.strip() for line in f if line.strip()]
                self.combo_classes.addItems(self.class_list)

    def update_class(self, index):
        self.canvas.set_class_index(index)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar pasta com imagens")
        if folder:
            self.image_paths = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            self.image_paths = [os.path.join(folder, f) for f in self.image_paths]
            self.image_index = 0
            if self.image_paths:
                self.canvas.load_image(self.image_paths[self.image_index])
            else:
                QMessageBox.warning(self, "Aviso", "Nenhuma imagem encontrada na pasta.")

    def save_annotation(self):
        if not self.image_paths:
            return
        boxes = self.canvas.get_normalized_boxes()
        if boxes:
            txt_path = os.path.splitext(self.image_paths[self.image_index])[0] + ".txt"
            with open(txt_path, "w") as f:
                for b in boxes:
                    f.write(" ".join([str(round(x, 6)) for x in b]) + "\n")
            print(f"✅ Anotação guardada: {txt_path}")
        else:
            print("⚠ Nenhuma caixa anotada.")

    def next_image(self):
        self.save_annotation()
        self.image_index += 1
        if self.image_index < len(self.image_paths):
            self.canvas.load_image(self.image_paths[self.image_index])
        else:
            QMessageBox.information(self, "Fim", "Não há mais imagens.")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Annotator()
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec())

