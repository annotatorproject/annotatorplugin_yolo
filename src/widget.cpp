#include "widget.h"
#include <QtWidgets/QFileDialog>
#include "ui_widget.h"
#include "yolo.h"

Widget::Widget(QWidget *parent) : QWidget(parent), ui(new Ui::Widget) {
  ui->setupUi(this);
}

Widget::~Widget() { delete ui; }

void Widget::setYOLO(Annotator::Plugins::YOLO *yolo) { this->yolo = yolo; }

void Widget::on_prototxtButton_clicked() {
  QString fileName = QFileDialog::getOpenFileName(this, tr("Load cfg File"), "",
                                                  tr("json (*.cfg)"));
  ui->prototxtLineEdit->setText(fileName);
  yolo->setPrototxt(fileName.toStdString());
}

void Widget::on_caffemodelButton_clicked() {
  QString fileName = QFileDialog::getOpenFileName(this, tr("Load weights File"),
                                                  "", tr("model (*.weights)"));
  ui->caffemodelLineEdit->setText(fileName);
  yolo->setModel(fileName.toStdString());
}

void Widget::on_labelmapButton_clicked() {
  QString fileName = QFileDialog::getOpenFileName(
      this, tr("Load label map File"), "", tr("model (*.*)"));
  ui->labelmapLineEdit->setText(fileName);
  yolo->setLabelmap(fileName.toStdString());
}

void Widget::on_confidenceSpinBox_editingFinished() {
  yolo->setConfidenceThreshold(ui->confidenceSpinBox->value());
}

void Widget::on_prototxtLineEdit_editingFinished() {
  yolo->setPrototxt(ui->prototxtLineEdit->text().toStdString());
}

void Widget::on_caffemodelLineEdit_editingFinished() {
  yolo->setModel(ui->caffemodelLineEdit->text().toStdString());
}

void Widget::on_labelmapLineEdit_editingFinished() {
  yolo->setLabelmap(ui->labelmapLineEdit->text().toStdString());
}
