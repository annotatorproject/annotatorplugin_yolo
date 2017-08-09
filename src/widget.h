#ifndef WIDGET_H
#define WIDGET_H

#include <QPixmap>
#include <QWidget>

namespace Ui {
class Widget;
}

namespace Annotator {
namespace Plugins {
class YOLO;
}
}

class Widget : public QWidget {
  Q_OBJECT

 public:
  explicit Widget(QWidget *parent = 0);
  ~Widget();
  void setYOLO(Annotator::Plugins::YOLO *yolo);

 private slots:

  void on_prototxtButton_clicked();

  void on_caffemodelButton_clicked();

  void on_confidenceSpinBox_editingFinished();

  void on_prototxtLineEdit_editingFinished();

  void on_caffemodelLineEdit_editingFinished();

  void on_labelmapButton_clicked();

  void on_labelmapLineEdit_editingFinished();

 private:
  Ui::Widget *ui;
  Annotator::Plugins::YOLO *yolo;
  bool training = false;
};

#endif  // WIDGET_H
