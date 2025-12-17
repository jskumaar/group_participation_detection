#ifndef GAZEVISUALIZER_H
#define GAZEVISUALIZER_H

#include <QVTKOpenGLNativeWidget.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkActor.h>
#include <vtkBillboardTextActor3D.h>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include "360_image_process.h" 

class GazeVisualizer : public QVTKOpenGLNativeWidget {
    Q_OBJECT

public:
    explicit GazeVisualizer(QWidget *parent = nullptr);
    ~GazeVisualizer();

    void updateData(const std::vector<PanoViewer::gaze>& gazes);

private:
    vtkSmartPointer<vtkRenderer> renderer;
    
    struct PersonActors {
        vtkSmartPointer<vtkActor> headActor;
        vtkSmartPointer<vtkActor> gazeArrowActor;
        vtkSmartPointer<vtkBillboardTextActor3D> idLabelActor;
    };
    
    std::map<int, PersonActors> actorsMap; // ID -> Actors
};

#endif // GAZEVISUALIZER_H
