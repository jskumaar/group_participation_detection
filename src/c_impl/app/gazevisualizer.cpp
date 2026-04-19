#include "app/gazevisualizer.h"

#include <vtkSphereSource.h>
#include <vtkArrowSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkTransform.h>
#include <vtkCamera.h>
#include <vtkTextProperty.h>

#include <cmath>

GazeVisualizer::GazeVisualizer(QWidget *parent) : QVTKOpenGLNativeWidget(parent) {
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    setRenderWindow(renderWindow);

    renderer = vtkSmartPointer<vtkRenderer>::New();
    renderWindow->AddRenderer(renderer);

    renderer->SetBackground(0.1, 0.1, 0.1);

    renderer->GetActiveCamera()->SetPosition(0, 500, 1000);
    renderer->GetActiveCamera()->SetFocalPoint(0, 0, 0);
    renderer->GetActiveCamera()->SetViewUp(0, 1, 0);
    renderer->ResetCamera();
}

GazeVisualizer::~GazeVisualizer() {
    // VTK smart pointers handle cleanup
}

void GazeVisualizer::updateData(const std::vector<PanoViewer::gaze>& gazes) {
    std::vector<int> activeIDs;

    for (const auto& p : gazes) {
        activeIDs.push_back(p.personID);

        if (actorsMap.find(p.personID) == actorsMap.end()) {
            PersonActors actors;

            vtkNew<vtkSphereSource> sphereSource;
            sphereSource->SetRadius(15.0);
            sphereSource->SetThetaResolution(16);
            sphereSource->SetPhiResolution(16);

            vtkNew<vtkPolyDataMapper> sphereMapper;
            sphereMapper->SetInputConnection(sphereSource->GetOutputPort());

            actors.headActor = vtkSmartPointer<vtkActor>::New();
            actors.headActor->SetMapper(sphereMapper);
            actors.headActor->GetProperty()->SetColor(0.0, 0.0, 1.0);
            renderer->AddActor(actors.headActor);

            vtkNew<vtkArrowSource> arrowSource;

            vtkNew<vtkPolyDataMapper> arrowMapper;
            arrowMapper->SetInputConnection(arrowSource->GetOutputPort());

            actors.gazeArrowActor = vtkSmartPointer<vtkActor>::New();
            actors.gazeArrowActor->SetMapper(arrowMapper);
            actors.gazeArrowActor->GetProperty()->SetColor(1.0, 0.0, 0.0);
            renderer->AddActor(actors.gazeArrowActor);

            actors.idLabelActor = vtkSmartPointer<vtkBillboardTextActor3D>::New();
            actors.idLabelActor->SetInput(std::to_string(p.personID).c_str());
            actors.idLabelActor->GetTextProperty()->SetFontSize(24);
            actors.idLabelActor->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
            actors.idLabelActor->GetTextProperty()->SetJustificationToCentered();
            renderer->AddActor(actors.idLabelActor);

            actorsMap[p.personID] = actors;
        }

        PersonActors& actors = actorsMap[p.personID];

        actors.headActor->SetPosition(p.start.x, p.start.y, p.start.z);
        actors.idLabelActor->SetPosition(p.start.x, p.start.y + 5.0, p.start.z);

        vtkNew<vtkTransform> transform;
        transform->Translate(p.start.x, p.start.y, p.start.z);

        double defaultDir[3] = {1.0, 0.0, 0.0};
        double targetDir[3] = { (double)p.direction[0], (double)p.direction[1], (double)p.direction[2] };

        double norm = std::sqrt(targetDir[0]*targetDir[0] + targetDir[1]*targetDir[1] + targetDir[2]*targetDir[2]);
        if (norm > 0) {
            targetDir[0] /= norm;
            targetDir[1] /= norm;
            targetDir[2] /= norm;
        }

        double axis[3];
        axis[0] = defaultDir[1]*targetDir[2] - defaultDir[2]*targetDir[1];
        axis[1] = defaultDir[2]*targetDir[0] - defaultDir[0]*targetDir[2];
        axis[2] = defaultDir[0]*targetDir[1] - defaultDir[1]*targetDir[0];

        double dotProd = defaultDir[0]*targetDir[0] + defaultDir[1]*targetDir[1] + defaultDir[2]*targetDir[2];
        if (dotProd < -1.0) dotProd = -1.0;
        else if (dotProd > 1.0) dotProd = 1.0;

        double angleRad = std::acos(dotProd);
        double angleDeg = angleRad * 180.0 / 3.14159265;

        if (std::abs(angleDeg) > 0.1 && std::abs(angleDeg) < 179.9) {
            transform->RotateWXYZ(angleDeg, axis);
        } else if (std::abs(angleDeg) >= 179.9) {
            transform->RotateWXYZ(180, 0, 1, 0);
        }

        transform->Scale(200.0, 200.0, 200.0);

        actors.gazeArrowActor->SetUserTransform(transform);
    }

    for (auto it = actorsMap.begin(); it != actorsMap.end(); ) {
        bool found = false;
        for (int id : activeIDs) {
            if (id == it->first) { found = true; break; }
        }

        if (!found) {
            renderer->RemoveActor(it->second.headActor);
            renderer->RemoveActor(it->second.gazeArrowActor);
            renderer->RemoveActor(it->second.idLabelActor);
            it = actorsMap.erase(it);
        } else {
            ++it;
        }
    }

    renderWindow()->Render();
}

