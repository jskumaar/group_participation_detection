#include "gazevisualizer.h"
#include <vtkSphereSource.h>
#include <vtkArrowSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkCamera.h>
#include <vtkTextProperty.h>
#include <cmath>

GazeVisualizer::GazeVisualizer(QWidget *parent) : QVTKOpenGLNativeWidget(parent) {
    // Initialize Renderer
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    setRenderWindow(renderWindow);
    
    renderer = vtkSmartPointer<vtkRenderer>::New();
    renderWindow->AddRenderer(renderer);
    
    renderer->SetBackground(0.1, 0.1, 0.1); // Dark Gray
    
    // Set default camera view
    renderer->GetActiveCamera()->SetPosition(0, 500, 1000);
    renderer->GetActiveCamera()->SetFocalPoint(0, 0, 0);
    renderer->GetActiveCamera()->SetViewUp(0, 1, 0);
    renderer->ResetCamera();
}

GazeVisualizer::~GazeVisualizer() {
// VTK smart pointers handle cleanup
}

void GazeVisualizer::updateData(const std::vector<PanoViewer::gaze>& gazes) {
    // Track active IDs to remove old ones
    std::vector<int> activeIDs;
    
    for (const auto& p : gazes) {
        activeIDs.push_back(p.personID);
        
        // Check if actor exists
        if (actorsMap.find(p.personID) == actorsMap.end()) {
            PersonActors actors;
            
            // Create Head Actor (Sphere)
            vtkNew<vtkSphereSource> sphereSource;
            sphereSource->SetRadius(15.0); // 15cm radius head
            sphereSource->SetThetaResolution(16);
            sphereSource->SetPhiResolution(16);
            
            vtkNew<vtkPolyDataMapper> sphereMapper;
            sphereMapper->SetInputConnection(sphereSource->GetOutputPort());
            
            actors.headActor = vtkSmartPointer<vtkActor>::New();
            actors.headActor->SetMapper(sphereMapper);
            actors.headActor->GetProperty()->SetColor(0.0, 0.0, 1.0); // Blue
            
            renderer->AddActor(actors.headActor);
            
            // Create Gaze Arrow Actor
            vtkNew<vtkArrowSource> arrowSource;
            
            vtkNew<vtkPolyDataMapper> arrowMapper;
            arrowMapper->SetInputConnection(arrowSource->GetOutputPort());
            
            actors.gazeArrowActor = vtkSmartPointer<vtkActor>::New();
            actors.gazeArrowActor->SetMapper(arrowMapper);
            actors.gazeArrowActor->GetProperty()->SetColor(1.0, 0.0, 0.0); // Red
            
            renderer->AddActor(actors.gazeArrowActor);
            
            // Create ID Label Actor
            actors.idLabelActor = vtkSmartPointer<vtkBillboardTextActor3D>::New();
            actors.idLabelActor->SetInput(std::to_string(p.personID).c_str());
            actors.idLabelActor->GetTextProperty()->SetFontSize(12); // Virtual size
            actors.idLabelActor->GetTextProperty()->SetColor(1.0, 1.0, 1.0); // White
            actors.idLabelActor->SetDisplayOffset(0, 30); // Offset in pixels? No, world usually for 3D props, but Billboard often behaves like 2D offset. Let's try positioning first.
            // Actually BillboardTextActor3D follows world coordinates anchor but faces camera.
            // FontSize is in points if used with vtkTextActor3D, but Billboard might scale differently.
            // Let's set typical values.
            
            // For vtkBillboardTextActor3D, FontSize is irrelevant if we don't scale? 
            // It draws text at the position.
            actors.idLabelActor->GetTextProperty()->SetFontSize(24);
            actors.idLabelActor->GetTextProperty()->SetJustificationToCentered();
            
            renderer->AddActor(actors.idLabelActor);

            actorsMap[p.personID] = actors;
        }
        
        PersonActors& actors = actorsMap[p.personID];

        // Update Head Position
        // Assuming p.start is (x, y, z)
        actors.headActor->SetPosition(p.start.x, p.start.y, p.start.z);
        
        // Update Label Position (above head)
        // Adjust Height: Head is at Y, radius 15. Let's put label at Y + 40
        actors.idLabelActor->SetPosition(p.start.x, p.start.y + 5.0, p.start.z);
        // Force text update if ID could change for same slot (unlikely here but safe practice)
        // actors.idLabelActor->SetInput(std::to_string(p.personID).c_str());
        
        // Update Arrow
        // vtkArrowSource points along +X by default. Length 1.
        // We need to scale it and rotate it to match direction.
        
        vtkNew<vtkTransform> transform;
        transform->Translate(p.start.x, p.start.y, p.start.z);
        
        // Calculate orientation
        // Default direction is (1, 0, 0)
        double defaultDir[3] = {1.0, 0.0, 0.0};
        double targetDir[3] = { (double)p.direction[0], (double)p.direction[1], (double)p.direction[2] };

        // Normalize target (should be already, but safety first)
        double norm = std::sqrt(targetDir[0]*targetDir[0] + targetDir[1]*targetDir[1] + targetDir[2]*targetDir[2]);
        if (norm > 0) {
           targetDir[0] /= norm;
           targetDir[1] /= norm;
           targetDir[2] /= norm;
        }
        
        // Cross product to get rotation axis
        double axis[3];
        axis[0] = defaultDir[1]*targetDir[2] - defaultDir[2]*targetDir[1];
        axis[1] = defaultDir[2]*targetDir[0] - defaultDir[0]*targetDir[2];
        axis[2] = defaultDir[0]*targetDir[1] - defaultDir[1]*targetDir[0];
        
        // Calculate dot product and clamp to [-1, 1] to avoid domain error (NaN)
        double dotProd = defaultDir[0]*targetDir[0] + defaultDir[1]*targetDir[1] + defaultDir[2]*targetDir[2];
        if (dotProd < -1.0) dotProd = -1.0;
        else if (dotProd > 1.0) dotProd = 1.0;

        double angleRad = std::acos(dotProd);
        double angleDeg = angleRad * 180.0 / 3.14159265;

        // If vectors are parallel (angle 0) or anti-parallel (angle 180), handle degenerate axis
        if (std::abs(angleDeg) > 0.1 && std::abs(angleDeg) < 179.9) {
             transform->RotateWXYZ(angleDeg, axis);
        } else if (std::abs(angleDeg) >= 179.9) {
             // Rotate 180 around up vector? or specific axis
             transform->RotateWXYZ(180, 0, 1, 0); 
        }

        transform->Scale(200.0, 200.0, 200.0); // Scale length to 200 units (2 meters)
        
        actors.gazeArrowActor->SetUserTransform(transform);
    }
    
    // Remove stale actors
    for (auto it = actorsMap.begin(); it != actorsMap.end(); ) {
        bool found = false;
        for (int id : activeIDs) {
            if (id == it->first) {
                found = true;
                break;
            }
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
