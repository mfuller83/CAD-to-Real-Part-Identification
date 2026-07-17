# CAD-to-Real Part Identification

> 🚧 **Project Status: Work in Progress**
>
> This is an experimental project exploring cross-modal representation learning between 3D CAD geometry and 2D images. The project is under active development, and the architecture and training methodology are expected to evolve as the project progresses.

## Overview

The intention of this project is to develop a neural network capable of identifying a physical component in a camera image based on its corresponding CAD data.

The overall goal is to create a link between **3D CAD geometry** and **real-world visual observations**, allowing a system to recognise physical components using their original engineering CAD models as references.

Rather than training a conventional image classifier with a fixed set of component classes, the proposed system will learn a **shared embedding space** between 3D geometry and 2D images.

This could potentially allow previously unseen CAD models to be added to the reference database without requiring the complete network to be retrained as a conventional classification model.

## Proposed Architecture

The system will use two separate neural network encoders:

* A **3D encoder** that processes CAD-derived point-cloud data.
* An **image encoder** that processes 2D images of components.

Both encoders will produce embedding vectors of the same size within a shared embedding space.

The training objective is to place embeddings representing the **same physical component close together**, while embeddings representing **different components are pushed further apart**.

Conceptually:

```text
                    ┌─────────────────┐
CAD Geometry ──────►│ Point Cloud     │──────► CAD Embedding
(.xyz)              │ Encoder         │
                    └─────────────────┘
                                                   │
                                                   │ Shared
                                                   │ Embedding
                                                   │ Space
                                                   │
                    ┌─────────────────┐            │
Component Image ──► │ Image Encoder   │──────► Image Embedding
                    └─────────────────┘
```

The initial 3D encoder is expected to use a PointNet-style architecture, while the image encoder will use a convolutional or vision-based neural network architecture.

These choices may change as the project develops.

## Training Strategy

The initial training approach will use metric learning to align CAD geometry and images within the shared embedding space.

For example:

```text
Anchor:   CAD point cloud for Part A
Positive: Image of Part A
Negative: Image of Part B
```

Using cosine similarity, a triplet-style loss can be expressed conceptually as:

```text
loss = max(
    0,
    similarity(A_cad, B_image)
    - similarity(A_cad, A_image)
    + margin
)
```

The objective is therefore to:

* Maximise the cosine similarity between a CAD embedding and images of the same component.
* Minimise the cosine similarity between a CAD embedding and images of different components.

Alternative metric-learning and contrastive-learning approaches may also be investigated as the project develops.

## Dataset

The development dataset consists of:

* 3D point-cloud files stored in `.xyz` format.
* Rendered images of components captured from multiple viewing angles.

The point-cloud data is generated directly from engineering CAD geometry. This provides clean and complete geometric information without the missing points, occlusion, sensor noise, or measurement errors that may occur when point clouds are captured from physical objects using real-world sensors.

During the initial development stage, the image dataset will also consist primarily of images rendered from CAD models.

This allows the basic shared-embedding architecture and training process to be developed and evaluated in a controlled environment before introducing the additional complexity of real-world camera images.

The longer-term objective is to train and evaluate the image encoder using photographs of physical components while continuing to use the original CAD-derived geometry as the 3D reference.

### Dataset Availability

The primary dataset used during development is proprietary and is **not included in this repository**.

The source CAD models, derived point clouds, rendered images, real-world images, and other confidential data will remain separate from the public repository.

The project code is intended to remain independent of the specific dataset used for development.

Where possible, publicly releasable or synthetic example CAD models and data may be added in the future to demonstrate the complete training and inference pipeline.

## Embedding Space

Both the 3D encoder and image encoder will output vectors of the same length.

The initial proposed embedding size is approximately:

```text
700 dimensions
```

This is currently an experimental starting point rather than a fixed architectural requirement.

The optimal embedding dimension will need to be determined experimentally. A smaller embedding may be sufficient and could improve:

* Training efficiency.
* Storage requirements.
* Similarity-search performance.
* Generalisation.

The effect of embedding dimensionality will therefore form part of the experimental evaluation of the system.

## Proposed Workflow

```text
                TRAINING

       CAD Model                Component Image
           │                          │
           ▼                          ▼
    Point Cloud (.xyz)           Image Processing
           │                          │
           ▼                          ▼
    3D Point Encoder             Image Encoder
           │                          │
           ▼                          ▼
      CAD Embedding             Image Embedding
           │                          │
           └────────────┬─────────────┘
                        │
                        ▼
                 Similarity / Loss
                        │
                        ▼
                   Optimisation
```

Once trained, the intended identification workflow is:

```text
                IDENTIFICATION

Reference CAD Database              Camera Image
         │                               │
         ▼                               ▼
    3D Encoder                      Image Encoder
         │                               │
         ▼                               ▼
  CAD Embeddings                  Image Embedding
         │                               │
         └───────────────┬───────────────┘
                         │
                         ▼
                  Cosine Similarity
                         │
                         ▼
               Nearest CAD Embedding
                         │
                         ▼
                Predicted Component
```

The intention is that CAD embeddings could eventually be generated once and stored within a searchable reference database.

When a new camera image is received, the image encoder would generate an embedding and compare it against the stored CAD embeddings to identify the most similar component.

## Current Development Goals

* [ ] Finalise initial project and dataset structure.
* [ ] Develop CAD-to-point-cloud generation pipeline.
* [ ] Develop automated multi-view CAD image generation pipeline.
* [ ] Implement point-cloud dataset loader.
* [ ] Implement image dataset loader.
* [ ] Implement initial 3D point-cloud encoder.
* [ ] Implement initial image encoder.
* [ ] Project both encoder outputs into a shared embedding space.
* [ ] Implement cosine-similarity-based training.
* [ ] Implement triplet or contrastive loss.
* [ ] Train using CAD-derived point clouds and rendered images.
* [ ] Visualise and evaluate the learned embedding space.
* [ ] Evaluate component retrieval accuracy.
* [ ] Investigate optimal embedding dimensionality.
* [ ] Introduce real-world component photographs.
* [ ] Investigate methods for reducing the synthetic-to-real domain gap.

## Generating STEP Views

`step_view_generator.py` is intended to run inside FreeCAD because rendering
requires its GUI. Add it as a FreeCAD macro and run it to select a STEP model
and output folder interactively. By default it creates front, rear, left,
right, top, bottom, and isometric PNG images plus a JSON manifest.

It can also be called from another FreeCAD Python script:

```python
from step_view_generator import generate_views

output_directory = generate_views(
    "/path/to/part.step",
    "/path/to/rendered_views",
    views=("front", "right", "top", "isometric"),
    image_size=(1024, 1024),
    background="White",
)
```

## Long-Term Objectives

The long-term objective is to investigate whether a model can learn a sufficiently general relationship between **3D engineering geometry** and **2D visual appearance** to identify physical components directly from their CAD data.

Potential future areas of investigation include:

* Real-world camera images.
* Synthetic-to-real domain adaptation.
* Data augmentation and domain randomisation.
* Alternative 3D representations.
* PointNet and PointNet++ architectures.
* Alternative image encoders.
* Contrastive learning approaches.
* Hard-negative mining.
* Approximate nearest-neighbour embedding search.
* Zero-shot identification of previously unseen components.

## Repository Philosophy

This repository is being developed as an ongoing engineering and machine-learning project rather than as a finished software product.

The project is intended to document the development process, experiments, architectural decisions, successes, and failures involved in exploring CAD-to-real-world representation learning.

As a result, the codebase, architecture, and documentation will evolve as new approaches are tested and the project develops.
