# Scene Classification and Visualization Tools Design Based on Waymo Dataset

## Project Overview
We recommend using the Waymo dataset for the project "Design of Scene Classification and Visualization Tools for the Waymo Dataset". The dataset contains rich camera images and LiDAR point cloud data, making it ideal for developing interactive visualization tools or demonstrating scene classification (such as urban, suburban, nighttime scenes, etc.).

---

## Implementation Steps

### 1. Data Download and Preparation
- Download data from [Waymo Open Dataset](https://waymo.com/open/)
- Or directly use the sample link provided by Foxglove Studio: [Waymo Example](https://app.foxglove.dev/~/view?ds=foxglove-sample-stream&ds.recordingId=rec_0dHYwkGj9g7eA9DE&ds.overrideLayoutId=df51964b-b84a-4e12-a11e-067e2fce9c1c)
- For custom processing, refer to the guide: [Converting Waymo Open Dataset to MCAP Format](https://foxglove.dev/blog/converting-the-waymo-open-dataset-to-mcap)

### 2. Tool Usage
- Use Foxglove Studio to create custom panels displaying scene classification results
- Refer to [Foxglove Documentation](https://docs.foxglove.dev/docs) for extension development
- Use Python libraries (such as Pandas, Matplotlib) for data processing
- Code examples can be found at [Waymo Open Dataset GitHub](https://github.com/waymo-research/waymo-open-dataset)

### 3. Report and Presentation
- Prepare 3-4 page project proposal slides, including:
  - Project title
  - Team information
  - Problem description (scene classification)
  - Project motivation (improving autonomous driving performance)
- Prepare 10-page final presentation slides, including:
  - Project contributions
  - Innovations
  - Visualization techniques, etc.
- Write a 4-6 page report using NeurIPS template (e.g., via Overleaf), including:
  - Project contributions
  - Innovations
  - Other required sections

---

## Contribution and Innovation Suggestions

### Project Contributions
- Develop scene classification tools to enhance the utility of Waymo dataset
- Support urban planning and autonomous driving research
- Create new visualization methods to enable dynamic exploration of different driving scenes

### Innovations
- Propose interactive scene classification visualization tools
- Innovative analysis methods, such as:
  - Using machine learning models to extract scene features
  - Visualizing critical driving scenarios

---

## Dataset Selection and Analysis

### Waymo Dataset Features
- The largest and most diverse dataset in autonomous driving
- Contains sensor data such as camera images and LiDAR point clouds
- Accessible via [Foxglove Examples Page](https://foxglove.dev/examples): [Waymo Example](https://app.foxglove.dev/~/view?ds=foxglove-sample-stream&ds.recordingId=rec_0dHYwkGj9g7eA9DE&ds.overrideLayoutId=df51964b-b84a-4e12-a11e-067e2fce9c1c)

### Dataset Advantages
- Diversity and scale suitable for demonstrating various visualization techniques
- Supports multiple formats including scatter plots, heatmaps, 3D visualizations
- Particularly suitable for autonomous driving related projects

---

## Technical Implementation Plan

### Data Processing Module
- Extract scene-related information (time, location, weather, etc.) from Waymo dataset
- Classify scenes by type
- Process data using Python libraries

### Visualization Development
- Create custom panels in Foxglove Studio to display scene classification results
- Develop Python scripts for statistical visualizations (heatmaps, bar charts, etc.)
- Implement 3D point cloud visualization

### Integration and Interaction
- Integrate processed data into Foxglove Studio
- Enable users to filter conditions (time, location, etc.) through the interface to view distributions

### Report and Presentation
- Prepare 3-4 page proposal slides
- Prepare 10-page final presentation slides
- Write 4-6 page report using NeurIPS template

---

## Key Resources

### Dataset Resources
- [Waymo Open Dataset](https://waymo.com/open/)
- [Waymo Open Dataset GitHub](https://github.com/waymo-research/waymo-open-dataset)

### Visualization Tools
- [Foxglove Studio Download](https://foxglove.dev/download)
- [Foxglove Documentation](https://docs.foxglove.dev/docs)
- [Foxglove Studio GitHub](https://github.com/foxglove/studio)

### Python Libraries
- [Matplotlib Documentation](https://matplotlib.org/)
- [NumPy Documentation](https://numpy.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
