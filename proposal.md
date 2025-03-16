## Project Overview
We use the Waymo dataset for the project "Scene Classification and Visualization Tools Based on Waymo Dataset". This dataset contains rich camera images and LiDAR point cloud data, making it ideal for developing interactive visualization tools or demonstrating scene classification (e.g., urban, suburban, nighttime, etc.).

## Implementation Steps
1. **Data Download and Preparation**  
   - Download the dataset from [Waymo Open Dataset](https://waymo.com/open/), or directly use the sample link provided by Foxglove Studio: [Waymo Example](https://app.foxglove.dev/~/view?ds=foxglove-sample-stream&ds.recordingId=rec_0dHYwkGj9g7eA9DE&ds.overrideLayoutId=df51964b-b84a-4e12-a11e-067e2fce9c1c).  
   - If custom processing is needed, refer to the guide [Converting Waymo Open Dataset to MCAP Format](https://foxglove.dev/blog/converting-the-waymo-open-dataset-to-mcap).

2. **Tool Usage** 
   - Use Foxglove Studio to create custom panels displaying scene classification results. Refer to [Foxglove Docs](https://docs.foxglove.dev/docs) to learn about extension development.  
   - Use Python libraries (e.g., Pandas, Matplotlib) for data processing. Code examples can be found at [Waymo Open Dataset GitHub](https://github.com/waymo-research/waymo-open-dataset).

3. **Report and Presentation**  
   - Prepare a 3-4 page proposal slide deck covering project title, team information, problem description (scene classification), and motivation (improving autonomous driving performance).  
   - For the final presentation, prepare a 10-page slide deck addressing contributions, novelty, visualization techniques, etc.  
   - Write a 4-6 page report using the NeurIPS template (e.g., via Overleaf), including contributions, novelty, and other required sections.

## Contribution and Novelty
- **Contribution**:  
  - Develop a scene classification tool to enhance the usability of the Waymo dataset, supporting urban planning and autonomous driving research.  
  - Create new visualization methods allowing users to dynamically explore different driving scenes.  
- **Novelty**:  
  - Propose an interactive scene classification visualization tool, which is rare among existing tools.  
  - Innovate analysis methods, such as extracting scene features with machine learning models and visualizing key driving scenarios.

## Dataset Selection and Analysis
We selected the Waymo dataset for this project, which is the largest and most diverse dataset in autonomous driving. It includes camera images, LiDAR point clouds, and other sensor data. The dataset is accessible via the [Foxglove Examples page](https://foxglove.dev/examples) at [Waymo Example](https://app.foxglove.dev/~/view?ds=foxglove-sample-stream&ds.recordingId=rec_0dHYwkGj9g7eA9DE&ds.overrideLayoutId=df51964b-b84a-4e12-a11e-067e2fce9c1c).

The Waymo dataset's diversity and scale make it ideal for showcasing various visualization techniques such as scatter plots, heatmaps, and 3D visualizations, making it particularly suitable for autonomous driving-related projects.

## Project Proposal Details
- **Team Members**: List team member names.  
- **Problem Description**: The Waymo Open Dataset provides rich data from autonomous vehicles, including LiDAR and camera data with labeled objects. Visualizing the distribution of different driving scenes (e.g., urban, suburban, nighttime) within this data can offer valuable insights into traffic patterns and autonomous vehicle behavior.  
- **Motivation**: Understanding the distribution of various scenes allows for better evaluation of autonomous vehicle performance and identification of areas needing improvement in perception and decision-making systems.  
- **Data Source**: Waymo Open Dataset, accessible via the Foxglove example page: [Waymo Example](https://app.foxglove.dev/~/view?ds=foxglove-sample-stream&ds.recordingId=rec_0dHYwkGj9g7eA9DE&ds.overrideLayoutId=df51964b-b84a-4e12-a11e-067e2fce9c1c) or downloadable from [Waymo Open Dataset](https://waymo.com/open/).  
- **Tools**: Foxglove Studio for visualization, Python and its libraries (e.g., Pandas, NumPy, Matplotlib) for data processing and custom visualization, and the Waymo Open Dataset GitHub repository for data processing code.  
- **Contribution**: This project will contribute a scene classification and visualization tool, enabling users to explore and understand different driving scenes in the Waymo dataset, aiding researchers in autonomous driving by providing insights into real-world traffic scenarios.  
- **Novelty**: The project's uniqueness lies in its focus on scene classification visualization, combining statistical analysis with interactive tools to offer a comprehensive view for various research purposes.  
- **Technical Implementation Plan**:  
  - **Data Processing Module**: Extract scene-related information (e.g., time, location, weather) from the Waymo dataset, classify scenes by type, and process data using Python libraries.  
  - **Visualization Development**: Create custom panels in Foxglove Studio to display scene classification results, develop Python scripts for statistical visualizations (e.g., heatmaps, bar charts), and implement 3D point cloud visualizations.  
  - **Integration and Interaction**: Integrate processed data into Foxglove Studio for interactive visualization, allowing users to filter conditions (e.g., time, location) via the interface to view distributions.  
  - **Report and Presentation**: Prepare a 3-4 page proposal slide deck and a 10-page final presentation slide deck, and write a 4-6 page report using the NeurIPS template (e.g., Overleaf), ensuring all required sections are covered.  
- **Visualization Techniques**: Employ a combination of 2D and 3D visualization techniques, such as heatmaps for scene density, bar charts for scene type comparison, and 3D point cloud visualizations for spatial distribution.  
- **Future Work**: Include real-time scene classification capabilities, integration with other datasets, and advanced machine learning analysis to predict scene changes.  
- **References**: Waymo Open Dataset documentation, Foxglove Studio documentation, and related research papers on autonomous driving data visualization.  
- **Team Member Contributions**: Outline specific tasks for each team member.

## Contribution and Novelty Analysis
- **Contribution**:  
  - Develop a scene classification tool to enhance the Waymo dataset's utility, supporting urban planning and autonomous driving research.  
  - Create new visualization methods enabling dynamic exploration of driving scenes.  
- **Novelty**:  
  - Propose an interactive scene classification visualization tool, a feature uncommon in existing tools.  
  - Innovate analysis methods, such as extracting scene features with machine learning models and visualizing critical driving scenarios.

## Code Resources and Guidance
- **Waymo Open Dataset Documentation**: Visit [Waymo Open Dataset](https://waymo.com/open/) for dataset details and download it via the GitHub repository [Waymo Open Dataset GitHub](https://github.com/waymo-research/waymo-open-dataset) for tutorials.  
- **Foxglove Studio Documentation**: Access [Foxglove Docs](https://docs.foxglove.dev/docs) to learn how to visualize data with Foxglove Studio.  
- **ROS 2 Data Visualization Guide**: Refer to [ROS 2 and Foxglove Studio](https://docs.ros.org/en/foxy/How-To-Guides/Visualizing-ROS-2-Data-With-Foxglove-Studio.html) to connect ROS 2 stacks and visualize data.  
- **GitHub Resources**: Foxglove Studio source code is available at [Foxglove Studio GitHub](https://github.com/foxglove/studio) for in-depth study or extension development. The foxglove-python library [foxglove-python GitHub](https://github.com/foxglove/foxglove-python) provides a Python interface for data processing.  
- **Installation and Usage**: Download Foxglove Studio from [Foxglove Studio Download](https://foxglove.dev/download), supporting Linux, Windows, and macOS for easy setup.  
- **Python Libraries**: Use Matplotlib, NumPy, and Pandas for custom visualizations.

## Key References
- [Waymo Open Dataset Overview](https://waymo.com/open/)  
- [Waymo Open Dataset GitHub Repository](https://github.com/waymo-research/waymo-open-dataset)  
- [Foxglove Studio Download Page](https://foxglove.dev/download)  
- [Foxglove Studio Documentation](https://docs.foxglove.dev/docs)  
- [foxglove-python GitHub Repository](https://github.com/foxglove/foxglove-python)  
- [ROS 2 Documentation Guide](https://docs.ros.org/en/foxy/How-To-Guides/Visualizing-ROS-2-Data-With-Foxglove-Studio.html)  
- [Waymo Example on Foxglove Platform](https://app.foxglove.dev/~/view?ds=foxglove-sample-stream&ds.recordingId=rec_0dHYwkGj9g7eA9DE&ds.overrideLayoutId=df51964b-b84a-4e12-a11e-067e2fce9c1c)  
- [Matplotlib Documentation](https://matplotlib.org/)  
- [NumPy Documentation](https://numpy.org/)  
- [Pandas Documentation](https://pandas.pydata.org/)  
- [Converting Waymo Open Dataset to MCAP Guide](https://foxglove.dev/blog/converting-the-waymo-open-dataset-to-mcap)
