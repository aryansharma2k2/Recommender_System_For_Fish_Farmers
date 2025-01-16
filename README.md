# Fish Breeder Recommender System & Logistics Optimization

## Overview
This project focuses on developing an intelligent **Recommender System** tailored for **Indian fish breeders**. The system integrates multiple data sources, such as **meteorological data**, **chemical profiles**, and **logistics data**, to predict material requirements and costs, optimizing the fish farming process. Additionally, it includes a **sales forecasting pipeline** and **logistics optimization algorithm** to improve operational efficiency.

---

## Key Features

1. **Recommender System for Fish Breeders:**
   - Analyzes meteorological, chemical, and logistical data to predict the optimal materials and resources required for fish farming.
   - Developed using **Python** and **JavaScript**.

2. **Data Ingestion Pipeline:**
   - Automated the collection and ingestion of sales data from the **FMPIS - National Fisheries Database**.
   - Utilized the **ARIMA** model to provide accurate **nine-month sales forecasting** for optimal inventory management.
   - Developed using **Python**.

3. **Sales Data Analytics Framework:**
   - Built an analytical framework that evaluates **historical sales data** and identifies the optimal fish species based on **geospatial analytics** and **facility constraints**.
   - Data mining techniques were used for pattern recognition and decision support.
   - Developed using **R** for data mining and analysis.

4. **Logistics Optimization Algorithm:**
   - Implemented a **logistics optimization algorithm** using **graph theory** to identify efficient **delivery routes** for transporting materials.
   - Incorporated predictive analytics for **evaporation loss** and **daily nutrient requirements**, improving resource allocation.
   - Developed using **Python**.

---

## Technologies Used

- **Programming Languages:** Python, R, JavaScript
- **Libraries & Frameworks:** 
  - **Python**: Pandas, NumPy, Scikit-learn, Statsmodels (ARIMA), Flask
  - **JavaScript**: for web-related features
- **Data Processing & Forecasting:** ARIMA model, Data Mining, Geospatial Analytics
- **Optimization Techniques:** Graph Theory for Logistics, Predictive Analytics

---

## Project Workflow

1. **Data Collection**:
   - Collected and ingested data from the **National Fisheries Database** and other external sources, automating data pipelines to ensure the latest information is always available.
   
2. **Sales Forecasting**:
   - Applied **ARIMA** model to historical sales data for accurate forecasting over a 9-month horizon.

3. **Species Selection**:
   - Used geospatial analysis to recommend optimal fish species based on the farm's location, available resources, and market trends.

4. **Logistics Optimization**:
   - Developed algorithms to optimize delivery routes, reducing cost and time for material transport while factoring in evaporation losses and nutrient requirements.

---

## Getting Started

### Prerequisites
To run this project locally, you'll need:
- **Python** (Version 3.7+)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/aryansharma2k2/Recommender_System_For_Fish_Farmers.git
   cd Recommender_System_For_Fish_Farmers
   python3 app.py
