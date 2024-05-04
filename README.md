# Crop Recommendation System

This repository contains the source code for a Streamlit-based web application that recommends the best crops based on environmental and soil conditions using machine learning models.

## Features

- Predicts the most suitable crops based on user input regarding soil nutrients (N, P, K), environmental factors (temperature, humidity, pH, rainfall), and other derived factors.
- Utilizes multiple machine learning models for prediction to compare results.
- Provides an interactive user interface to easily modify input parameters and view predictions.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Aiaxis/crops-predicts.git
cd crops-predicts
  ```
2.**Create a virtual environment:**

  ```bash
  python -m venv venv
  
  ```
3.**Create a virtual environment:**

  * **Windows**
    ```bash
    .\venv\Scripts\activate
    ```
  
 *  **Mac**
    ```bash
    source venv/bin/activate
    ```

4.**Install dependencies:**

  ```bash
  pip install -r requirements.txt
  
  ```

```bash
streamlit run app.py
```

Now, the app should be running locally on your machine.
