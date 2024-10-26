## Dominos - Predictive Purchase Order System

Dominos wants to optimize the process of ordering ingredients by predicting future sales and creating a purchase order. By accurately forecasting sales, Dominos can ensure that it has the right amount of ingredients in stock, minimizing waste and preventing stockouts. This project aims to leverage historical sales data and ingredient information to develop a predictive model and generate an efficient purchase order system.


## Table of Contents
- [Dominos - Predictive Purchase Order System](#dominos---predictive-purchase-order-system)
- [Table of Contents](#table-of-contents)
- [Setup](#setup)
  - [Softwares needed](#softwares-needed)
  - [Code](#code)
  - [Python packages](#python-packages)
  - [Environment\_variables](#environment_variables)
  - [Database Setup](#database-setup)
  - [Run App](#run-app)

## Setup
### Softwares needed
1. IDE (VS Code)
2. Jupyter Notebook
3. Python
4. Git (with git bash)


### Code

<!-- Clone this repository and ```cd``` into that directory
``` 
git clone https://github.com/Gengatharan1/ds-dominos_purchase_order_analysis.git

cd ds-dominos_purchase_order_analysis
``` -->

- Download project folder or full repo
- cd into the folder
```
cd ds-dominos_purchase_order_analysis
```

### Python packages

Install all necessary packages
``` 
pip install -r requirements.txt
```

### Environment_variables
Creating ```.env``` file using template
``` 
cp .env_template .env
```

### Database Setup

Create database and table in PostgreSQL and add copy its credentials in ```.env``` file.

### Run App
``` 
streamlit run app.py
```


---
^ [Back to table of contents](#table-of-contents)