# California Flight Delay Analysis
[![Binder](https://mybinder.org/badge_logo.svg)](
https://mybinder.org/v2/git/https%3A%2F%2Fgithub.com%2FUCB-stat-159-f25%2Ffinal-group21/HEAD?urlpath=%2Fdoc%2Ftree%2Fmain.ipynb
)

## Overview
This project analyzes and models flight delay behavior in California using data from the U.S. Department of Transportation. The main objectives are to look at airline and airport reliability, explore clustering structures among airlines and airports, and evaluate how well we can predict flight delays using statistical and machine learning methods.

## Motivation
Because of the many recent flight delays due to the government shutdown, we found it interesting to take a closer look at how delays behave in practice. This motivated us to analyze which airlines and airports perform best and whether delays can be predicted using historical data.

## Dataset
The dataset used in this project is the U.S. Flight Delays Dataset (@USDOTFlightDelays2015), available on Kaggle:  
https://www.kaggle.com/datasets/usdot/flight-delays  

The data comes from from the Bureau of Transportation Statistics and contains on-time performance records of domestic U.S. flights across 14 airlines and 322 airports during 2015. Due to the large size of the dataset, a subset is used in this project focusing only on flights leaving from California airports.

## Project Website
The project's website can be accessed here:  
https://ucb-stat-159-f25.github.io/final-group21/

## Repository Structure
The repository is structured in the following way:  
INSERT WHEN DONE (describe what is done in each file/directory, and how to navigate between them)

## Setup
Clone the repository:   
```bash
git clone https://github.com/UCB-stat-159-f25/final-group21.git
cd final-group21
```

Create and activate the environment:  
```bash
conda env update -f environment.yml --name proj03 --prune
conda activate proj03
```

Install the IPython kernel:  
```bash
python -m ipykernel install --user --name proj03 --display-name "IPython - proj03"
```

## Usage
Create or update the environment:  
```bash
make env
```

Run all notebooks in the project:  
```bash
make all
```

## Testing
Tests can be executed from the project root using:  
```bash
PYTHONPATH=./ pytest
```

## License
This project is licensed under the BSD 3-Clause License.

## References

