# AutomatedML

AutomatedML is a modular framework for automated machine learning, supporting neural network architectures, model evaluation, optimization, and data handling.

# Citation 

A General-Purpose Neural Architecture Search Algorithm for Building Deep Neural Networks
https://doi.org/10.1007/978-3-031-62922-8_9

@inbook{Zito2024,
  title = {A General-Purpose Neural Architecture Search Algorithm for Building Deep Neural Networks},
  ISBN = {9783031629228},
  ISSN = {1611-3349},
  url = {http://dx.doi.org/10.1007/978-3-031-62922-8_9},
  DOI = {10.1007/978-3-031-62922-8_9},
  booktitle = {Metaheuristics},
  publisher = {Springer Nature Switzerland},
  author = {Zito,  Francesco and Cutello,  Vincenzo and Pavone,  Mario},
  year = {2024},
  pages = {126–141}
}

## Features

- Modular component factory for ML pipelines
- Data container utilities
- Model evaluation and optimization
- Support for custom layers and models
- Extensible engine and utility modules

## Installation

Clone the repository and install dependencies:

```powershell
# Clone the repository
git clone <repo-url>
cd automl
# Install dependencies
pip install -e .
```

### Configure Python Virtual Environment (Linux/macOS)

You can use the following shell commands to set up a virtual environment and install the package:

```sh
# Create a virtual environment
python3 -m venv venv
# Activate the virtual environment
source venv/bin/activate
# Install the package
pip install -e .
```

## Directory Structure

```
src/automatedML/      # Core library modules
    ann/              # Neural network components
    annmodels/        # Predefined models
    component/        # Pipeline components
    engine/           # Execution engine
    flatgenerator/    # Model generator
    models/           # Model definitions
    optimizator/      # Optimization algorithms
    utils/            # Utility functions

tests/                # Test scripts and datasets
anns/                 # Example neural network configs and images
datasets/             # Example datasets
plots/                # Generated plots and visualizations
```

## License

See `LICENSE` for details.
