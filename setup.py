from setuptools import setup, find_packages

setup(
    name="fraud_detection",        # Change to your package's name
    version="0.1.0",
    package_dir={"": "src"},  
    packages=find_packages(),      # Automatically finds submodules
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tables",
        "fastapi",
        "uvicorn",
        "flask",
        # Add any runtime deps here (not dev tools or testing libs)
    ],
    author="Your Name",
    description="A deep learning-based fraud detection system",
    python_requires=">=3.6",
)
