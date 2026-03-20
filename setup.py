from setuptools import find_packages, setup


setup(
    name="general-user-model-experiment",
    version="0.1.0",
    description="Open-source general user model from computer-use telemetry",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.1",
        "scikit-learn>=1.4",
        "joblib>=1.3",
        "fastapi>=0.110",
        "uvicorn>=0.29",
        "pydantic>=2.6",
        "python-multipart>=0.0.9",
        "streamlit>=1.34",
        "plotly>=5.20",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "httpx>=0.27",
        ]
    },
)
