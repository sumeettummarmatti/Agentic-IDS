from setuptools import setup, find_packages

setup(
    name="agentic-ids",
    version="0.1.0",
    description="Agentic AI-powered Intrusion Detection System",
    author="Sumeet",
    author_email="sumeettummarmatti@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "gymnasium>=0.28.0",
        "stable-baselines3>=2.0.0",
        "langchain>=0.1.0",
        "langchain-groq>=0.0.10",
        "ollama>=0.1.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "jupyter>=1.0.0"],
    },
)
