from setuptools import setup, find_packages

setup(
    name="rag-assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langchain==0.0.350",
        "chromadb==0.4.18",
        "sentence-transformers==2.2.2",
        "gradio==4.8.0",
        "torch==2.1.1",
        "transformers==4.35.2",
        "python-dotenv==1.0.0"
    ]
)