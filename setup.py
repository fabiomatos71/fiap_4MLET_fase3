from setuptools import setup, find_packages

setup(
    name="fase3_fiap_4mlet",
    version="0.1.0",
    description="Projeto Fase 3 Tech Challenge - FIAP 4MLET",
    author="Fabio Vargas Matos",
    packages=find_packages(),  # Encontra todos os pacotes Python (com __init__.py)
    install_requires=[
        # Dependências principais do projeto (opcional, já que você tem requirements.txt)
        # 'numpy',
        # 'pandas',
        # 'scikit-learn',
        # 'tensorflow',
    ],
    include_package_data=True,  # Inclui arquivos do MANIFEST, se houver
    python_requires=">=3.12",
)
