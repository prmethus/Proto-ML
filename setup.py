from setuptools import setup

setup(name='protoml',
    version='1.0.0',
    description="Creating Prototype ML models ASAP!",
    author='prmethus',
    packages=['protoml'],
    install_requires=['joblib>=1.1.0','matplotlib>=3.5.2','pandas>=1.4.2','scikit_learn>=1.1.0','seaborn>=0.11.2','string_color>=1.2.1','xgboost>=1.6.1']
    )