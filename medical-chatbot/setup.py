from setuptools import setup, find_packages

setup(
    name='medical-chatbot',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A deep learning-based medical chatbot that suggests medications based on symptoms.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'Flask',
        'tensorflow',
        'torch',
        'transformers',
        'pandas',
        'numpy',
        'scikit-learn',
        'nltk',
        'requests',
        'beautifulsoup4'
    ],
    entry_points={
        'console_scripts': [
            'medical-chatbot=main:main',
        ],
    },
)