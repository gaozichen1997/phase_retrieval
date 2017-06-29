from setuptools import setup 

setup(
    name = 'phase_retrieval',
    packages = ['phase_retrieval','phase_retrieval.cpu','phase_retrieval.gpu','phase_retrieval.data'],
    version = '1.0.0',
    description = 'Runs three phase retrieval algorithms on sample apperatures and produces visual output',
    author = 'Austin Jones',
    author_email = 'austinjones051@gmail.com',
    license = 'GPL',
    url = 'https://github.com/automorphisms-of-a-square/phase_retrieval',
    install_requires = ['scipy','matplotlib','reikna'],
    python_requires = '>=3.0, <4',
    package_data = {'phase_retrieval':['phase_retrieval/data/squares.gif','LICENSE', 'Manifest.in', 'README.txt']},
    include_package_data=True,
    classifiers = ["Programming Language :: Python",
	               "Programming Language :: Python :: 3",
                   "License :: osi approved :: GPLv3 License"],
    entry_points = {
        'console_scripts':['phase_retrieval=phase_retrieval.run:start']
        }
)
