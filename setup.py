from distutils.core import setup

setup(
    name = 'phase_retrieval',
    packages = ['phase_retrieval','phase_retrieval.cpu','phase_retrieval.gpu'],
    version = '1.0.0',
    description = 'Runs three phase retrieval algorithms on sample apperatures and produces visual output',
    author = 'Austin Jones',
    author_email = 'austinjones051@gmail.com',
	url = 'https://github.com/automorphisms-of-a-square/phase_retrieval',
	requires = ['scipy','pyqt5','matplotlib','reikna'],
	classifiers = ["Programming Language :: Python",
	               "Programming Language :: Python :: 3",
				   "License :: GPLv3 License"]
)