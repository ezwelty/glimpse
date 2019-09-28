import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name='glimpse',
    version='0.1.0b1',
    description='Timelapse camera calibration and surface motion extraction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ezwelty/glimpse',
    author="Ethan Welty & Douglas Brinkerhoff",
    author_email="ethan.welty@gmail.com",
    maintainer="Ethan Welty",
    maintainer_email="ethan.welty@gmail.com",
    license='',
    keywords='glaciology timelapse photogrammetry camera calibration feature tracking',
    project_urls={
        'Documentation': '',
        'Source': 'https://github.com/ezwelty/glimpse',
        'Tracker': 'https://github.com/ezwelty/glimpse/issues'
    },
    packages=[
        'glimpse'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Multimedia :: Graphics :: 3D Rendering',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    python_requires='~=3.6',
    install_requires=[
        'lmfit ~= 0.9, < 0.9.12',
        'lxml ~= 4.2',
        'matplotlib ~= 2.2',
        'numpy ~= 1.14',
        'opencv-contrib-python ~= 3.0, < 3.4.3',
        'pandas ~= 0.22',
        'piexif ~= 1.0',
        'Pillow ~= 5.0.0',
        'progress ~= 1.5',
        'pyproj ~= 1.9',
        'scikit-learn ~= 0.19',
        'scipy ~= 1.0',
        'shapely ~= 1.6',
        'sharedmem ~= 0.3'
    ],
    extras_require = {
        'io': [
            'gdal ~= 2.2, < 2.4.2'
        ],
        'dev': [
            'pytest',
            'sphinx',
            'sphinx-rtd-theme'
        ]
    }
)
