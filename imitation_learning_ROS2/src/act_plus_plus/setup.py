from setuptools import find_packages, setup

package_name = 'act_plus_plus'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='moon',
    maintainer_email='1095286992@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'eval_jk_robot = act_plus_plus.eval_jk_robot:main',
            'act_plus_test = act_plus_plus.act_plus_test:main',
        ],
    },
)
