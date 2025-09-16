import os
import setuptools
import shutil
import subprocess
from setuptools import find_packages
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import CppExtension, CUDA_HOME
os.environ['MO_JIT_DEBUG'] = '1'
os.environ['MO_JIT_PRINT_COMPILER_COMMAND'] = '1'
current_dir = os.path.dirname(os.path.realpath(__file__))
conda_prefix = os.environ.get("CONDA_PREFIX", "")
conda_include = os.path.join(conda_prefix, "include")
conda_lib = os.path.join(conda_prefix, "lib")
cxx_flags = ['-std=c++20', '-O3', '-fPIC', '-Wno-psabi']
cxx_flags += ['-g']
sources = ['csrc/python_api.cpp']
build_include_dirs = [
    f'{CUDA_HOME}/include',
    'MiraOperator/include',
    'third-party/cutlass/include',
    'third-party/fmt/include',
    conda_include,
    current_dir
]
build_libraries = ['cuda', 'cudart']
build_library_dirs = [
    f'{CUDA_HOME}/lib64',
    f'{CUDA_HOME}/lib64/stub',
    conda_lib, 
]
third_party_include_dirs = [
    'third-party/cutlass/include/cute',
    'third-party/cutlass/include/cutlass',
]


class CustomBuildPy(build_py):#继承build_py
    def run(self):
        # First, prepare the include directories
        self.prepare_includes()

        # Second, make clusters' cache setting default into `envs.py`
        self.generate_default_envs()

        # Finally, run the regular build
        build_py.run(self)

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        for name in ('MO_JIT_CACHE_DIR', 'MO_JIT_PRINT_COMPILER_COMMAND', 'MO_JIT_DISABLE_SHORTCUT_CACHE'):
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''
        print(name)
        with open(os.path.join(self.build_lib, 'MiraOperator', 'envs.py'), 'w') as f:
            f.write(code)

    def prepare_includes(self):
        # Create temporary build directory instead of modifying package directory
        build_include_dir = os.path.join(self.build_lib, 'MiraOperator/include')
        os.makedirs(build_include_dir, exist_ok=True)

        # Copy third-party includes to the build directory
        for d in third_party_include_dirs:
            dirname = d.split('/')[-1]
            src_dir = os.path.join(current_dir, d)
            dst_dir = os.path.join(build_include_dir, dirname)

            # Remove existing directory if it exists
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            # Copy the directory
            shutil.copytree(src_dir, dst_dir)


if __name__ == '__main__':
    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        revision = ''

    # noinspection PyTypeChecker
    setuptools.setup(
        name='MiraOperator',
        version='1.0.0' + revision,
        packages=find_packages('.'),
        package_data={
            'MiraOperator': [
                'include/MiraOperator/**/*',
                'include/cute/**/*',
                'include/cutlass/**/*',
            ]
        },
        ext_modules=[
            CppExtension(name='mira_operator_cpp',
                         sources=sources,
                         include_dirs=build_include_dirs,
                         libraries=build_libraries,
                         library_dirs=build_library_dirs,
                         extra_compile_args=cxx_flags)
        ],
        zip_safe=False,
        cmdclass={
            'build_py': CustomBuildPy,
        },
    )