import glob
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, library_paths

# third_party_libs = ["ThreadPool"]
# third_party_includes = [os.path.realpath(os.path.join("third_party", lib)) for lib in third_party_libs]
third_party_includes = []

ext_libs = []
compile_args = []

transducer_related_cpp = ['src/transducer/pmerge.cpp']

extension = CppExtension(
    name='dhelper._ext.dhelper',
    package=True,
    sources=['src/binding.cpp'] + transducer_related_cpp,
    include_dirs=third_party_includes,
    libraries=ext_libs,
    extra_compile_args=compile_args,
    language='c++'
)

setup(
    name='dhelper',
    version = '0.1',
    description="DecodeHelper: A decode toolkit implemented by C++ for end-to-end speech recognition model.",
    url='https://github.com/ZhengkunTian/DecodeHelper',
    author='Zhengkun Tian',
    author_email='zhengkun.tian@outlook.com',
     packages=find_packages(exclude=["build"]),
    ext_modules=[extension],
    cmdclass={
        'build_ext': BuildExtension
})