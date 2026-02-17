from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pathlib
import subprocess
import re

# Assumes script directory is root (it is)
ROOT_DIR = pathlib.Path(__file__).resolve().parent

def get_version():
    pyproject_toml_text = open(ROOT_DIR / "pyproject.toml", "r").read()
    pattern = "\[project\]\nname = \"trainweights\"\nversion = \"([0-9.]*)\""
    matches = re.search(pattern, pyproject_toml_text)
    return matches.group(1)

def get_requirements() -> list[str]:
    # TODO: Make requirements.txt better -- not just a lazy pip freeze
    with open(ROOT_DIR / "requirements.txt", "r") as f:
        return f.readlines()


def call_cmake_build():
    import cmake

    dir = cmake.CMAKE_BIN_DIR
    build_dir = ROOT_DIR / "build"

    subprocess.check_call(
        [f"{dir}/cmake", str(ROOT_DIR), "-B", str(build_dir) ],
    )

class CMakeBuildExt(build_ext):
    def run(self):
        call_cmake_build()

c_extension = Extension(name="trainweights._C", sources=[])
ext_modules = [c_extension]

setup(
    # static metadata should rather go in pyproject.toml
    version=get_version(),
    ext_modules=ext_modules,
    install_requires=get_requirements(),
    cmdclass={"build_ext": CMakeBuildExt},
)

