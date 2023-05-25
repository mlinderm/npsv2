import os, platform, re, shutil, subprocess, sys
from setuptools import Command, Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from distutils.spawn import find_executable

# Project structure and CMake build steps adapted from
# https://www.benjack.io/2018/02/02/python-cpp-revisited.html

cmake = find_executable(os.environ.get("CMAKE", "cmake"))

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output([cmake, "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            [cmake, ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            [cmake, "--build", "."] + build_args, cwd=self.build_temp
        )
        print()  # Add an empty line for cleaner output


def file_find_or_append(original_path, match_pattern, append, sep=" "):
    # Adapted from SeqLib python package.
    with open(original_path) as original_file:
        original = original_file.readlines()
    with open(original_path, "w") as replaced_file:
        for line in original:
            match = re.match(match_pattern, line)
            if match:
                print(line.rstrip(), *[flag for flag in append if flag not in match[0]], sep=sep, file=replaced_file)
            else:
                replaced_file.write(line)

def file_replace(original_path, match_pattern, replace_pattern):
    # Adapted from SeqLib python package.
    with open(original_path, "r") as original_file:
        original = original_file.readlines()
    with open(original_path, "w") as replaced_file:
        for line in original:
            replaced_file.write(re.sub(match_pattern, replace_pattern, line))


class SeqLibCMakeBuild(CMakeBuild):
    def run(self):
        # To link into a shared library we need to add the -fPIC and other flags to SeqLib dependencies
        # before building. Adapted from SeqLib python package.
        bwa_makefile_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "lib" , "seqlib", "bwa", "Makefile"
        )
        file_find_or_append(bwa_makefile_path, r"^CFLAGS\s*=.*$", ["-fPIC","-Wno-unused-result"])

        fermi_makefile_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "lib", "seqlib", "fermi-lite", "Makefile"
        )
        file_find_or_append(fermi_makefile_path, r"^CFLAGS\s*=.*$", ["-Wno-unused-result", "-Wno-unused-but-set-variable"])

        for lib in ("bwa", "fermi-lite"):
            file_replace(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "lib" , "seqlib", lib, "rle.h"
            ), r"^(const uint8_t rle_auxtab\[8\];)", r"extern \1")

        super().run()


# Find the Protocol Buffer compiler (protoc)
protoc = find_executable(os.environ.get("PROTOC", "protoc"))

def generate_proto(source: str):
    """Invoke Protocol Buffer compiler if .proto file is newer than generated code

    Args:
        source (str): Path to .proto file
    """
    # Adapted from Chromium
    output = source.replace(".proto", "_pb2.py")
    if not os.path.exists(output) or (
        os.path.exists(source) and os.path.getmtime(source) > os.path.getmtime(output)
    ):
        if not os.path.exists(source):
            sys.stderr.write("Can't find required file: %s\n" % source)
            sys.exit(-1)
        if protoc == None:
            sys.stderr.write("protoc is not installed .\n")
            sys.exit(-1)
        protoc_command = [protoc, "-I.", "--python_out=.", source]
        if subprocess.call(protoc_command) != 0:
            sys.exit(-1)


class BuildProtobuf(Command):
    description = "Compile protobuf descriptions"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        generate_proto("src/npsv2/npsv2.proto")


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="npsv2",
    version="0.1.0",
    description="Non-parametric genotyper for structural variants",
    long_description=readme,
    author="Michael Linderman",
    author_email="mlinderman@middlebury.edu",
    license=license,
    url="https://github.com/mlinderm/npsv2",
    scripts=["scripts/synthBAM"],
    entry_points="""
        [console_scripts]
        npsv2=npsv2.main:main
        npsv2u=npsv2.utilities.main:main
    """,
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"": ["etc/*", "conf/*.yaml", "conf/**/*.yaml"]},
    ext_modules=[CMakeExtension("npsv2/npsv2r")],
    cmdclass=dict(build_ext=SeqLibCMakeBuild, protobuf=BuildProtobuf),
    zip_safe=False,
    test_suite="tests",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
    ],
)
