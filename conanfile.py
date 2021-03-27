from conans import ConanFile, CMake, tools
from conans.errors import ConanInvalidConfiguration
import glob
import os
import textwrap

required_conan_version = ">=1.33.0"


class LibtorchConan(ConanFile):
    name = "libtorch"
    description = "Tensors and Dynamic neural networks with strong GPU acceleration."
    license = "BSD-3-Clause"
    topics = ("conan", "libtorch", "pytorch", "machine-learning",
              "deep-learning", "neural-network", "gpu", "tensor")
    homepage = "https://pytorch.org"
    url = "https://github.com/conan-io/conan-center-index"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "blas": ["eigen", "atlas", "openblas", "mkl", "veclib", "flame", "generic"], # generic means "whatever blas lib found"
        "aten_parallel_backend": ["native", "openmp", "tbb"],
        "with_cuda": [True, False],
        "with_cudnn": [True, False],
        "with_nvrtc": [True, False],
        "with_tensorrt": [True, False],
        "with_rocm": [True, False],
        "with_nccl": [True, False],
        "with_fbgemm": [True, False],
        "fakelowp": [True, False],
        "with_ffmpeg": [True, False],
        "with_gflags": [True, False],
        "with_leveldb": [True, False],
        "with_lmdb": [True, False],
        "with_metal": [True, False],
        "with_nnapi": [True, False],
        "with_nnpack": [True, False],
        "with_numa": [True, False],
        "observers": [True, False],
        "with_opencl": [True, False],
        "with_opencv": [True, False],
        "profiling": [True, False],
        "with_qnnpack": [True, False],
        "with_redis": [True, False],
        "with_rocksdb": [True, False],
        "with_snpe": [True, False],
        "with_vulkan": [True, False],
        "vulkan_shaderc_runtime": [True, False],
        "vulkan_relaxed_precision": [True, False],
        "with_xnnpack": [True, False],
        "with_zmq": [True, False],
        "with_zstd": [True, False],
        "with_mkldnn": [True, False],
        "distributed": [True, False],
        "with_mpi": [True, False],
        "with_gloo": [True, False],
        "with_tensorpipe": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "blas": "openblas", # should be mkl on non mobile os
        "aten_parallel_backend": "native",
        "with_cuda": False,
        "with_cudnn": True,
        "with_nvrtc": False,
        "with_tensorrt": False,
        "with_rocm": False,
        "with_nccl": True,
        "with_fbgemm": False, # TODO: should be True
        "fakelowp": False,
        "with_ffmpeg": False,
        "with_gflags": False,
        "with_leveldb": False,
        "with_lmdb": False,
        "with_metal": True,
        "with_nnapi": False,
        "with_nnpack": False, # TODO: should be True
        "with_qnnpack": True,
        "with_xnnpack": True,
        "with_numa": True,
        "observers": False,
        "with_opencl": False,
        "with_opencv": False,
        "profiling": False,
        "with_redis": False,
        "with_rocksdb": False,
        "with_snpe": False,
        "with_vulkan": False,
        "vulkan_shaderc_runtime": False,
        "vulkan_relaxed_precision": False,
        "with_zmq": False,
        "with_zstd": False,
        "with_mkldnn": False,
        "distributed": True,
        "with_mpi": True,
        "with_gloo": False, # TODO: should be True
        "with_tensorpipe": True,
    }

    exports_sources = ["CMakeLists.txt", "patches/**"]
    generators = "cmake", "cmake_find_package", "cmake_find_package_multi"
    _cmake = None

    @property
    def _source_subfolder(self):
        return "source_subfolder"

    @property
    def _build_subfolder(self):
        return "build_subfolder"

    def config_options(self):
        # Change default options for several OS
        if self.settings.os in ["Android", "iOS"]:
            self.options.blas = "eigen"
        if self.settings.os not in ["Linux", "Windows"]:
            self.options.distributed = False

        # Remove several options not supported for several OS
        if self.settings.os == "Windows":
            del self.options.fPIC
            del self.options.with_nnpack
            del self.options.with_qnnpack
            del self.options.with_mpi
            del self.options.with_tensorpipe
        if self.settings.os != "iOS":
            del self.options.with_metal
        if self.settings.os != "Android":
            del self.options.with_nnapi
            del self.options.with_snpe
        if self.settings.os != "Linux":
            del self.options.with_numa

    def configure(self):
        if self.options.shared:
            del self.options.fPIC
        if not self.options.with_cuda:
            del self.options.with_cudnn
            del self.options.with_nvrtc
            del self.options.with_tensorrt
        if not (self.options.with_cuda or self.options.with_rocm):
            del self.options.with_nccl
        if not self.options.with_vulkan:
            del self.options.vulkan_shaderc_runtime
            del self.options.vulkan_relaxed_precision
        if not self.options.with_fbgemm:
            del self.options.fakelowp
        if not self.options.distributed:
            del self.options.with_mpi
            del self.options.with_gloo
            del self.options.with_tensorpipe

        if self.settings.compiler.get_safe("cppstd"):
            tools.check_min_cppstd(self, 14)
        if self.options.with_cuda and self.options.with_rocm:
            raise ConanInvalidConfiguration("libtorch doesn't yet support simultaneously building with CUDA and ROCm")
        if self.options.with_ffmpeg and not self.options.with_opencv:
            raise ConanInvalidConfiguration("libtorch video support with ffmpeg also requires opencv")
        if self.options.blas == "veclib" and not tools.is_apple_os(self.settings.os):
            raise ConanInvalidConfiguration("veclib only available on Apple family OS")

        if self.options.distributed and self.settings.os not in ["Linux", "Windows"]:
            self.output.warn("Distributed libtorch is not tested on {} and likely won't work".format(str(self.settings.os)))

    def requirements(self):
        self.requires("cpuinfo/cci.20201217")
        self.requires("eigen/3.3.9")
        self.requires("fmt/7.1.3")
        self.requires("foxi/cci.20210217")
        self.requires("onnx/1.8.1")
        self.requires("protobuf/3.15.5")
        self.requires("pybind11/2.6.2")
        if self._depends_on_sleef:
            self.requires("sleef/3.5.1")
        if self.options.blas == "openblas":
            self.requires("openblas/0.3.13")
        elif self.options.blas in ["atlas", "mkl", "flame"]:
            raise ConanInvalidConfiguration("{} recipe not yet available in CCI".format(self.options.blas))
        if self.options.aten_parallel_backend == "tbb":
            self.requires("tbb/2020.3")
        if self.options.with_cuda:
            self.output.warn("cuda recipe not yet available in CCI, assuming that NVIDIA CUDA SDK is installed on your system")
        if self.options.get_safe("with_cudnn"):
            self.output.warn("cudnn recipe not yet available in CCI, assuming that NVIDIA CuDNN is installed on your system")
        if self.options.get_safe("with_tensorrt"):
            self.output.warn("tensorrt recipe not yet available in CCI, assuming that NVIDIA TensorRT SDK is installed on your system")
        if self.options.with_rocm:
            raise ConanInvalidConfiguration("rocm recipe not yet available in CCI")
        if self.options.with_fbgemm:
            raise ConanInvalidConfiguration("fbgemm recipe not yet available in CCI")
            self.requires("fbgemm/cci.20210309")
        if self.options.with_ffmpeg:
            raise ConanInvalidConfiguration("ffmpeg recipe not yet available in CCI")
        if self.options.with_gflags:
            self.requires("gflags/2.2.2")
        if self.options.with_leveldb:
            self.requires("leveldb/1.22")
        if self.options.with_lmdb:
            # should be part of OpenLDAP or packaged separately?
            raise ConanInvalidConfiguration("lmdb recipe not yet available in CCI")
        if self.options.get_safe("with_nnpack"):
            raise ConanInvalidConfiguration("nnpack recipe not yet available in CCI")
        if self.options.get_safe("with_qnnpack"):
            self.requires("fp16/cci.20200514")
            self.requires("fxdiv/cci.20200417")
            self.requires("psimd/cci.20200517")
        if self.options.with_xnnpack:
            self.requires("xnnpack/cci.20210310")
        if self.options.get_safe("with_nnpack") or self.options.get_safe("with_qnnpack") or self.options.with_xnnpack:
            self.requires("pthreadpool/cci.20210218")
        if self.options.get_safe("with_numa"):
            self.requires("libnuma/2.0.14")
        if self.options.with_opencl:
            self.requires("opencl-headers/2020.06.16")
            self.requires("opencl-icd-loader/2020.06.16")
        if self.options.with_opencv:
            self.requires("opencv/4.5.1")
        if self.options.with_redis:
            self.requires("hiredis/1.0.0")
        if self.options.with_rocksdb:
            self.requires("rocksdb/6.10.2")
        if self.options.with_vulkan:
            self.requires("vulkan-headers/1.2.170.0")
            self.requires("vulkan-loader/1.2.170.0")
        if self.options.get_safe("vulkan_shaderc_runtime"):
            self.requires("shaderc/2019.0")
        if self.options.with_zmq:
            self.requires("zeromq/4.3.4")
        if self.options.with_zstd:
            self.requires("zstd/1.4.9")
        if self.options.with_mkldnn:
            raise ConanInvalidConfiguration("oneDNN (MKL-DNN) recipe not yet available in CCI")
        if self.settings.os == "Windows" and self.options.distributed:
            self.requires("libuv/1.41.0")
        if self.options.get_safe("with_mpi"):
            self.requires("openmpi/4.1.0")
        if self.options.get_safe("with_gloo"):
            raise ConanInvalidConfiguration("gloo recipe not yet available in CCI")
        if self.options.get_safe("with_tensorpipe"):
            self.requires("tensorpipe/cci.20210309")

    @property
    def _depends_on_sleef(self):
        return self.settings.compiler != "Visual Studio" and self.settings.os not in ["Android", "iOS"]

    def build_requirements(self):
        # FIXME: libtorch 1.8.0 requires at least python 3.6.2 to run several python scripts during build
        pass
        # self.build_requires("cpython/3.9.1")

    def source(self):
        tools.get(**self.conan_data["sources"][self.version])
        os.rename("pytorch-" + self.version, self._source_subfolder)

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)
        self._cmake.definitions["ATEN_NO_TEST"] = True
        self._cmake.definitions["BUILD_BINARY"] = True
        self._cmake.definitions["BUILD_DOCS"] = False
        self._cmake.definitions["BUILD_CUSTOM_PROTOBUF"] = False
        self._cmake.definitions["BUILD_PYTHON"] = False
        self._cmake.definitions["BUILD_CAFFE2"] = True
        self._cmake.definitions["BUILD_CAFFE2_OPS"] = True
        self._cmake.definitions["BUILD_CAFFE2_MOBILE"] = False
        self._cmake.definitions["CAFFE2_LINK_LOCAL_PROTOBUF"] = False
        self._cmake.definitions["CAFFE2_USE_MSVC_STATIC_RUNTIME"] = self.settings.compiler.get_safe("runtime") in ["MT", "MTd", "static"]
        self._cmake.definitions["BUILD_TEST"] = False
        self._cmake.definitions["BUILD_STATIC_RUNTIME_BENCHMARK"] = False
        self._cmake.definitions["BUILD_MOBILE_BENCHMARKS"] = False
        self._cmake.definitions["BUILD_MOBILE_TEST"] = False
        self._cmake.definitions["BUILD_JNI"] = False
        self._cmake.definitions["BUILD_MOBILE_AUTOGRAD"] = False
        self._cmake.definitions["INSTALL_TEST"] = False
        self._cmake.definitions["USE_CPP_CODE_COVERAGE"] = False
        self._cmake.definitions["COLORIZE_OUTPUT"] = True
        self._cmake.definitions["USE_ASAN"] = False
        self._cmake.definitions["USE_TSAN"] = False
        self._cmake.definitions["USE_CUDA"] = self.options.with_cuda
        self._cmake.definitions["USE_ROCM"] = self.options.with_rocm
        self._cmake.definitions["CAFFE2_STATIC_LINK_CUDA"] = False
        self._cmake.definitions["USE_CUDNN"] = self.options.get_safe("with_cudnn", False)
        self._cmake.definitions["USE_STATIC_CUDNN"] = False
        self._cmake.definitions["USE_FBGEMM"] = self.options.with_fbgemm
        self._cmake.definitions["USE_FAKELOWP"] = self.options.get_safe("fakelowp", False)
        self._cmake.definitions["USE_FFMPEG"] = self.options.with_ffmpeg
        self._cmake.definitions["USE_GFLAGS"] = self.options.with_gflags
        self._cmake.definitions["USE_LEVELDB"] = self.options.with_leveldb
        self._cmake.definitions["USE_LITE_PROTO"] = False
        self._cmake.definitions["USE_LMDB"] = self.options.with_lmdb
        self._cmake.definitions["USE_METAL"] = self.options.get_safe("with_metal", False)
        self._cmake.definitions["USE_NATIVE_ARCH"] = False
        self._cmake.definitions["USE_NCCL"] = self.options.get_safe("with_nccl", False)
        self._cmake.definitions["USE_STATIC_NCCL"] = False
        self._cmake.definitions["USE_SYSTEM_NCCL"] = False # technically we could create a recipe for nccl with 0 packages (because it requires CUDA at build time)
        self._cmake.definitions["USE_NNAPI"] = self.options.get_safe("with_nnapi", False)
        self._cmake.definitions["USE_NNPACK"] = self.options.get_safe("with_nnpack", False)
        self._cmake.definitions["USE_NUMA"] = self.options.get_safe("with_numa", False)
        self._cmake.definitions["USE_NVRTC"] = self.options.get_safe("with_nvrtc", False)
        self._cmake.definitions["USE_NUMPY"] = False
        self._cmake.definitions["USE_OBSERVERS"] = self.options.observers
        self._cmake.definitions["USE_OPENCL"] = self.options.with_opencl
        self._cmake.definitions["USE_OPENCV"] = self.options.with_opencv
        self._cmake.definitions["USE_OPENMP"] = self.options.aten_parallel_backend == "openmp"
        self._cmake.definitions["USE_PROF"] = self.options.profiling
        self._cmake.definitions["USE_QNNPACK"] = False                                                # QNNPACK is now integrated into libtorch and official repo
        self._cmake.definitions["USE_PYTORCH_QNNPACK"] = self.options.get_safe("with_qnnpack", False) # is archived, so prefer to use vendored QNNPACK
        self._cmake.definitions["USE_REDIS"] = self.options.with_redis
        self._cmake.definitions["USE_ROCKSDB"] = self.options.with_rocksdb
        self._cmake.definitions["USE_SNPE"] = self.options.get_safe("with_snpe", False)
        self._cmake.definitions["USE_SYSTEM_EIGEN_INSTALL"] = True
        self._cmake.definitions["USE_TENSORRT"] = self.options.get_safe("with_tensorrt", False)
        self._cmake.definitions["USE_VULKAN"] = self.options.with_vulkan
        self._cmake.definitions["USE_VULKAN_WRAPPER"] = False
        self._cmake.definitions["USE_VULKAN_SHADERC_RUNTIME"] = self.options.get_safe("vulkan_shaderc_runtime", False)
        self._cmake.definitions["USE_VULKAN_RELAXED_PRECISION"] = self.options.get_safe("vulkan_relaxed_precision", False)
        self._cmake.definitions["USE_XNNPACK"] = self.options.with_xnnpack
        self._cmake.definitions["USE_ZMQ"] = self.options.with_zmq
        self._cmake.definitions["USE_ZSTD"] = self.options.with_zstd
        self._cmake.definitions["USE_MKLDNN"] = self.options.with_mkldnn
        self._cmake.definitions["USE_MKLDNN_CBLAS"] = False # This option has no logic and is useless in libtorch actually
        self._cmake.definitions["USE_DISTRIBUTED"] = self.options.distributed
        self._cmake.definitions["USE_MPI"] = self.options.get_safe("with_mpi", False)
        self._cmake.definitions["USE_GLOO"] = self.options.get_safe("with_gloo", False)
        self._cmake.definitions["USE_TENSORPIPE"] = self.options.get_safe("with_tensorpipe", False)
        self._cmake.definitions["USE_TBB"] = self.options.aten_parallel_backend == "tbb"
        self._cmake.definitions["ONNX_ML"] = True
        self._cmake.definitions["HAVE_SOVERSION"] = True
        self._cmake.definitions["USE_SYSTEM_LIBS"] = True

        self._cmake.definitions["BUILDING_WITH_TORCH_LIBS"] = True
        self._cmake.definitions["BLAS"] = self._blas_cmake_option_value

        self._cmake.definitions["MSVC_Z7_OVERRIDE"] = False

        # Custom variables for our CMake wrapper
        self._cmake.definitions["CONAN_LIBTORCH_USE_PTHREADPOOL"] = self._use_nnpack_family
        self._cmake.definitions["CONAN_LIBTORCH_USE_CPUINFO"] = True
        self._cmake.definitions["CONAN_LIBTORCH_USE_FXDIV"] = self.options.with_xnnpack
        self._cmake.definitions["CONAN_LIBTORCH_USE_PSIMD"] = self.options.with_xnnpack
        self._cmake.definitions["CONAN_LIBTORCH_USE_FP16"] = self.options.with_xnnpack
        self._cmake.definitions["CONAN_LIBTORCH_USE_SLEEF"] = self._depends_on_sleef

        self._cmake.configure(build_folder=self._build_subfolder)
        return self._cmake

    @property
    def _blas_cmake_option_value(self):
        return {
            "eigen": "Eigen",
            "atlas": "ATLAS",
            "openblas": "OpenBLAS",
            "mkl": "MKL",
            "veclib": "vecLib",
            "flame": "FLAME",
            "generic": "Generic"
        }[str(self.options.blas)]

    @property
    def _use_nnpack_family(self):
        return self.options.get_safe("with_nnpack") or self.options.get_safe("with_qnnpack") or self.options.with_xnnpack

    def build(self):
        if self.options.get_safe("with_snpe"):
            self.output.warn("with_snpe is enabled. Pay attention that you should have properly set SNPE_LOCATION and SNPE_HEADERS CMake variables")
        for patch in self.conan_data.get("patches", {}).get(self.version, []):
            tools.patch(**patch)
        # conflict with macros.h generated at build time
        os.remove(os.path.join(self.build_folder, self._source_subfolder, "caffe2", "core", "macros.h"))
        cmake = self._configure_cmake()
        cmake.build()

    def package(self):
        self.copy("LICENSE", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install() # FIXME: something is wrong with includes layout
        # TODO: Keep share/Aten/Declarations.yml?
        tools.rmdir(os.path.join(self.package_folder, "share"))
        self._create_cmake_module_variables(
            os.path.join(self.package_folder, self._module_subfolder, self._module_file)
        )

    @staticmethod
    def _create_cmake_module_variables(module_file):
        content = textwrap.dedent("""\
            if(DEFINED Torch_FOUND)
                set(TORCH_FOUND ${Torch_FOUND})
            endif()
            if(NOT DEFINED TORCH_INCLUDE_DIRS)
                get_target_property(TORCH_INCLUDE_DIRS Torch::Torch INTERFACE_INCLUDE_DIRECTORIES)
            endif()
            if(NOT DEFINED TORCH_LIBRARIES)
                set(TORCH_LIBRARIES "Torch::Torch")
            endif()
        """)
        tools.save(module_file, content)

    @property
    def _module_subfolder(self):
        return os.path.join("lib", "cmake")

    @property
    def _module_file(self):
        return "conan-official-{}-variables.cmake".format(self.name)

    def package_info(self):
        self.cpp_info.names["cmake_find_package"] = "Torch"
        self.cpp_info.names["cmake_find_package_multi"] = "Torch"
        self.cpp_info.builddirs.append(self._module_subfolder)
        self.cpp_info.build_modules["cmake_find_package"] = [os.path.join(self._module_subfolder, self._module_file)]
        self.cpp_info.build_modules["cmake_find_package_multi"] = [os.path.join(self._module_subfolder, self._module_file)]
        self.cpp_info.includedirs.append(os.path.join("include", "torch", "csrc", "api", "include"))

        # FIXME: - properly define order of libs:
        #          - torch_cpu depends on c10, Caffe2_perfkernels_avx, Caffe2_perfkernels_avx2, Caffe2_perfkernels_avx512, pytorch_qnnpack
        #          - torch depends on torch_cpu & torch_cuda & torch_hip
        #          - Caffe2_perfkernels_avx depends on c10
        #          - Caffe2_perfkernels_avx2 depends on c10
        #          - Caffe2_perfkernels_avx512 depends on c10
        #          - caffe2_detectron_ops_[gpu|hip] is a module library
        #          - c10 has no internal dependency?
        #          - torch_cuda depends on c10_cuda
        #          - c10_cuda depends on c10
        #          - torch_hip depends on c10_hip
        #          - c10_hip depends on c10
        #          - caffe2_observers depends on torch
        #        - if static:
        #          torch, caffe2_observers, torch_cpu, caffe2_protos, torch_cuda, torch_hip, Caffe2_perfkernels_avx,
        #          Caffe2_perfkernels_avx2, Caffe2_perfkernels_avx512 libs should be linked with whole archive
        #          - if clang: -Wl,-force_load,<lib>
        #          - if msvc : -WHOLEARCHIVE:<lib>
        #          - if gcc  : -Wl,--whole-archive,<lib> -Wl,--no-whole-archive

        if self.options.observers:
            self.cpp_info.libs.append("caffe2_observers")

        self.cpp_info.libs.append("torch")

        if self.options.with_cuda or self.options.with_rocm:
            self.cpp_info.libs.append("caffe2_nvrtc")

        if self.options.with_cuda:
            self.cpp_info.libs.extend(["torch_cuda", "c10_cuda"])
        elif self.options.with_rocm:
            self.cpp_info.libs.extend(["torch_hip", "c10_hip"])

        self.cpp_info.libs.append("torch_cpu")

        if self.options.get_safe("with_qnnpack"):
            self.cpp_info.libs.append("pytorch_qnnpack")

        def _add_lib_if_exists(name):
            if glob.glob(os.path.join(self.package_folder, "lib", "*{}.*".format(name))):
                self.cpp_info.libs.append(name)

        if not self.options.shared:
            # These libs are always static
            _add_lib_if_exists("Caffe2_perfkernels_avx")
            _add_lib_if_exists("Caffe2_perfkernels_avx2")
            _add_lib_if_exists("Caffe2_perfkernels_avx512")
            self.cpp_info.libs.append("caffe2_protos")

        self.cpp_info.libs.append("c10")

        # FIXME: system libs
        if self.options.blas == "veclib":
            self.cpp_info.frameworks.append("Accelerate")

        bin_path = os.path.join(self.package_folder, "bin")
        self.output.info("Appending PATH environment variable: {}".format(bin_path))
        self.env_info.PATH.append(bin_path)
