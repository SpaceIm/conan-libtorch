from conans import ConanFile, CMake, tools
from conans.errors import ConanInvalidConfiguration
import os


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
        "with_cuda": [True, False],
        "with_rocm": [True, False],
        "with_cudnn": [True, False],
        "with_fbgemm": [True, False],
        "with_fakelowp": [True, False],
        "with_ffmpeg": [True, False],
        "with_gflags": [True, False],
        "with_leveldb": [True, False],
        "with_lmdb": [True, False],
        "with_metal": [True, False],
        "with_nnapi": [True, False],
        "with_nnpack": [True, False],
        "with_numa": [True, False],
        "with_nvrtc": [True, False],
        "observers": [True, False],
        "with_opencl": [True, False],
        "with_opencv": [True, False],
        "with_openmp": [True, False],
        "profiling": [True, False],
        "qnnpack": [True, False],
        "pytorch_qnnpack": [True, False],
        "with_redis": [True, False],
        "with_rocksdb": [True, False],
        "with_snpe": [True, False],
        "with_tensorrt": [True, False],
        "with_vulkan": [True, False],
        "vulkan_wrapper": [True, False],
        "vulkan_shaderc_runtime": [True, False],
        "vulkan_relaxed_precision": [True, False],
        "with_xnnpack": [True, False],
        "with_zmq": [True, False],
        "with_zstd": [True, False],
        "with_mkldnn": [True, False],
        "mkldnn_cblas": [True, False],
        "with_distributed": [True, False],
        "with_mpi": [True, False],
        "with_gloo": [True, False],
        "with_tensorpipe": [True, False],
        "with_tbb": [True, False],
        "onnx_ml_api": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "blas": "openblas", # should be mkl on non mobile os
        "with_cuda": False,
        "with_rocm": False,
        "with_cudnn": False,
        "with_fbgemm": False,
        "with_fakelowp": False,
        "with_ffmpeg": False,
        "with_gflags": False,
        "with_leveldb": False,
        "with_lmdb": False,
        "with_metal": False,
        "with_nnapi": False,
        "with_nnpack": False,
        "with_numa": False,
        "with_nvrtc": False,
        "observers": False,
        "with_opencl": False,
        "with_opencv": False,
        "with_openmp": False,
        "profiling": False,
        "qnnpack": False,
        "pytorch_qnnpack": False,
        "with_redis": False,
        "with_rocksdb": False,
        "with_snpe": False,
        "with_tensorrt": False,
        "with_vulkan": False,
        "vulkan_wrapper": False,
        "vulkan_shaderc_runtime": False,
        "vulkan_relaxed_precision": False,
        "with_xnnpack": False,
        "with_zmq": False,
        "with_zstd": False,
        "with_mkldnn": False,
        "mkldnn_cblas": False,
        "with_distributed": False,
        "with_mpi": False,
        "with_gloo": False,
        "with_tensorpipe": False,
        "with_tbb": False,
        "onnx_ml_api": False,
    }

    short_paths = True

    exports_sources = "CMakeLists.txt"
    generators = "cmake", "cmake_find_package", "cmake_find_package_multi"
    _cmake = None

    @property
    def _source_subfolder(self):
        return "source_subfolder"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC
        if self.settings.os in ["Android", "iOS"]:
            self.options.blas = "eigen"

    def configure(self):
        if self.options.shared:
            del self.options.fPIC
        del self.settings.compiler.cppstd
        del self.settings.compiler.libcxx

    def requirements(self):
        self.requires("eigen/3.3.9")
        self.requires("fmt/7.1.3")
        self.requires("protobuf/3.13.0")
        self.requires("pybind11/2.6.2")
        if self.options.blas == "openblas":
            self.requires("openblas/0.3.13")
        elif self.options.blas in ["atlas", "mkl", "veclib", "flame"]:
            raise ConanInvalidConfiguration("{} recipe not yet in CCI".format(self.options.blas))
        if self.options.with_gflags:
            self.requires("gflags/2.2.2")
        if self.options.with_opencv:
            self.requires("opencv/4.5.1")
        if self.options.with_redis:
            self.requires("hiredis/1.0.0")
        if self.options.with_rocksdb:
            self.requires("rocksdb/6.10.2")
        if self.options.with_zstd:
            self.requires("zstd/1.4.8")

    def source(self):
        tools.get(**self.conan_data["sources"][self.version])
        os.rename("pytorch-" + self.version, self._source_subfolder)

    def _patch_sources(self):
        # Inject external fmt
        tools.replace_in_file(os.path.join(self._source_subfolder, "caffe2", "CMakeLists.txt"),
                              "fmt::fmt-header-only", "CONAN_PKG::fmt")
        tools.replace_in_file(os.path.join(self._source_subfolder, "cmake", "Dependencies.cmake"),
                              "add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/fmt)", "")
        tools.replace_in_file(os.path.join(self._source_subfolder, "cmake", "Dependencies.cmake"),
                              "set_target_properties(fmt-header-only PROPERTIES INTERFACE_COMPILE_FEATURES \"\")", "")
        tools.replace_in_file(os.path.join(self._source_subfolder, "torch", "CMakeLists.txt"),
                              "fmt::fmt-header-only", "CONAN_PKG::fmt")

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
        self._cmake.definitions["BUILD_CAFFE2_OPS"] = False
        self._cmake.definitions["BUILD_CAFFE2_MOBILE"] = False
        self._cmake.definitions["CAFFE2_LINK_LOCAL_PROTOBUF"] = False
        self._cmake.definitions["CAFFE2_USE_MSVC_STATIC_RUNTIME"] = False
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
        self._cmake.definitions["USE_CUDNN"] = self.options.with_cuda and self.options.with_cudnn
        self._cmake.definitions["USE_STATIC_CUDNN"] = False
        self._cmake.definitions["USE_FBGEMM"] = self.options.with_fbgemm
        self._cmake.definitions["USE_FAKELOWP"] = self.options.with_fakelowp
        self._cmake.definitions["USE_FFMPEG"] = self.options.with_ffmpeg
        self._cmake.definitions["USE_GFLAGS"] = self.options.with_gflags
        self._cmake.definitions["USE_LEVELDB"] = self.options.with_leveldb
        self._cmake.definitions["USE_LITE_PROTO"] = False
        self._cmake.definitions["USE_LMDB"] = self.options.with_lmdb
        self._cmake.definitions["USE_METAL"] = self.options.get_safe("with_metal", False)
        self._cmake.definitions["USE_NATIVE_ARCH"] = False
        self._cmake.definitions["USE_NCCL"] = self.settings.os not in ["Windows", "Macos"] and (self.options.with_cuda or self.options.with_rocm)
        self._cmake.definitions["USE_STATIC_NCCL"] = False
        self._cmake.definitions["USE_SYSTEM_NCCL"] = False
        self._cmake.definitions["USE_NNAPI"] = self.options.with_nnapi
        self._cmake.definitions["USE_NNPACK"] = self.options.with_nnpack
        self._cmake.definitions["USE_NUMA"] = self.options.get_safe("with_numa", False)
        self._cmake.definitions["USE_NVRTC"] = self.options.with_nvrtc
        self._cmake.definitions["USE_NUMPY"] = False
        self._cmake.definitions["USE_OBSERVERS"] = self.options.observers
        self._cmake.definitions["USE_OPENCL"] = self.options.with_opencl
        self._cmake.definitions["USE_OPENCV"] = self.options.with_opencv
        self._cmake.definitions["USE_OPENMP"] = self.options.with_openmp
        self._cmake.definitions["USE_PROF"] = self.options.profiling
        self._cmake.definitions["USE_QNNPACK"] = self.options.qnnpack
        self._cmake.definitions["USE_PYTORCH_QNNPACK"] = self.options.pytorch_qnnpack
        self._cmake.definitions["USE_REDIS"] = self.options.with_redis
        self._cmake.definitions["USE_ROCKSDB"] = self.options.with_rocksdb
        self._cmake.definitions["USE_SNPE"] = self.options.with_snpe
        self._cmake.definitions["USE_SYSTEM_EIGEN_INSTALL"] = True
        self._cmake.definitions["USE_TENSORRT"] = self.options.with_tensorrt
        self._cmake.definitions["USE_VULKAN"] = self.options.with_vulkan
        self._cmake.definitions["USE_VULKAN_WRAPPER"] = self.options.vulkan_wrapper
        self._cmake.definitions["USE_VULKAN_SHADERC_RUNTIME"] = self.options.vulkan_shaderc_runtime
        self._cmake.definitions["USE_VULKAN_RELAXED_PRECISION"] = self.options.vulkan_relaxed_precision
        self._cmake.definitions["USE_XNNPACK"] = self.options.with_xnnpack
        self._cmake.definitions["USE_ZMQ"] = self.options.with_zmq
        self._cmake.definitions["USE_ZSTD"] = self.options.with_zstd
        self._cmake.definitions["USE_MKLDNN"] = self.options.get_safe("with_mkldnn", False)
        self._cmake.definitions["USE_MKLDNN_CBLAS"] = self.options.mkldnn_cblas
        self._cmake.definitions["USE_DISTRIBUTED"] = self.options.with_distributed
        self._cmake.definitions["USE_MPI"] = self.options.with_mpi
        self._cmake.definitions["USE_GLOO"] = self.options.with_gloo
        self._cmake.definitions["USE_TENSORPIPE"] = self.options.with_tensorpipe
        self._cmake.definitions["USE_TBB"] = self.options.with_tbb
        self._cmake.definitions["ONNX_ML"] = self.options.onnx_ml_api
        self._cmake.definitions["HAVE_SOVERSION"] = True

        # FIXME: unvendor some dependencies if possible
        self._cmake.definitions["USE_SYSTEM_LIBS"] = False
        self._cmake.definitions["USE_SYSTEM_CPUINFO"] = False
        self._cmake.definitions["USE_SYSTEM_SLEEF"] = False
        self._cmake.definitions["USE_SYSTEM_GLOO"] = False
        self._cmake.definitions["USE_SYSTEM_FP16"] = False
        self._cmake.definitions["USE_SYSTEM_PTHREADPOOL"] = False
        self._cmake.definitions["USE_SYSTEM_PSIMD"] = False
        self._cmake.definitions["USE_SYSTEM_FXDIV"] = False
        self._cmake.definitions["USE_SYSTEM_BENCHMARK"] = False
        self._cmake.definitions["USE_SYSTEM_ONNX"] = False
        self._cmake.definitions["USE_SYSTEM_XNNPACK"] = False

        self._cmake.definitions["BUILDING_WITH_TORCH_LIBS"] = True
        self._cmake.definitions["BLAS"] = self._blas_cmake_option_value
        self._cmake.configure()
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

    def build(self):
        self._patch_sources()
        cmake = self._configure_cmake()
        cmake.build()

    def package(self):
        self.copy("LICENSE", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
