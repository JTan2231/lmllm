project('my_project', 'cpp',
  default_options : ['cpp_std=c++23'])

executable('t', 'src/kernels.cpp', 'src/types.cpp', 'main.cpp',
  cpp_args : ['-std=c++23', '-Wall', '-Wextra', '-Werror', '-I./include', '-DLOG_TO_FILE'])
