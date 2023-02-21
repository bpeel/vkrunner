project('vkrunner', ['c', 'rust'],
        version : '0.1',
        default_options : 'default_library=static')

cc = meson.get_compiler('c')

foreach a : ['-Wall', '-Wuninitialized', '-Wempty-body', '-Wformat',
      '-Wformat-security', '-Winit-self', '-Wundef',
      '-Wvla', '-Wpointer-arith', '-Wmissing-declarations']
  if cc.has_argument(a)
    add_project_arguments(a, language : ['c'])
  endif
endforeach

m_dep = cc.find_library('m', required : false)

rust = import('rust', required : false)

subdir('vkrunner')
subdir('src')
