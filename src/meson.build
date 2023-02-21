vkrunner_exe = executable('vkrunner',
                          sources : 'main.rs',
                          link_with : vkrunner_rust_lib,
                          install : true)
if rust.found()
    rust.test('exe-rust-unit-tests', vkrunner_exe)
endif

gtk_dep = dependency('gtk+-3.0', required : false)

if gtk_dep.found() and target_machine.system() == 'linux'
    gui_sources = [
        'gui.c',
        'gui-initial-script.c',
        'gui-worker.c',
    ] 

    executable('vkrunner-gui',
               sources : gui_sources,
               link_with : vkrunner_static_lib,
               install : true,
               include_directories : '..',
               dependencies : gtk_dep)
endif
