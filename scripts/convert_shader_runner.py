#!/usr/bin/python3

# Copyright © 2023 Neil Roberts
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re
import sys
import os
from collections import namedtuple

SECTION_RE = re.compile(r'^\[([^\]]+)\]\s*$')
VERTEX_RE = re.compile(r'\bgl_Vertex\b')
SHADER_SECTION_RE = re.compile(r'(.*) shader$')
COMMENT_RE = re.compile(r'^\s*(?://|#|/\*| \*|\*/)(.*)')
TYPE_PREFIX = r'(|d|f16|i16|i64|i|i8|u16|u8|u|u64)'
VEC_RE = re.compile(TYPE_PREFIX + r'vec(\d+)$')
RECTANGLE_MAT_RE = re.compile(TYPE_PREFIX + r'mat(\d+)x(\d+)$')
SQUARE_MAT_RE = re.compile(TYPE_PREFIX + r'mat(\d+)$')
MVP_RE = re.compile(r'\bgl_ModelViewProjectionMatrix\b')
FRAG_COLOR_RE = re.compile(r'\bgl_FragColor\b')
FRONT_COLOR_RE = re.compile(r'\bgl_FrontColor\b')
COLOR_RE = re.compile(r'\bgl_Color\b')
SET_UNIFORM_RE = re.compile(r'uniform\s+(\S+)\s+(\S+)\s+(.*)$')
PROBE_PIXEL_RE = re.compile(r'(relative\s+)?probe\s+rgba?'
                            r'\s+([0-9\.]+)\s+([0-9\.]+)'
                            r'\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)'
                            r'(?:\s+([0-9\.]+))?\s*$')
PROBE_PIXEL_BRACKETS_RE = re.compile(r'(relative\s+)?probe\s+rgba?'
                                     r'\s+\(([0-9\.]+),\s*([0-9\.]+)\s*\)\s*'
                                     r'\(([0-9\.]+),\s*([0-9\.]+),\s*'
                                     r'([0-9\.]+)'
                                     r'(?:,\s*([0-9\.]+))?\s*\)\s*$')
PROBE_RECT_RE = re.compile(r'(relative\s+)?probe\s+rect\s+rgba?'
                           r'\s+\(([0-9\.]+),\s*([0-9\.]+),\s*'
                           r'([0-9\.]+),\s*([0-9\.]+)\s*\)\s*'
                           r'\(([0-9\.]+),\s*([0-9\.]+),\s*'
                           r'([0-9\.]+)'
                           r'(?:,\s*([0-9\.]+))?\s*\)\s*$')
PROBE_ALL_RE = re.compile(r'probe\s+all\s+rgba?'
                          r'\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)'
                          r'(?:\s+([0-9\.]+))?\s*$')
DRAW_RECT_RE = re.compile(r'draw rect( ortho| patch)*\s+[-0-9\.]')
ORTHO_RE = re.compile(r'ortho(?:\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)'
                      r'\s+([0-9\.]+))?\s*')
SIZE_RE = re.compile(r'SIZE\s+(\d+)\s+(\d+)\s*$')
VERSION_RE = re.compile(r'^[ \t]*#version\s+(\d+)[ \t]*\n',
                        re.MULTILINE)
ATTRIBUTE_HEADER_RE = re.compile(r'^([^\s/]+)/([^\s/]+)/([^\s/]+)'
                                 r'(?:/(\d+))?$')
FORMAT_COMPONENTS = "RGBA"

MIN_VERSION = 420

FB_WIDTH = 250
FB_HEIGHT = 250

# The tuple is the base type ('S' for signed integer, 'U' for unsigned
# integer and 'F' for floating-point) combined with the number of
# bytes.
TYPE_PREFIX_DATA = {
    'f16': ('F', 2),
    '': ('F', 4),
    'd': ('F', 8),
    'i8': ('S', 1),
    'i16': ('S', 2),
    'i': ('S', 4),
    'i64': ('S', 8),
    'u8': ('U', 1),
    'u16': ('U', 2),
    'u': ('U', 4),
    'u64': ('U', 8),
    'b': ('U', 4),
}

SIMPLE_TYPE_DATA = {
    'float16_t': ('F', 2),
    'float': ('F', 4),
    'double': ('F', 8),
    'int8_t': ('S', 1),
    'int16_t': ('S', 2),
    'int': ('S', 4),
    'int64_t': ('S', 8),
    'uint8_t': ('U', 1),
    'uint16_t': ('U', 2),
    'uint': ('U', 4),
    'uint64_t': ('U', 8),
    'bool': ('U', 4),
}


TypeData = namedtuple('TypeData', ['base', 'n_bytes', 'rows', 'columns'])


class ScriptError(Exception):
    pass


class Variable:
    def __init__(self, type_name, variable_name, array_size = None):
        self.type_name = type_name
        self.variable_name = variable_name
        self.array_size = array_size

    def alignment(self):
        alignment = type_alignment(self.type_name)

        if self.array_size is not None:
            return align_std140(alignment)
        else:
            return alignment

    def size(self):
        size = type_size(self.type_name)

        if self.array_size is not None:
            return self.array_size * align_std140(size)
        else:
            return size


class Shader:
    def __init__(self, stage, source):
        self.version = MIN_VERSION
        self.stage = stage
        self.source = source
        self.inputs = []
        self.outputs = []
        self.uniforms = []
        self.line_num = 1

    def _output_varyings(self, variables, direction, output):
        if len(variables) == 0:
            return

        for var in variables:
            print("layout(location = {}) {} {} {}".format(
                var.location,
                direction,
                var.type_name,
                var.variable_name),
                  file=output,
                  end='')

            if var.array_size is not None:
                print("[{}]".format(var.array_size), end='', file=output)

            print(";", file=output)

        print(file=output)

    def output(self, output):
        print("[{} shader]".format(self.stage), file=output)

        print("#version {}\n".format(self.version), file=output)

        self._output_varyings(self.inputs, "in", output)
        self._output_varyings(self.outputs, "out", output)

        if len(self.uniforms) > 0:
            print("layout(binding = 3) uniform block {", file=output)

            for uniform in self.uniforms:
                print("    {} {}".format(uniform.type_name,
                                       uniform.variable_name),
                      file=output,
                      end='')

                if uniform.array_size is not None:
                    print("[{}]".format(uniform.array_size),
                          file=output,
                          end='')

                print(";", file=output)

            print("};\n", file=output)

        print(self.source, file=output)


class ScriptProcessor:
    def __init__(self):
        self.header_comments = []
        self.requires = []
        self.shaders = []
        self.vertex_data = []
        self.had_vertex_data_header = False
        self.tests = []
        self.in_stage = None
        self.in_section = None
        self.buffer = []
        self.line_num = 1
        self.ran_filters = False
        self.passthrough_vertex_shader = False

    def run_filters(self):
        if self.ran_filters:
            return

        for shader in processor.shaders:
            for filter in SHADER_FILTERS:
                filter(shader)

        for filter in PROGRAM_FILTERS:
            filter(processor)

        self.ran_filters = True

    def end_section(self):
        if self.in_section is None:
            return

        if self.in_stage is not None:
            self.shaders.append(Shader(self.in_stage, "\n".join(self.buffer)))

        self.buffer.clear()
        self.in_stage = None
        self.in_section = None

    def get_uniform(self, name):
        for var in self.shaders[0].uniforms:
            if var.variable_name == name:
                return var

        raise ScriptError("missing var: {}".format(name))

    def _get_shader_stage(self, stage):
        for shader in self.shaders:
            if shader.stage == stage:
                return shader

        raise ScriptError("missing {} shader".format(stage))

    def _get_vertex_shader(self):
        return self._get_shader_stage("vertex")

    def _check_can_draw(self):
        if not self.passthrough_vertex_shader:
            self._get_shader_stage("vertex")
        self._get_shader_stage("fragment")

    def _add_vertex_data_line(self, line):
        self.run_filters()

        stripped = line.strip()

        if (len(stripped) > 0 and
            not stripped.startswith('#') and
            not self.had_vertex_data_header):
            vertex_shader = processor._get_vertex_shader()

            headers = map(lambda header:
                          convert_vertex_data_header(vertex_shader, header),
                          stripped.split())

            self.vertex_data.append(" ".join(headers))

            self.had_vertex_data_header = True
        else:
            self.vertex_data.append(line)

    def _add_test_line(self, line):
        self.run_filters()

        stripped = line.strip()

        if len(stripped) == 0 or stripped.startswith('#'):
            self.tests.append(line)
            return

        if stripped.startswith("clear"):
            self.tests.append(line)
            return

        md = ORTHO_RE.match(stripped)
        if md:
            if md.group(1):
                mat = ortho_matrix(float(md.group(1)),
                                   float(md.group(2)),
                                   float(md.group(3)),
                                   float(md.group(4)))
            else:
                mat = ortho_matrix(0, FB_WIDTH, 0, FB_HEIGHT)
            self.tests.append("ubo 3 subdata mat4 {}  {}"
                              .format(self.get_uniform("piglit_mvp").offset,
                                      " ".join(map(str, mat))))
            return

        md = SET_UNIFORM_RE.match(stripped)
        if md:
            self.tests.append("ubo 3 subdata {} {}  {}"
                              .format(md.group(1),
                                      self.get_uniform(md.group(2)).offset,
                                      md.group(3)))
            return

        md = PROBE_PIXEL_RE.match(stripped)
        if md is None:
            md = PROBE_PIXEL_BRACKETS_RE.match(stripped)
        if md:
            parts = []

            if md.group(1):
                parts.append(md.group(1))

            if md.group(7) is None:
                components = "rgb"
            else:
                components = "rgba"

            parts.extend(["probe ", components,
                          " (", md.group(2), ", ", md.group(3), ") (",
                          md.group(4), ", ", md.group(5), ", ", md.group(6)])

            if md.group(7):
                parts.extend([", ", md.group(7)])

            parts.append(")")

            self.tests.append("".join(parts))
            return

        md = PROBE_RECT_RE.match(stripped)
        if md:
            parts = []

            if md.group(1):
                parts.append(md.group(1))

            if md.group(9) is None:
                components = "rgb"
            else:
                components = "rgba"

            parts.extend(["probe rect ", components,
                          " (", md.group(2), ", ", md.group(3), ", ",
                          md.group(4), ", ", md.group(5), ") (",
                          md.group(6), ", ", md.group(7), ", ", md.group(8)])

            if md.group(9):
                parts.extend([", ", md.group(9)])

            parts.append(")")

            self.tests.append("".join(parts))
            return

        md = PROBE_ALL_RE.match(stripped)
        if md:
            if md.group(4) is None:
                components = "rgb"
            else:
                components = "rgba"

            parts = ["probe", "all", components,
                     md.group(1), md.group(2), md.group(3)]

            if md.group(4):
                parts.append(md.group(4))

            self.tests.append(" ".join(parts))
            return

        if DRAW_RECT_RE.match(stripped):
            self._check_can_draw()
            self.tests.append(line)
            return

        if stripped.startswith("draw arrays"):
            self._check_can_draw()
            self.tests.append(line)
            return

        raise ScriptError("line {}: unsupported test command: {}"
                          .format(self.line_num, line))

    def _add_require_line(self, line):
        stripped = line.strip()

        if stripped.startswith('#'):
            self.requires.append(line)
            return

        if len(stripped) == 0:
            if len(self.requires) > 0:
                self.requires.append(line)
            return

        if stripped.startswith("GL >="):
            return

        if stripped.startswith("GLSL >="):
            return

        if stripped in ["GL_ARB_compute_shader",
                        "GL_ARB_draw_instanced"]:
            return

        if stripped == "GL_ARB_gpu_shader_fp64":
            self.requires.append("shaderFloat64")
            return

        if stripped == "GL_AMD_shader_trinary_minmax":
            self.requires.append("VK_AMD_shader_trinary_minmax")
            return

        md = SIZE_RE.match(stripped)
        if md:
            self.requires.append("fbsize {} {}".format(md.group(1),
                                                       md.group(2)))
            return

        raise ScriptError("line {}: unsupported requirement: {}".format(
            self.line_num,
            stripped))

    def add_line(self, line):
        md = SECTION_RE.match(line)

        if md:
            self.end_section()

            section = md.group(1)
            md = SHADER_SECTION_RE.match(section)

            if md:
                self.in_stage = md.group(1)
                self.in_section = section
            elif section == 'vertex shader passthrough':
                self.passthrough_vertex_shader = True
                self.in_section = section
            elif section in ['require', 'vertex data', 'test']:
                self.in_section = section
            else:
                raise ScriptError("unknown section “{}”".format(section))
        elif self.in_section is None:
            if len(line.strip()) == 0:
                if len(self.header_comments) > 0:
                    self.header_comments.append(line)
            else:
                md = COMMENT_RE.match(line)
                if md is None:
                    raise ScriptError(("line {}: unexpected line before first "
                                       "section")
                                      .format(self.line_num))
                else:
                    self.header_comments.append("# " + md.group(1))
        elif self.in_stage:
            self.buffer.append(line)
        elif self.in_section == 'require':
            self._add_require_line(line)
        elif self.in_section == 'vertex data':
            self._add_vertex_data_line(line)
        elif self.in_section == 'test':
            self._add_test_line(line)

        self.line_num += 1

    def output(self, output):
        if len(self.header_comments) > 0:
            print("\n".join(self.header_comments), file=output)

        if len(self.requires) > 0:
            print("[require]", file=output)
            for require in self.requires:
                print(require, file=output)

        if self.passthrough_vertex_shader:
            print("[vertex shader passthrough]\n", file=output)

        for shader in self.shaders:
            shader.output(output)

        if len(self.vertex_data) > 0:
            print("[vertex data]", file=output)
            print("\n".join(self.vertex_data), file=output)

        if len(self.tests) > 0:
            print("[test]", file=output)
            print("\n".join(self.tests), file=output)


def align(pos, alignment):
    return (pos + alignment - 1) & ~(alignment - 1)


def align_std140(pos):
    # In std140 the positions are aligned to the size of a vec4
    return align(pos, 16)


def get_type_data(type_name):
    md = VEC_RE.match(type_name)
    if md:
        base, n_bytes = TYPE_PREFIX_DATA[md.group(1)]
        rows = int(md.group(2))
        return TypeData(base, n_bytes, rows, 1)

    md = RECTANGLE_MAT_RE.match(type_name)
    if md:
        base, n_bytes = TYPE_PREFIX_DATA[md.group(1)]
        columns = int(md.group(2))
        rows = int(md.group(3))
        return TypeData(base, n_bytes, rows, columns)

    md = SQUARE_MAT_RE.match(type_name)
    if md:
        base, n_bytes = TYPE_PREFIX_DATA[md.group(1)]
        size = int(md.group(2))
        return TypeData(base, n_bytes, size, size)

    try:
        base, n_bytes = SIMPLE_TYPE_DATA[type_name]
    except KeyError:
        raise ScriptError("unknown type: {}".format(type_name))

    return TypeData(base, n_bytes, 1, 1)


def type_alignment(type_name):
    type_data = get_type_data(type_name)

    multiplier = type_data.rows
    if multiplier == 3:
        multiplier = 4

    alignment = type_data.n_bytes * multiplier

    if type_data.columns > 1:
        alignment = align_std140(alignment)

    return alignment


def type_size(type_name):
    type_data = get_type_data(type_name)

    size = type_data.n_bytes * type_data.rows

    if type_data.columns > 1:
        size = align_std140(size) * type_data.columns

    return size


def ortho_matrix(left, right, bottom, top):
    return [2.0 / (right - left), 0.0, 0.0, 0.0,
            0.0, 2.0 / (top - bottom), 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -(right + left) / (right - left),
            -(top + bottom) / (top - bottom),
            0.0, 1.0]


def convert_vertex_data_header(vertex_shader, header):
    md = ATTRIBUTE_HEADER_RE.match(header)

    if md is None:
        raise ScriptError("Unsupported vertex data header: {}".format(header))

    for var in vertex_shader.inputs:
        if var.variable_name == md.group(1):
            location = var.location
            break
    else:
        raise ScriptError("missing input: {}".format(md.group(1)))

    data_type = get_type_data(md.group(2))

    try:
        count = int(md.group(3))
    except ValueError:
        count = get_type_data(md.group(3)).rows

    if md.group(4) is not None:
        location += int(md.group(4))

    vk_format = list(map(lambda c: "{}{}".format(c, data_type.n_bytes * 8),
                         FORMAT_COMPONENTS[:count]))

    if data_type.base == 'F':
        vk_format.append("_SFLOAT")
    elif data_type.base == 'U':
        vk_format.append("_UINT")
    elif data_type.base == 'S':
        vk_format.append("_SINT")

    return "{}/{}".format(location, "".join(vk_format))

def add_version(shader):
    def apply_version(md):
        version = int(md.group(1))

        if version >= shader.version:
            shader.version = version

        return ""

    shader.source = VERSION_RE.sub(apply_version, shader.source)


def add_gl_vertex_inputs(shader):
    if VERTEX_RE.search(shader.source):
        shader.inputs.append(Variable("vec4", "piglit_vertex"))
        shader.source = VERTEX_RE.sub("piglit_vertex", shader.source)


def replace_variables(keywords, source, callback):
    regexp = re.compile(r'^[ \t]*' +
                        r'(?P<keyword>' +
                        "|".join(map(re.escape, keywords)) +
                        r')\s+' +
                        r'(?P<type>\S+)\s+' +
                        r'(?P<names>\S+(?:\s*\[\d+\])?' +
                        r'(?:\s*,\s*\S+(?:\s*\[\d+\])?)*)' +
                        r'\s*;\s*?\n',
                        re.MULTILINE)
    name_re = re.compile(r'\s*(?P<name>\S+)(?:\s*\[(?P<array_size>\d+)\])?',
                         re.MULTILINE)

    def handle_replacement(md):
        keyword = md.group('keyword')
        type_name = md.group('type')
        names = md.group('names')

        for var_part in names.split(','):
            md = name_re.match(var_part)
            callback(keyword,
                     type_name,
                     md.group('name'),
                     md.group('array_size'))

        return ""

    return regexp.sub(handle_replacement, source)


def convert_uniforms_to_ubo(shader):
    def handle_uniform(keyword, type_name, name, array_size):
        var = Variable(type_name, name)

        if array_size is not None:
            var.array_size = array_size

        shader.uniforms.append(var)

    shader.source = replace_variables(['uniform'],
                                      shader.source,
                                      handle_uniform)


def convert_attributes(shader):
    if shader.stage != "vertex":
        return

    def handle_attribute(keyword, type_name, name, array_size):
        var = Variable(type_name, name)

        if array_size is not None:
            var.array_size = array_size

        shader.inputs.append(var)

    shader.source = replace_variables(['in', 'attribute'],
                                      shader.source,
                                      handle_attribute)


def convert_varyings(shader):
    keywords = ['varying']

    if shader.stage == "fragment":
        keywords.append('in')
    elif shader.stage == "vertex":
        keywords.append('out')

    def handle_varying(keyword, type_name, name, array_size):
        var = Variable(type_name, name)

        if array_size is not None:
            var.array_size = array_size

        if shader.stage == "fragment":
            shader.inputs.append(var)
        elif shader.stage == "vertex":
            shader.outputs.append(var)
        else:
            raise ScriptError("unsupported shader stage for varying")

    shader.source = replace_variables(keywords,
                                      shader.source,
                                      handle_varying)


def convert_frag_outputs(shader):
    if shader.stage != "fragment":
        return

    def handle_output(keyword, type_name, name, array_size):
        var = Variable(type_name, name)

        if array_size is not None:
            var.array_size = array_size

        shader.outputs.append(var)

    shader.source = replace_variables(['out'],
                                      shader.source,
                                      handle_output)


def convert_mvp(shader):
    if MVP_RE.search(shader.source):
        shader.source = MVP_RE.sub('piglit_mvp', shader.source)
        shader.uniforms.append(Variable('mat4', 'piglit_mvp'))


def convert_frag_color(shader):
    if FRAG_COLOR_RE.search(shader.source):
        shader.source = FRAG_COLOR_RE.sub('piglit_frag_color', shader.source)
        shader.outputs.append(Variable('vec4', 'piglit_frag_color'))


def convert_front_color(shader):
    if FRONT_COLOR_RE.search(shader.source):
        shader.source = FRONT_COLOR_RE.sub('piglit_color_attrib', shader.source)
        shader.outputs.append(Variable('vec4', 'piglit_color_attrib'))


def convert_color(shader):
    if COLOR_RE.search(shader.source):
        shader.source = COLOR_RE.sub('piglit_color_attrib', shader.source)
        shader.inputs.append(Variable('vec4', 'piglit_color_attrib'))


def assign_locations_for_list(variables):
    location = 0

    for var in variables:
        var.location = location

        location += (var.size() + 15) // 16


def assign_locations(shader):
    assign_locations_for_list(shader.inputs)
    assign_locations_for_list(shader.outputs)

def combine_uniforms(processor):
    uniform_names = set()
    uniforms = []

    for shader in processor.shaders:
        for uniform in shader.uniforms:
            if uniform.variable_name in uniform_names:
                continue

            uniforms.append(uniform)
            uniform_names.add(uniform.variable_name)

    for shader in processor.shaders:
        shader.uniforms = uniforms


def assign_uniform_offsets(processor):
    offset = 0

    for uniform in processor.shaders[0].uniforms:
        offset = align(offset, uniform.alignment())

        uniform.offset = offset

        offset += uniform.size()


def set_ubo_size(processor):
    try:
        uniform = processor.shaders[0].uniforms[-1]
    except IndexError:
        return

    size = uniform.offset + type_size(uniform.type_name)

    processor.tests.append("ubo 3 {}".format(size))


def initialize_mvp(processor):
    # Initialise piglit_mvp to the identity matrix
    for uniform in processor.shaders[0].uniforms:
        if uniform.variable_name == "piglit_mvp":
            processor.tests.append(("ubo 3 subdata mat4 {}  "
                                    "1.0 0.0 0.0 0.0 "
                                    "0.0 1.0 0.0 0.0 "
                                    "0.0 0.0 1.0 0.0 "
                                    "0.0 0.0 0.0 1.0").format(uniform.offset))
            break


SHADER_FILTERS = [
    add_version,
    add_gl_vertex_inputs,
    convert_uniforms_to_ubo,
    convert_attributes,
    convert_varyings,
    convert_frag_outputs,
    convert_mvp,
    convert_frag_color,
    convert_front_color,
    convert_color,
    assign_locations,
]

PROGRAM_FILTERS = [
    combine_uniforms,
    assign_uniform_offsets,
    set_ubo_size,
    initialize_mvp,
]

if __name__ == '__main__':
    result = True

    for script_name in sys.argv[1:]:
        try:
            processor = ScriptProcessor()

            with open(script_name, "r", encoding="utf-8") as f:
                for line in f:
                    processor.add_line(line.rstrip())

            processor.end_section()
            processor.run_filters()

            dir_name = "vulkan/" + os.path.dirname(script_name)
            try:
                os.makedirs(dir_name)
            except FileExistsError:
                pass

            vk_script_name = (os.path.splitext(script_name)[0] +
                              ".vk_shader_test")

            with open("vulkan/" + vk_script_name, "w", encoding="utf-8") as f:
                processor.output(f)

        except ScriptError as e:
            print("{}: {}".format(script_name, e), file=sys.stderr)
            result = False
            continue

    if not result:
        sys.exit(1)
