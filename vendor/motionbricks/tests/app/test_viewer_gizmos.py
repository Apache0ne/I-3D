# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from functools import partial
from types import SimpleNamespace
import unittest

import newton
import numpy as np
import warp as wp
from imgui_bundle import imgui, imguizmo
from newton._src.viewer.viewer_gui import ViewerGui
from pyglet.math import Mat4, Vec3

from app.authoring.common import NewtonViewerUIBridge, log_transform_gizmo
from app.runtime.common.io.scene_document import SceneDefinition
from app.runtime.common.io.smart_primitive_document import SmartPrimitiveDocument
from app.scene_editor import SceneEditor
from app.smart_primitive_editor import SmartPrimitiveEditor


class ViewerGizmoTests(unittest.TestCase):
    """Exercise native ImGuizmo drawing/input without a GL window or model engine."""

    def setUp(self):
        self.context = imgui.create_context()
        self.io = imgui.get_io()
        self.io.set_ini_filename("")
        self.io.display_size = (1920, 1080)
        self.io.delta_time = 1 / 60
        self.io.mouse_double_click_time = 0.001
        self.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        view = np.array(Mat4.look_at(Vec3(4, -7, 8), Vec3(0, 0, 0), Vec3(0, 0, 1)))
        projection = np.array(Mat4.perspective_projection(1920 / 1080, 0.01, 100, 45))
        self.viewer = SimpleNamespace(
            camera=SimpleNamespace(get_view_matrix=lambda: view, get_projection_matrix=lambda: projection),
            _gizmo_log={},
            gizmo_is_using=False,
        )
        self.viewer.log_gizmo = partial(newton.viewer.ViewerGL.log_gizmo, self.viewer)
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = self.viewer
        gui._gizmo_active = {}
        gui.ui = SimpleNamespace(imgui=imgui, giz=imguizmo.im_guizmo, io=self.io, is_available=True)
        self.viewer.gui = gui
        self.bridge = NewtonViewerUIBridge(self.viewer)
        self.bridge.install(imgui)

    def tearDown(self):
        self.io.add_mouse_button_event(0, False)
        self.draw()
        self.viewer._gizmo_log.clear()
        self.draw()
        imgui.destroy_context(self.context)

    def draw(self):
        imgui.new_frame()
        imguizmo.im_guizmo.begin_frame()
        imguizmo.im_guizmo.set_drawlist(imgui.get_background_draw_list())
        self.bridge._render_gizmos()
        vertices = defaultdict(list)
        for vertex in imgui.get_background_draw_list().vtx_buffer:
            vertices[vertex.col].append((vertex.pos.x, vertex.pos.y))
        imgui.render()
        return vertices

    def drag(self, start, end):
        for position, down in ((start, False), (start, True), (end, True), (end, False)):
            self.io.add_mouse_pos_event(*position)
            self.io.add_mouse_button_event(0, down)
            for _ in range(2):
                self.draw()
            self.assertEqual(self.viewer.gizmo_is_using, down)

    def test_editor_locks_show_only_available_planes_and_keep_rotation_independent(self):
        editor = SmartPrimitiveEditor.__new__(SmartPrimitiveEditor)
        editor.viewer = self.viewer
        editor.authoring = SmartPrimitiveDocument("soma", SimpleNamespace(), wp.transform_identity())
        editor.selected_id = editor.authoring.root.id
        # ImGuizmo's default plane fills and axis/ring colors, ordered by normal X/Y/Z.
        plane_colors = (0x610000AA, 0x6100AA00, 0x61AA0000)
        axis_colors = (0xFF0000AA, 0xFF00AA00, 0xFFAA0000)
        for translation_locks, rotation_locks in (
            ((False, False, False), (False, False, False)),
            ((True, False, False), (False, True, False)),
            ((False, True, True), (True, False, True)),
            ((True, True, True), (False, False, False)),
        ):
            with self.subTest(translation=translation_locks, rotation=rotation_locks):
                editor.selected.translation_locks = list(translation_locks)
                editor.selected.rotation_locks = list(rotation_locks)
                editor._draw_selection_gizmo()
                vertices = self.draw()
                for normal in range(3):
                    available = not any(translation_locks[axis] for axis in range(3) if axis != normal)
                    self.assertEqual(len(vertices[plane_colors[normal]]), 4 if available else 0)
                    # A ring has many more vertices than the arrow and square outlines.
                    self.assertEqual(len(vertices[axis_colors[normal]]) >= 64, not rotation_locks[normal])

    def test_plane_and_center_drags_preserve_the_locked_coordinate(self):
        for normal, plane_color in enumerate((0x610000AA, 0x6100AA00, 0x61AA0000)):
            for handle in ("plane", "center"):
                with self.subTest(normal=normal, handle=handle):
                    transform = wp.transform(wp.vec3(0.2, 0.3, 0.4), wp.quat_identity())
                    self.viewer._gizmo_log.clear()
                    log_transform_gizmo(
                        self.viewer, "pose", transform,
                        translation_locks=[axis == normal for axis in range(3)],
                        rotation_locks=[True, True, True],
                    )
                    self.io.add_mouse_pos_event(10, 1000)
                    vertices = self.draw()
                    color = plane_color if handle == "plane" else 0xFFFFFFFF
                    cursor = np.mean(vertices[color], axis=0)
                    before = np.asarray(transform).copy()
                    self.drag(cursor, cursor + [35, 15])
                    after = np.asarray(transform)
                    self.assertEqual(after[normal], before[normal])
                    for axis in range(3):
                        if axis != normal:
                            self.assertGreater(abs(after[axis] - before[axis]), 1e-4)
                    np.testing.assert_allclose(after[3:], before[3:], atol=1e-6)

    def scene_editor_child(self, translation_locks, rotation_locks):
        """Select a cube under a Smart Primitive that is turned 90 degrees, and draw its gizmo."""
        editor = SceneEditor.__new__(SceneEditor)
        editor.viewer = self.viewer
        editor.scene = SceneDefinition()
        editor._runtime_views = {}
        editor._gizmo_item_id = editor._gizmo_transform = editor._gizmo_source_transform = None
        parent, child = editor.scene.add_smart_primitive(), editor.scene.add_solid_cube()
        parent.transform = wp.transform(
            wp.vec3(0.5, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi),
        )
        child.parent_id = parent.id
        child.transform = wp.transform(wp.vec3(0.3, 0.0, 0.2), wp.quat_identity())
        editor.selected_item_id = child.id
        editor._transform_locks = {child.id: (translation_locks, rotation_locks)}
        editor._draw_selection_gizmo()
        self.io.add_mouse_pos_event(10, 1000)
        return editor, child, self.draw()

    def test_child_handles_follow_a_rotated_parent_and_keep_its_locked_axis(self):
        editor, child, vertices = self.scene_editor_child([True, False, False], [True, True, True])
        # With local X locked, only the plane across the parent's Y and Z axes remains.
        self.assertEqual(len(vertices[0x610000AA]), 4)
        self.assertEqual(len(vertices[0x6100AA00]) + len(vertices[0x61AA0000]), 0)
        before = np.asarray(child.transform).copy()
        world_before = np.asarray(editor.scene.world_transform(child)).copy()
        cursor = np.mean(vertices[0x610000AA], axis=0)
        self.drag(cursor, cursor + [35, 15])
        editor._apply_selection_gizmo()
        after = np.asarray(child.transform)
        world_after = np.asarray(editor.scene.world_transform(child))
        self.assertEqual(after[0], before[0])
        self.assertGreater(abs(after[1] - before[1]), 1e-2)
        # The parent's X axis points along the scene's Y axis and its Y axis along -X.
        self.assertAlmostEqual(world_after[1], world_before[1], places=5)
        self.assertGreater(abs(world_after[0] - world_before[0]), 1e-2)
        np.testing.assert_allclose(after[3:], before[3:], atol=1e-6)

    def test_child_rotation_ring_turns_about_the_parent_axis(self):
        editor, child, vertices = self.scene_editor_child([True, True, True], [False, True, True])
        ring = np.asarray(vertices[0xFF0000AA])
        self.assertGreaterEqual(len(ring), 64)
        start = ring[np.argmax(ring[:, 0])]
        self.drag(start, start + [0, 40])
        editor._apply_selection_gizmo()
        roll, pitch, yaw = (float(value) for value in wp.quat_to_rpy(wp.quat(*np.asarray(child.transform)[3:])))
        self.assertGreater(abs(roll), np.radians(1.0))
        self.assertAlmostEqual(pitch, 0.0, places=5)
        self.assertAlmostEqual(yaw, 0.0, places=5)

    def test_masks_do_not_leak_between_gizmos_or_outside_the_bridge(self):
        for name, position, translation_locks in (
            ("xy", (-1, 0, 0), [False, False, True]),
            ("yz", (1, 0, 0), [True, False, False]),
        ):
            log_transform_gizmo(
                self.viewer, name, wp.transform(wp.vec3(*position), wp.quat_identity()),
                translation_locks=translation_locks, rotation_locks=[True, True, True],
            )
        vertices = self.draw()
        self.assertEqual(len(vertices[0x610000AA]), 4)
        self.assertEqual(len(vertices[0x6100AA00]), 0)
        self.assertEqual(len(vertices[0x61AA0000]), 4)
        self.assertIs(self.viewer.gui.ui.giz, imguizmo.im_guizmo)

        self.viewer._gizmo_log.clear()
        self.viewer.log_gizmo("unlocked", wp.transform_identity(), rotate=())
        # Bypass the bridge to verify that native global mask state was restored.
        self.bridge._render_gizmos = self.bridge._gizmos_callback
        vertices = self.draw()
        for color in (0x610000AA, 0x6100AA00, 0x61AA0000):
            self.assertEqual(len(vertices[color]), 4)


if __name__ == "__main__":
    unittest.main()
