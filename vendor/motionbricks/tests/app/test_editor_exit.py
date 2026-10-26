# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import warp as wp
from imgui_bundle import imgui

from app.authoring.common import FileAction
from app.runtime.common.io.scene_document import SceneDefinition
from app.runtime.common.io.smart_primitive_document import SmartPrimitiveDocument
from app.scene_editor import SceneEditor
from app.smart_primitive_editor import SmartPrimitiveEditor


_EDITORS = (SceneEditor, SmartPrimitiveEditor)


def _editor(editor_type):
    editor = editor_type.__new__(editor_type)
    editor.viewer = Mock()
    editor.viewer.is_running.return_value = True
    editor._exit_requested = False
    editor._document_path = None
    editor._status = None
    editor._update_window_title = Mock()
    editor._capture_camera_state = Mock(return_value={
        "pos": [1.0, 2.0, 3.0], "pivot": [0.0, 0.0, 0.0],
        "pitch": -20.0, "yaw": 45.0, "fov": 60.0, "near": 0.01, "far": 1000.0,
    })
    if editor_type is SceneEditor:
        editor.scene = SceneDefinition()
        item = editor.scene.add_solid_cube()
        editor.selected_item_id = item.id
    else:
        editor.authoring = SmartPrimitiveDocument("soma", SimpleNamespace(), wp.transform_identity())
        item = editor.authoring.add_node()
        editor.selected_id = item.id
    item.name = "Unsaved edit"
    # The Save As dialog returns this mock's value.
    return editor, Mock(return_value="")


def _confirmation_ui(choice):
    ui = Mock()
    ui.begin_popup_modal.return_value = (True, None)
    ui.button.side_effect = lambda label: label == choice
    return ui


class EditorExitTests(unittest.TestCase):
    def test_save_writes_current_document_and_closes_only_after_the_frame(self):
        for editor_type in _EDITORS:
            for new_document in (False, True):
                with self.subTest(editor=editor_type.__name__, new=new_document):
                    with tempfile.TemporaryDirectory() as directory:
                        editor, chooser = _editor(editor_type)
                        suffix = "scene" if editor_type is SceneEditor else "smart_primitive"
                        path = Path(directory) / f"saved.{suffix}.json"
                        chooser.return_value = str(path)
                        if not new_document:
                            editor._document_path = path
                        events = []

                        def render():
                            events.append("frame start")
                            editor._draw_exit_confirmation(_confirmation_ui("Save"))
                            self.assertIn("Unsaved edit", path.read_text(encoding="utf-8"))
                            editor.viewer.close.assert_not_called()
                            events.append("frame complete")

                        editor.render = Mock(side_effect=render)
                        editor.viewer.close.side_effect = lambda: events.append("close")
                        with patch("app.authoring.common.choose_file", chooser):
                            editor.run(max_frames=3)

                        self.assertEqual(events, ["frame start", "frame complete", "close"])
                        self.assertEqual(chooser.call_count, int(new_document))
                        self.assertTrue(editor._exit_requested)

    def test_failed_save_keeps_the_editor_open_and_preserves_the_existing_file(self):
        for editor_type in _EDITORS:
            with self.subTest(editor=editor_type.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    editor, chooser = _editor(editor_type)
                    chooser.return_value = str(Path(directory) / "existing")
                    with patch("app.authoring.common.choose_file", chooser):
                        self.assertTrue(editor._handle_file_action(FileAction.SAVE))
                    path = editor._document_path
                    original = path.read_bytes()
                    editor.selected.name = "Another unsaved edit"
                    with patch("os.replace", side_effect=OSError("disk full")):
                        editor._draw_exit_confirmation(_confirmation_ui("Save"))
                    self.assertFalse(editor._exit_requested)
                    self.assertIn("disk full", editor._status)
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(list(path.parent.iterdir()), [path])
                    editor.viewer.close.assert_not_called()

    def test_dont_save_and_cancel_never_save(self):
        # The document is new, so Save opens Save As. The user cancels it.
        for editor_type in _EDITORS:
            for choice, should_exit in (("Don't Save", True), ("Cancel", False), ("Save", False)):
                with self.subTest(editor=editor_type.__name__, choice=choice):
                    editor, chooser = _editor(editor_type)
                    with patch("app.authoring.common.choose_file", chooser):
                        editor._draw_exit_confirmation(_confirmation_ui(choice))
                    self.assertEqual(chooser.call_count, int(choice == "Save"))
                    self.assertEqual(editor._exit_requested, should_exit)
                    self.assertIsNone(editor._document_path)
                    editor.viewer.close.assert_not_called()

    def test_menu_exit_opens_a_real_imgui_modal_without_closing_the_viewer(self):
        for editor_type in _EDITORS:
            with self.subTest(editor=editor_type.__name__):
                context = imgui.create_context()
                try:
                    io = imgui.get_io()
                    io.set_ini_filename("")
                    io.display_size = (1000.0, 700.0)
                    io.delta_time = 1.0 / 60.0
                    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
                    editor, _ = _editor(editor_type)
                    editor._draw_file_menu = Mock(return_value=FileAction.EXIT)
                    editor._draw_tools_menu = Mock()
                    imgui.new_frame()
                    editor._draw_main_menu_bar(imgui)
                    editor._draw_exit_confirmation(imgui)
                    self.assertTrue(imgui.is_popup_open("Exit"))
                    imgui.render()
                    self.assertFalse(editor._exit_requested)
                    editor.viewer.close.assert_not_called()
                finally:
                    imgui.destroy_context(context)


if __name__ == "__main__":
    unittest.main()
