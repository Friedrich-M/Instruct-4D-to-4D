"""The demo helpers, and the demo commands the README documents.

The demos themselves need model weights, so what is checked here is everything
around them: frame discovery, resizing, batching, and that each command in the
README still matches the argument parser it is aimed at.
"""

import importlib.util
import io
import os
import shlex
import sys
from contextlib import redirect_stdout

import pytest
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMOS = os.path.join(REPO, "demos")
sys.path.insert(0, DEMOS)

from _common import (  # noqa: E402
    SIZE_MULTIPLE,
    fit_to_multiple,
    load_frames,
    output_name,
    save_sheet,
    sorted_image_paths,
)


def write_frames(directory, names, size=(64, 48)):
    for name in names:
        Image.new("RGB", size, (7, 7, 7)).save(os.path.join(directory, name))
    return directory


def test_frames_are_ordered_numerically(tmp_path):
    """Frames are named 0.png, 1.png, 10.png, which sorts wrong as text."""
    write_frames(tmp_path, ["0.png", "1.png", "2.png", "10.png", "11.png"])
    names = [os.path.basename(p) for p in sorted_image_paths(str(tmp_path))]
    assert names == ["0.png", "1.png", "2.png", "10.png", "11.png"]


def test_non_image_files_are_skipped(tmp_path):
    write_frames(tmp_path, ["0.png", "1.png"])
    (tmp_path / ".DS_Store").write_text("")
    (tmp_path / "notes.txt").write_text("")
    assert len(sorted_image_paths(str(tmp_path))) == 2


def test_non_numeric_names_still_work(tmp_path):
    write_frames(tmp_path, ["cam_b.png", "cam_a.png"])
    names = [os.path.basename(p) for p in sorted_image_paths(str(tmp_path))]
    assert names == ["cam_a.png", "cam_b.png"]


def test_an_empty_or_missing_directory_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no images"):
        sorted_image_paths(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not a directory"):
        sorted_image_paths(str(tmp_path / "nope"))


@pytest.mark.parametrize("size", [(200, 150), (1013, 761), (64, 64)])
def test_resizing_lands_on_a_valid_resolution(size):
    """Both the VAE and the UNet need sides divisible by 64."""
    out = fit_to_multiple(Image.new("RGB", size), 512)
    assert out.size[0] % SIZE_MULTIPLE == 0 and out.size[1] % SIZE_MULTIPLE == 0
    assert min(out.size) > 0


def test_frames_load_as_a_batch(tmp_path):
    write_frames(tmp_path, ["0.png", "1.png", "2.png"])
    frames = load_frames(sorted_image_paths(str(tmp_path)), 128, "cpu")
    assert frames.shape[0] == 3 and frames.shape[1] == 3
    assert 0.0 <= float(frames.min()) and float(frames.max()) <= 1.0


def test_mixed_resolutions_are_reported(tmp_path):
    Image.new("RGB", (64, 64)).save(tmp_path / "0.png")
    Image.new("RGB", (96, 32)).save(tmp_path / "1.png")
    with pytest.raises(ValueError, match="share one resolution"):
        load_frames(sorted_image_paths(str(tmp_path)), None, "cpu")


def test_contact_sheet_is_one_row(tmp_path):
    frames = torch.rand(4, 3, 16, 20)
    path = save_sheet(frames, str(tmp_path / "sheet.png"))
    width, height = Image.open(path).size
    assert (width, height) == (4 * 20, 16)


def test_output_names_are_tagged_with_the_prompt():
    assert output_name("What if it was painted by Van Gogh?", "ip2p") == "ip2p_Gogh.png"
    assert output_name("make it snow.", "x") == "x_snow.png"


def readme_demo_commands():
    text = open(os.path.join(REPO, "README.md")).read()
    block = text[text.index("## 🔥 Framework components"):text.index("## 📂 Notes")]
    return [line.strip() for line in block.replace("\\\n", " ").split("\n")
            if line.strip().startswith("python demos/")]


def test_the_readme_documents_every_demo():
    documented = {shlex.split(c)[1] for c in readme_demo_commands()}
    shipped = {f"demos/{n}" for n in os.listdir(DEMOS)
               if n.endswith(".py") and not n.startswith("_")}
    assert documented == shipped, f"undocumented: {shipped - documented}"


@pytest.mark.parametrize("command", readme_demo_commands())
def test_readme_commands_match_the_argument_parsers(command, monkeypatch):
    """A documented flag that the parser does not accept is a broken README."""
    argv = shlex.split(command)
    script, args = os.path.join(REPO, argv[1]), argv[2:]

    spec = importlib.util.spec_from_file_location("demo_under_test", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "argv", [script] + args)
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)     # module body only; main() is guarded
        parsed = module.parse_args()
    assert parsed.prompt


def test_compat_shim_resolves_the_moved_diffusers_symbols():
    """The pseudo-3D UNet forks diffusers internals, and several of the names it
    needs moved between releases. The shim must find them wherever they live."""
    from instruct4d.ip2p import _compat

    assert callable(_compat.maybe_allow_in_graph)
    assert callable(_compat.randn_tensor)
    assert isinstance(_compat.HF_HUB_OFFLINE, bool)


def test_compat_shim_drops_unsupported_keyword_arguments():
    """`use_auth_token` became `token` and `resume_download` went away, so one
    call site has to serve several signatures."""
    from instruct4d.ip2p._compat import call_supported

    def modern(checkpoint_file, token=None):
        return ("modern", checkpoint_file, token)

    def legacy(checkpoint_file, use_auth_token=None, resume_download=False):
        return ("legacy", checkpoint_file, use_auth_token)

    assert call_supported(modern, checkpoint_file="f", token="t",
                          use_auth_token="t", resume_download=True) == ("modern", "f", "t")
    assert call_supported(legacy, checkpoint_file="f", token="t",
                          use_auth_token="t", resume_download=True) == ("legacy", "f", "t")


def test_compat_shim_passes_everything_to_a_var_keyword_callable():
    from instruct4d.ip2p._compat import call_supported

    seen = {}
    call_supported(lambda **kw: seen.update(kw), anything=1, at_all=2)
    assert seen == {"anything": 1, "at_all": 2}
