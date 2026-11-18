from clipper.reframe import Framing, camera_path, fill_gaps, resolve_content_type, to_points, x_expression
from clipper.render import video_filter


def test_fill_gaps_uses_nearest_known():
    assert fill_gaps([None, 0.4, None, None, 0.8]) == [0.4, 0.4, 0.4, 0.8, 0.8]
    assert fill_gaps([None, None]) is None


def test_camera_path_ignores_jitter_inside_deadzone():
    times = [i * 0.5 for i in range(20)]
    centers = [0.5, 0.52, 0.48, 0.51] * 5
    assert camera_path(times, centers) == [(0.0, 0.5)]


def test_camera_path_moves_on_big_shift():
    times = [i * 0.5 for i in range(20)]
    centers = [0.3] * 10 + [0.7] * 10
    path = camera_path(times, centers)
    assert len(path) == 2 and path[1][1] == 0.7


def test_to_points_clamps_and_builds_transition():
    points = to_points([(0.0, 0.0), (5.0, 1.0)], src_w=1920, crop_w=606)
    assert points == [(0.0, 0), (5.0, 0), (5.35, 1314)]


def test_x_expression_static_and_moving():
    assert x_expression([(0.0, 100)]) == "100"
    expr = x_expression([(0.0, 0), (5.0, 0), (5.35, 1314)])
    assert expr == "if(lt(t,5.0),0,if(lt(t,5.35),0+(1314)*(t-5.0)/0.35,1314))"


def test_gameplay_titles_use_full_screen_layout_and_override_is_available():
    assert resolve_content_type("GTA 6 - Início de Gameplay") == "gameplay"
    assert resolve_content_type("Conversa sobre tecnologia") == "podcast"
    assert resolve_content_type("Conversa sobre GTA", "podcast") == "podcast"
    assert "force_original_aspect_ratio=decrease" in video_filter(Framing("gameplay", 1920, 1080), None)


def test_anime_and_mass_titles_use_wide_layouts():
    assert resolve_content_type("Anime dublado episodio 3") == "anime"
    assert resolve_content_type("Santa Missa e homilia de domingo") == "mass"
    for mode in ("anime", "mass"):
        assert "force_original_aspect_ratio=decrease" in video_filter(Framing(mode, 1920, 1080), None)
