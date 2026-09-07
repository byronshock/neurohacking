import math

import pygame
import pytest

from walnutbutter import main
from walnutbutter.cli import cli_main
from walnutbutter.grid import DIRECTIONS, GridOfNeurons
from walnutbutter.neuron import Neuron
from walnutbutter import visualizer as viz


def test_origin_maps_to_zero():
    assert viz.axial_to_pixel(0, 0, 10) == (0, 0)


def test_all_six_neighbours_are_equally_spaced():
    spacing = 10 * viz.SQRT3  # centre-to-centre distance for pointy-top hexagons
    for dq, dr in DIRECTIONS:
        x, y = viz.axial_to_pixel(dq, dr, 10)
        assert math.hypot(x, y) == pytest.approx(spacing)


def test_hexagon_corners_lie_on_the_radius():
    points = viz.hexagon_points(50, 50, 20)
    assert len(points) == 6
    for x, y in points:
        assert math.hypot(x - 50, y - 50) == pytest.approx(20)


@pytest.mark.parametrize("columns, rows", [(9, 5), (3, 11), (1, 1), (24, 20)])
def test_layout_keeps_every_whole_hexagon_inside_the_margin(columns, rows):
    width, height, margin = 640, 480, 10
    grid = GridOfNeurons(columns=columns, rows=rows)
    radius, ox, oy = viz.layout(grid, width, height, margin)
    assert radius > 0
    xs, ys = [], []
    for q, r in grid.neurons:
        dx, dy = viz.axial_to_pixel(q, r, radius)
        for x, y in viz.hexagon_points(ox + dx, oy + dy, radius):
            assert margin - 1e-6 <= x <= width - margin + 1e-6
            assert margin - 1e-6 <= y <= height - margin + 1e-6
            xs.append(x)
            ys.append(y)
    # The grid fills the window in at least one direction and is centred in both.
    assert min(xs) == pytest.approx(margin) or min(ys) == pytest.approx(margin)
    assert (min(xs) + max(xs)) / 2 == pytest.approx(width / 2)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(height / 2)


def test_default_grid_fills_the_default_window_in_both_directions():
    radius, _, _ = viz.layout(GridOfNeurons(columns=24, rows=20), 800, 600)
    grid_w = radius * viz.SQRT3 * 24.5  # 24 columns plus the half-cell shift of odd rows
    grid_h = radius * (1.5 * 19 + 2)
    assert grid_w == pytest.approx(752)  # 800 minus two 24px margins
    assert grid_h / (600 - 48) > 0.95


def test_unfired_is_grey_and_fired_shades_by_wave():
    assert viz.neuron_colour(None, 3) == viz.UNFIRED
    assert viz.neuron_colour(0, 3) == viz.FIRED_CENTRE
    assert viz.neuron_colour(3, 3) == viz.FIRED_EDGE
    assert viz.neuron_colour(0, 0) == viz.FIRED_CENTRE  # a single wave must not divide by zero


def test_draw_grid_paints_fired_and_unfired_neurons(capsys):
    grid = GridOfNeurons(columns=5, rows=5)
    surface = pygame.Surface((300, 300))

    viz.draw_grid(surface, grid)  # nothing fired yet
    radius, ox, oy = viz.layout(grid, 300, 300)
    dx, dy = viz.axial_to_pixel(1, 0, radius)
    neighbour_pixel = (round(ox + dx), round(oy + dy))
    assert surface.get_at(neighbour_pixel)[:3] == viz.UNFIRED
    assert surface.get_at((2, 2))[:3] == viz.BACKGROUND

    grid.activate_origin()
    viz.draw_grid(surface, grid)
    assert surface.get_at(neighbour_pixel)[:3] != viz.UNFIRED


def test_save_writes_an_image_file(tmp_path, capsys):
    grid = GridOfNeurons(columns=5, rows=5)
    grid.activate_origin()
    out = tmp_path / "grid.png"
    viz.save(grid, str(out), width=200, height=200)
    assert out.exists() and out.stat().st_size > 0
    assert pygame.image.load(str(out)).get_size() == (200, 200)


def test_cli_save_option(tmp_path, capsys):
    out = tmp_path / "cli.png"
    assert cli_main(["--headless", "--columns", "4", "--rows", "4", "--weight", "1", "--save", str(out)]) == 0
    assert out.exists()
    assert "Saved" in capsys.readouterr().err


def test_cli_window_option_sets_image_size(tmp_path, capsys):
    out = tmp_path / "wide.png"
    assert cli_main(["--headless", "--columns", "4", "--rows", "4", "--save", str(out), "--window", "320", "200"]) == 0
    assert pygame.image.load(str(out)).get_size() == (320, 200)


# --- the interactive window ---------------------------------------------------


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


def test_space_runs_a_new_epoch_every_time(capsys):
    grid = main(columns=8, rows=4, weight=1.0, seed=1)
    inputs = {tuple(grid.input_bits)}
    for expected_epoch in (2, 3, 4, 5):
        assert viz.handle_event(key(pygame.K_SPACE), grid) == (True, True)
        assert grid.epoch == expected_epoch
        assert grid.waves[0].fired == grid.input_neurons()  # fresh wave 0 each time
        inputs.add(tuple(grid.input_bits))
    assert len(inputs) > 1


def test_r_key_is_no_longer_special(capsys):
    grid = main(columns=8, rows=4, weight=1.0, seed=1)
    assert viz.handle_event(key(pygame.K_r), grid) == (True, False)
    assert grid.fired_neurons()  # nothing was reset


def test_quit_keys_and_window_close_stop_the_loop():
    grid = GridOfNeurons(columns=4, rows=3)
    assert viz.handle_event(key(pygame.K_ESCAPE), grid) == (False, False)
    assert viz.handle_event(key(pygame.K_q), grid) == (False, False)
    assert viz.handle_event(pygame.event.Event(pygame.QUIT), grid) == (False, False)
    assert viz.handle_event(key(pygame.K_x), grid) == (True, False)  # unknown key ignored
    assert grid.fired_neurons() == []


def test_caption_reports_state(capsys):
    grid = GridOfNeurons(columns=4, rows=3, weight=1.0, omega=0)
    assert viz.caption(grid).startswith("walnutbutter 4x3: unfired")
    grid.set_input_bits([True, False])
    grid.fire_input()
    text = viz.caption(grid)
    assert "epoch 1: 12 of 12 fired" in text and "[Space] new input" in text and "[R]" not in text


def test_show_opens_on_the_fired_mesh_and_space_advances_epochs(monkeypatch, capsys):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    grid = main(columns=8, rows=4, weight=1.0, seed=1)
    seen = []
    scripted = [[key(pygame.K_SPACE)], [key(pygame.K_SPACE)], [pygame.event.Event(pygame.QUIT)]]

    def fake_get():
        seen.append(viz.caption(grid))
        return scripted.pop(0) if scripted else [pygame.event.Event(pygame.QUIT)]

    monkeypatch.setattr(pygame.event, "get", fake_get)
    viz.show(grid, 200, 150)
    assert "epoch 1:" in seen[0] and "unfired" not in seen[0]  # no pre-activation preview
    assert "epoch 3:" in seen[2]
    assert grid.epoch == 3  # state of the last epoch survives closing


def test_cli_show_opens_on_an_already_fired_mesh(monkeypatch, capsys):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    shown = []

    def fake_show(grid, width, height, fast=False, **kwargs):
        shown.append((grid.epoch, len(grid.fired_neurons())))

    monkeypatch.setattr(viz, "show", fake_show)
    assert cli_main(["--columns", "4", "--rows", "4", "--weight", "1", "--step"]) == 0
    assert shown == [(1, 16)]
    assert "16 of 16 neurons fired" in capsys.readouterr().err


def test_space_input_lands_on_the_bottom_row(capsys):
    grid = main(columns=4, rows=3, weight=1.0, seed=1)
    assert viz.handle_event(key(pygame.K_SPACE), grid) == (True, True)
    bottom = max(r for _, r in grid.neurons)
    assert grid.waves[0].fired == grid.input_neurons()
    assert all(n.position[1] == bottom for n in grid.waves[0].fired)
    assert grid.get_origin_neuron().fired_in_wave > 0


def test_stimulus_ring_is_drawn_on_input_neurons_before_firing(capsys):
    grid = GridOfNeurons(columns=4, rows=3, weight=1.0)
    grid.set_input([True, False, False, False])
    surface = pygame.Surface((300, 300))
    viz.draw_grid(surface, grid)
    radius, ox, oy = viz.layout(grid, 300, 300)
    dx, dy = viz.axial_to_pixel(*grid.input_row()[0].position, radius)
    # the ring is a hexagon outline at 0.55 of the cell radius; sample straight up from the centre
    ring_pixel = (round(ox + dx), round(oy + dy - 0.55 * radius))
    assert surface.get_at(ring_pixel)[:3] == viz.ORIGIN_RING
    dx, dy = viz.axial_to_pixel(0, 0, radius)
    assert surface.get_at((round(ox + dx), round(oy + dy - 0.55 * radius)))[:3] != viz.ORIGIN_RING


def test_fast_mode_free_runs_between_monitor_frames(monkeypatch, capsys):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    grid = main(columns=8, rows=4, weight=1.0, seed=1)
    capsys.readouterr()
    frames = []
    scripted = [[], [], [], [pygame.event.Event(pygame.QUIT)]]

    def fake_get():
        frames.append(grid.epoch)
        return scripted.pop(0) if scripted else [pygame.event.Event(pygame.QUIT)]

    monkeypatch.setattr(pygame.event, "get", fake_get)
    viz.show(grid, 200, 150, fast=True, fps=50)  # 20 ms per frame; an 8x4 epoch takes well under 1 ms
    assert frames[0] == 1
    assert all(later > earlier for earlier, later in zip(frames, frames[1:]))  # epochs advanced between every sample
    assert frames[1] - frames[0] > 1  # many epochs per frame, not one: the system is not paced by the display
    assert grid.epoch == frames[-1]  # no epochs after the quit
    assert capsys.readouterr().out == ""  # the free run is silent
    assert Neuron.verbose  # and printing is restored afterwards


def test_fast_caption_reports_the_rate(capsys):
    grid = main(columns=8, rows=4, weight=1.0, seed=1)
    text = viz.caption_fast(grid, 1234.5, 30)
    assert "free-running at 1,234 epochs/s, monitored at 30 Hz" in text
    assert "[Space]" not in text


def test_cli_defaults_to_a_free_running_learning_window(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(viz, "show", lambda grid, w, h, fast=False, teacher=None, **kw: calls.append((fast, teacher is not None, kw.get("report_seconds"))))
    assert cli_main(["--columns", "4", "--rows", "4", "--weight", "1"]) == 0
    assert calls == [(True, True, 1.0)]
    calls.clear()
    assert cli_main(["--columns", "4", "--rows", "4", "--weight", "1", "--step", "--no-learn", "--report", "5"]) == 0
    assert calls == [(False, False, 5.0)]
    calls.clear()
    assert cli_main(["--headless", "--columns", "4", "--rows", "4", "--weight", "1"]) == 0
    assert calls == []  # headless never opens the window


def test_window_teaches_after_each_epoch_and_shows_accuracy(monkeypatch, capsys):
    from walnutbutter.learning import Teacher
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    grid = main(columns=8, rows=4, weight=None, seed=1)
    teacher = Teacher(grid, target="all-off", lr=0.01, seed=1)
    teacher.step()  # learn from the first epoch, as the command line does
    assert viz.handle_event(key(pygame.K_SPACE), grid, teacher) == (True, True)
    assert teacher.epochs == 2 and grid.epoch == 2
    assert any(n.noise != 0 for n in grid.neurons.values())  # Space ran the epoch with exploration
    assert "learning all-off" in viz.caption(grid, teacher)
    scripted = [[], [pygame.event.Event(pygame.QUIT)]]
    monkeypatch.setattr(pygame.event, "get", lambda: scripted.pop(0) if scripted else [pygame.event.Event(pygame.QUIT)])
    viz.show(grid, 200, 150, fast=True, fps=50, teacher=teacher)
    assert teacher.epochs == grid.epoch  # one teaching step per epoch, in fast mode too


def test_format_elapsed():
    assert viz.format_elapsed(0) == "0:00:00"
    assert viz.format_elapsed(59.9) == "0:00:59"
    assert viz.format_elapsed(3661) == "1:01:01"
    assert viz.format_elapsed(10 * 3600 + 5) == "10:00:05"


def test_free_run_logs_progress_with_accuracy_to_date(monkeypatch, capsys):
    from walnutbutter.learning import Teacher
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    grid = main(columns=8, rows=4, weight=None, seed=1)
    teacher = Teacher(grid, target="all-off", seed=1)
    lines = []
    scripted = [[], [], [pygame.event.Event(pygame.QUIT)]]
    monkeypatch.setattr(pygame.event, "get", lambda: scripted.pop(0) if scripted else [pygame.event.Event(pygame.QUIT)])
    viz.show(grid, 200, 150, fast=True, fps=50, teacher=teacher, report_seconds=0, log=lines.append)
    assert len(lines) >= 2  # one line per frame when report_seconds is 0
    assert lines[-1].startswith("[0:00:0") and "to date over" in lines[-1] and "epochs/s" in lines[-1]
    assert capsys.readouterr().err == ""  # the custom log took the lines, nothing leaked to stderr


def test_free_run_calls_on_report_after_each_report(monkeypatch, capsys):
    from walnutbutter.learning import Teacher
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    grid = main(columns=8, rows=4, weight=None, seed=1)
    teacher = Teacher(grid, seed=1)
    reports, saves = [], []
    scripted = [[], [], [pygame.event.Event(pygame.QUIT)]]
    monkeypatch.setattr(pygame.event, "get", lambda: scripted.pop(0) if scripted else [pygame.event.Event(pygame.QUIT)])
    viz.show(grid, 200, 150, fast=True, fps=50, teacher=teacher, report_seconds=0,
             log=reports.append, on_report=lambda: saves.append(grid.epoch))
    assert len(saves) == len(reports) >= 2


def test_free_run_records_history_before_each_checkpoint(monkeypatch, capsys):
    from walnutbutter.learning import Teacher
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    grid = main(columns=8, rows=4, weight=None, seed=1)
    teacher = Teacher(grid, seed=1)
    seen_at_checkpoint = []
    scripted = [[], [], [pygame.event.Event(pygame.QUIT)]]
    monkeypatch.setattr(pygame.event, "get", lambda: scripted.pop(0) if scripted else [pygame.event.Event(pygame.QUIT)])
    viz.show(grid, 200, 150, fast=True, fps=50, teacher=teacher, report_seconds=0, log=lambda line: None,
             on_report=lambda: seen_at_checkpoint.append(len(teacher.history)))
    assert len(teacher.history) >= 2
    assert seen_at_checkpoint == list(range(1, len(teacher.history) + 1))  # each checkpoint saw the entry just added
    assert all(e["elapsed"] is not None and e["epochs_per_second"] is not None for e in teacher.history)
