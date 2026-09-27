"""Before / after change screening through the API (two synthetic Mode B jobs of the same place, stub model)."""
from tests.api.test_mode_b import _run, synthetic_scene  # noqa: F401 - fixture re-export


def test_change_candidates_and_run(client, synthetic_scene):
    files = lambda: {"file": ("scene.tif", synthetic_scene["image"], "image/tiff"), "dem": ("dem.tif", synthetic_scene["dem"], "image/tiff")}  # noqa: E731
    a, _ = _run(client, files())
    b, _ = _run(client, files())
    cands = client.get(f"/api/jobs/{a}/change/candidates").json()["candidates"]
    assert [c["job_id"] for c in cands] == [b] and cands[0]["overlap_fraction"] == 1.0
    r = client.post(f"/api/jobs/{a}/change", json={"after": b})
    assert r.status_code == 200, r.text
    s = r.json()["summary"]
    assert s["overlap_fraction"] == 1.0 and s["valid_fraction"] > 0.9
    assert s["pixels"]["loss_area_m2"] == 0 and s["pixels"]["gain_area_m2"] == 0  # same image twice: nothing changed
    assert s["buildings"]["counts"].get("MAJOR_HEIGHT_LOSS", 0) == 0 and s["noise"]["threshold_pixels_m"] >= 2.5
    for k in ("dh_tif", "class_tif", "overlay_png", "after_png"):
        assert client.get(f"/api/jobs/{a}/artifact/{s['artifacts'][k]}").status_code == 200, k
    assert client.post(f"/api/jobs/{a}/change", json={"after": a}).status_code == 400
    assert client.post(f"/api/jobs/{a}/change", json={}).status_code == 400
    assert client.post(f"/api/jobs/{a}/change", json={"after": "nope"}).status_code == 404


def test_change_refused_for_mode_a(client, png_bytes, synthetic_scene):
    a, _ = _run(client, {"file": ("chk.png", png_bytes, "image/png")})
    b, _ = _run(client, {"file": ("scene.tif", synthetic_scene["image"], "image/tiff"), "dem": ("dem.tif", synthetic_scene["dem"], "image/tiff")})
    assert client.get(f"/api/jobs/{a}/change/candidates").json()["candidates"] == []
    assert client.post(f"/api/jobs/{a}/change", json={"after": b}).status_code == 400
