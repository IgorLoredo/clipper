from pathlib import Path
from types import SimpleNamespace

import pytest
import yt_dlp

from clipper import ingest


@pytest.fixture
def downloader(monkeypatch):
    state = SimpleNamespace(calls=[], fail_video=False, fail_chat=False, heatmap=[])

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):
            return {"title": "Video", "id": "123", "heatmap": state.heatmap}

        def process_ie_result(self, info, download=True):
            state.calls.append(self.opts)
            if self.opts.get("skip_download"):
                assert Path(self.opts["outtmpl"].replace("%(ext)s", "mp4")).is_file()
                if state.fail_chat:
                    raise yt_dlp.utils.DownloadError("HTTP Error 503: Service Unavailable")
            else:
                if state.fail_video:
                    raise yt_dlp.utils.DownloadError("video unavailable")
                Path(self.opts["outtmpl"].replace("%(ext)s", "mp4")).touch()

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(ingest, "ffmpeg_exe", lambda: "bundled-ffmpeg.exe")
    return state


def test_chat_failure_does_not_prevent_using_downloaded_video(tmp_path, downloader, capsys):
    downloader.fail_chat = True
    source = ingest._download("https://example.invalid/video", tmp_path)
    assert source.video.is_file()
    assert source.chat is None
    assert len(downloader.calls) == 2
    assert not downloader.calls[0].get("writesubtitles")
    assert downloader.calls[0]["ffmpeg_location"] == "bundled-ffmpeg.exe"
    assert downloader.calls[1]["skip_download"] is True
    assert downloader.calls[1]["ignoreerrors"] is True
    assert "continuing without it" in capsys.readouterr().out


def test_video_failure_is_not_ignored(tmp_path, downloader):
    downloader.fail_video = True
    with pytest.raises(yt_dlp.utils.DownloadError, match="video unavailable"):
        ingest._download("https://example.invalid/video", tmp_path)
    assert len(downloader.calls) == 1


def test_existing_video_does_not_treat_source_metadata_as_chat(tmp_path, downloader):
    workdir = tmp_path / "video-123"
    workdir.mkdir()
    (workdir / "source.mp4").touch()
    (workdir / "source.json").write_text("{}")
    source = ingest._download("https://example.invalid/video", tmp_path)
    assert source.chat is None
    assert downloader.calls == []


def test_unmerged_stream_is_not_a_completed_video(tmp_path):
    (tmp_path / "source.f137.mp4").touch()
    assert ingest._find_video(tmp_path) is None


def test_replay_metrics_are_saved_even_when_video_is_already_downloaded(tmp_path, downloader):
    workdir = tmp_path / "video-123"
    workdir.mkdir()
    (workdir / "source.mp4").touch()
    downloader.heatmap = [{"start_time": 0, "end_time": 30, "value": 0.9}]
    source = ingest._download("https://example.invalid/video", tmp_path)
    assert source.engagement.is_file()
    assert source.chat is None
    assert downloader.calls == []
