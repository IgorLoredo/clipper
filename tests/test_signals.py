import json

import numpy as np

from clipper.signals import chat_density, find_peaks, parse_chat_times


def test_find_peaks_detects_spikes_and_respects_gap():
    levels = np.array([-30.5, -29.5] * 100)
    levels[50] = 0.0
    levels[55] = -2.0
    levels[150] = -5.0
    peaks = find_peaks(levels, 1.0, "audio", min_gap=20)
    assert [p.time for p in peaks] == [50.0, 150.0]


def test_find_peaks_flat_signal_returns_nothing():
    assert find_peaks(np.zeros(100), 1.0, "audio") == []


def test_parse_youtube_live_chat_jsonl(tmp_path):
    lines = [json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(ms)}}) for ms in (1500, 62000)]
    path = tmp_path / "live_chat.json"
    path.write_text("\n".join(lines), encoding="utf-8")
    assert parse_chat_times(path) == [1.5, 62.0]


def test_parse_twitch_downloader_json(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps({"comments": [{"content_offset_seconds": 3}, {"content_offset_seconds": 9.5}]}))
    assert parse_chat_times(path) == [3.0, 9.5]


def test_parse_plain_list(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps([1, {"t": 2}, {"time": 3}]))
    assert parse_chat_times(path) == [1.0, 2.0, 3.0]


def test_chat_density_buckets():
    counts = chat_density([0, 1, 12, 25, 99], duration=30, bucket_seconds=10)
    assert counts.tolist() == [2, 1, 1]


def test_youtube_chat_counts_messages_not_banners_or_duplicate_tickers(tmp_path):
    packet = {"replayChatItemAction": {"videoOffsetTimeMsec": "1000", "actions": [
        {"addChatItemAction": {}}, {"addChatItemAction": {}},
        {"addLiveChatTickerItemAction": {}}, {"addBannerToLiveChatCommand": {}},
    ]}}
    path = tmp_path / "chat.json"
    path.write_text(json.dumps(packet))
    assert parse_chat_times(path) == [1.0, 1.0]
