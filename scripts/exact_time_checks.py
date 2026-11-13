#!/usr/bin/env python3
"""Red/green checks for exact media-time trim coordinates.

The suite exercises the media layer, the plan compiler, and the public HTTP
boundary. Seam assertions use per-sample/per-frame distinguishable source
media so a duplicated or dropped sample is observable.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import math
import struct
import sys
import tempfile
import threading
import unittest
import wave
from fractions import Fraction
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avprovenance.errors import PlanInvalid
from avprovenance.media.audio_ops import trim_audio
from avprovenance.media.errors import MediaOperationFailure
from avprovenance.media.video_ops import trim_video
from avprovenance.media.wav import AudioClip, decode_wav, encode_wav
from avprovenance.media.y4m import decode_y4m, encode_y4m
from avprovenance.renditions.plan import compile_plan
from avprovenance.http_api import build_http_server


# --------------------------------------------------------------------------
# Media fixtures
# --------------------------------------------------------------------------


def make_index_ramp_wav(sample_rate: int, frame_count: int) -> bytes:
    """Every sample equals its index, so positions are byte-observable."""
    values = list(range(frame_count))
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(values)}h", *values))
    return payload.getvalue()


def make_sine_wav(
    *,
    frequency: float,
    duration_seconds: float = 0.6,
    sample_rate: int = 8000,
    amplitude: int = 9000,
) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    values = [
        int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
        for index in range(frame_count)
    ]
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(values)}h", *values))
    return payload.getvalue()


def make_index_ramp_y4m(
    *,
    width: int = 16,
    height: int = 8,
    frame_count: int = 8,
    rate_numerator: int = 12,
    rate_denominator: int = 1,
) -> bytes:
    header = (
        "YUV4MPEG2 "
        f"W{width} H{height} "
        f"F{rate_numerator}:{rate_denominator} "
        "Ip C420jpeg\n"
    ).encode("ascii")
    chroma_width = width // 2
    chroma_height = height // 2
    parts = [header]
    for frame_index in range(frame_count):
        parts.append(b"FRAME\n")
        y_plane = bytes(frame_index for _ in range(width * height))
        u_plane = bytes(128 for _ in range(chroma_width * chroma_height))
        v_plane = bytes(128 for _ in range(chroma_width * chroma_height))
        parts.extend((y_plane, u_plane, v_plane))
    return b"".join(parts)


def frame_luma_values(clip: Any) -> list[int]:
    return [frame.y_plane[0] for frame in clip.frames]


def first_channel_values(data: bytes) -> list[int]:
    return decode_wav(data).channels[0]


def exact_op(
    start: tuple[int, int],
    *,
    end: tuple[int, int] | None = None,
    duration: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build a client-shaped exact trim (no persisted interpretation fields)."""
    operation: dict[str, Any] = {
        "type": "media_trim",
        "start_time": {"numerator": start[0], "denominator": start[1]},
    }
    if end is not None:
        operation["end_time"] = {"numerator": end[0], "denominator": end[1]}
    else:
        assert duration is not None
        operation["duration_time"] = {
            "numerator": duration[0],
            "denominator": duration[1],
        }
    return operation


def compiled_op(
    start: tuple[int, int],
    *,
    end: tuple[int, int] | None = None,
    duration: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Compile a client-shaped exact trim into its persisted/normalized form."""
    plan = compile_plan("audio", [exact_op(start, end=end, duration=duration)], has_secondary=False)
    return plan.operations[0]


# --------------------------------------------------------------------------
# HTTP helper
# --------------------------------------------------------------------------


class Client:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=15)
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        status, body = self.request(
            "POST",
            "/v1/assets",
            {
                "filename": filename,
                "data_base64": base64.b64encode(data).decode("ascii"),
            },
        )
        assert status == 201, body
        return body

    def render(self, payload: dict[str, Any]) -> dict[str, Any]:
        status, body = self.request("POST", "/v1/renditions", payload)
        assert status == 201, body
        return body


class ServerFixture:
    def __enter__(self) -> tuple[Client, Any]:
        self._temporary = tempfile.TemporaryDirectory(prefix="avprovenance-exact-")
        self.server = build_http_server(
            "127.0.0.1",
            0,
            Path(self._temporary.name),
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        self.thread.start()
        host, port = self.server.server_address[:2]
        return Client(str(host), int(port)), self.server

    def __exit__(self, *exc_info: object) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()
        self._temporary.cleanup()


# --------------------------------------------------------------------------
# Audio seams
# --------------------------------------------------------------------------


class ExactAudioSeamTests(unittest.TestCase):
    sample_rate = 8000

    def test_adjacent_selections_keep_every_sample_once(self) -> None:
        data = make_index_ramp_wav(self.sample_rate, 64)
        clip = decode_wav(data)
        # 20/8000 s is exactly sample index 20 at this rate.
        split = (20, 8000)
        first = trim_audio(clip, compiled_op((0, 1), end=split))
        second = trim_audio(clip, compiled_op(split, end=(40, 8000)))
        ids = first.channels[0] + second.channels[0]
        self.assertEqual(ids, list(range(40)))
        self.assertEqual(len(first.channels[0]), 20)
        self.assertEqual(len(second.channels[0]), 20)

    def test_fractional_boundary_floors_to_one_sample(self) -> None:
        clip = AudioClip(
            sample_rate=self.sample_rate,
            channels=(list(range(self.sample_rate)),),
        )
        # 1/3 s * 8000 = 2666.66.. samples -> floor at 2666; end at 8000.
        trimmed = trim_audio(clip, compiled_op((1, 3), end=(1, 1)))
        self.assertEqual(trimmed.channels[0][0], 2666)
        self.assertEqual(len(trimmed.channels[0]), 8000 - 2666)

    def test_duration_uses_exact_sum_not_rounded_offsets(self) -> None:
        clip = AudioClip(
            sample_rate=self.sample_rate,
            channels=(list(range(self.sample_rate)),),
        )
        # start=1/3 (2666.6..), duration=1/3 -> end=2/3 (5333.3..).
        trimmed = trim_audio(clip, compiled_op((1, 3), duration=(1, 3)))
        self.assertEqual(trimmed.channels[0][0], 2666)
        self.assertEqual(trimmed.channels[0][-1], 5332)
        self.assertEqual(len(trimmed.channels[0]), 5333 - 2666)

    def test_seam_at_rate_not_dividing_denominator(self) -> None:
        # 48 kHz, 1/30 s selections -> exactly 1600 samples each.
        clip = AudioClip(
            sample_rate=48000,
            channels=(list(range(48000)),),
        )
        for index in range(3):
            first = trim_audio(
                clip,
                compiled_op((index, 30), end=(index + 1, 30)),
            )
            second = trim_audio(
                clip,
                compiled_op((index + 1, 30), end=(index + 2, 30)),
            )
            expected = list(range(index * 1600, (index + 2) * 1600))
            self.assertEqual(
                first.channels[0] + second.channels[0],
                expected,
                f"seam at {index}/30s",
            )


# --------------------------------------------------------------------------
# Video seams
# --------------------------------------------------------------------------


class ExactVideoSeamTests(unittest.TestCase):
    def test_adjacent_frame_selections_keep_every_frame_once(self) -> None:
        clip = decode_y4m(
            make_index_ramp_y4m(frame_count=8, rate_numerator=12)
        )
        # 1/12 s per frame. Select frames [1,2) and [2,3) at exact boundaries.
        first = trim_video(clip, compiled_op((1, 12), end=(2, 12)))
        second = trim_video(clip, compiled_op((2, 12), end=(3, 12)))
        self.assertEqual(
            frame_luma_values(first) + frame_luma_values(second),
            [1, 2],
        )

    def test_fractional_frame_boundary_floors(self) -> None:
        clip = decode_y4m(
            make_index_ramp_y4m(frame_count=8, rate_numerator=12)
        )
        # 1/8 s * 12 fps = 1.5 frames -> floor at frame 1.
        trimmed = trim_video(clip, compiled_op((1, 8), duration=(1, 8)))
        # start 1/8 s -> floor(1.5)=1; end 2/8 s -> floor(3.0)=3.
        self.assertEqual(frame_luma_values(trimmed), [1, 2])
        self.assertEqual(trimmed.frame_count, 2)

    def test_duration_does_not_drift_from_editing_fraction(self) -> None:
        clip = decode_y4m(
            make_index_ramp_y4m(
                frame_count=8,
                rate_numerator=24,
                rate_denominator=1,
            )
        )
        # Two adjacent selections each lasting 1/24 s must tile frames.
        first = trim_video(
            clip,
            compiled_op((0, 1), duration=(1, 24)),
        )
        second = trim_video(
            clip,
            compiled_op((1, 24), duration=(1, 24)),
        )
        self.assertEqual(
            frame_luma_values(first) + frame_luma_values(second),
            [0, 1],
        )

    def test_tiling_covers_source_period(self) -> None:
        clip = decode_y4m(
            make_index_ramp_y4m(frame_count=6, rate_numerator=24)
        )
        frames = []
        index = 0
        while index < 5:
            segment = trim_video(
                clip,
                compiled_op((index, 24), end=(index + 1, 24)),
            )
            frames.extend(frame_luma_values(segment))
            index += 1
        self.assertEqual(frames, [0, 1, 2, 3, 4])


# ------------------------------------------------------------------------------
# Legacy compatibility
# ------------------------------------------------------------------------------


class LegacySecondsCompatibilityTests(unittest.TestCase):
    def test_seconds_plan_keeps_legacy_rounding_interpretation(self) -> None:
        plan = compile_plan(
            "audio",
            [
                {
                    "type": "media_trim",
                    "start_seconds": 0.1,
                    "duration_seconds": 0.3,
                }
            ],
            has_secondary=False,
        )
        self.assertEqual(
            plan.operations[0],
            {
                "type": "media_trim",
                "start_seconds": 0.1,
                "duration_seconds": 0.3,
            },
        )
        self.assertNotIn("time_mode", plan.operations[0])

    def test_seconds_plan_still_renders_with_round_not_floor(self) -> None:
        data = make_index_ramp_wav(8000, 4000)
        clip = decode_wav(data)
        # 0.0007 s * 8000 = 5.6 -> round=6, floor would be 5.
        trimmed = trim_audio(
            clip,
            {"start_seconds": 0.0007, "duration_seconds": 0.3},
        )
        self.assertEqual(trimmed.channels[0][0], 6)

    def test_seconds_plan_digest_is_unchanged_by_extension(self) -> None:
        # Digest of a legacy plan must be identical before and after the
        # exact-time capability ships, because persisted plans are replayed.
        plan = compile_plan(
            "video",
            [{"type": "media_trim", "start_frame": 1, "frame_count": 4}],
            has_secondary=False,
        )
        self.assertEqual(
            plan.digest,
            hashlib.sha256(
                json.dumps(
                    {
                        "media_kind": "video",
                        "operations": [
                            {
                                "type": "media_trim",
                                "start_frame": 1,
                                "frame_count": 4,
                            }
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
        )


# ------------------------------------------------------------------------------
# Plan validation
# ------------------------------------------------------------------------------


class ExactPlanValidationTests(unittest.TestCase):
    def test_exact_trim_records_interpretation(self) -> None:
        plan = compile_plan(
            "audio",
            [exact_op((1, 3), duration=(2, 3))],
            has_secondary=False,
        )
        operation = plan.operations[0]
        self.assertEqual(operation["time_mode"], "exact")
        self.assertEqual(operation["boundary_rule"], "floor")

    def test_media_time_is_reduced_to_lowest_terms(self) -> None:
        plan = compile_plan(
            "audio",
            [
                {
                    "type": "media_trim",
                    "start_time": {"numerator": 10, "denominator": 40},
                    "duration_time": {"numerator": 10, "denominator": 40},
                }
            ],
            has_secondary=False,
        )
        self.assertEqual(
            plan.operations[0]["start_time"],
            {"numerator": 1, "denominator": 4},
        )

    def test_string_fraction_is_accepted_and_normalized(self) -> None:
        plan = compile_plan(
            "video",
            [
                {
                    "type": "media_trim",
                    "start_time": "1/12",
                    "end_time": "2/12",
                }
            ],
            has_secondary=False,
        )
        self.assertEqual(
            plan.operations[0]["start_time"],
            {"numerator": 1, "denominator": 12},
        )
        self.assertEqual(
            plan.operations[0]["end_time"],
            {"numerator": 1, "denominator": 6},
        )

    def test_both_end_and_duration_is_invalid(self) -> None:
        operation = exact_op((0, 1), end=(1, 1))
        operation["duration_time"] = {"numerator": 1, "denominator": 1}
        operation["duration_time"] = {"numerator": 1, "denominator": 1}
        with self.assertRaises(PlanInvalid):
            compile_plan("audio", [operation], has_secondary=False)

    def test_mixed_time_and_seconds_fields_is_invalid(self) -> None:
        operation = exact_op((0, 1), end=(1, 1))
        operation["start_seconds"] = 0.0
        with self.assertRaises(PlanInvalid):
            compile_plan("audio", [operation], has_secondary=False)

    def test_non_positive_denominator_is_invalid(self) -> None:
        operation = {
            "type": "media_trim",
            "start_time": {"numerator": 1, "denominator": 0},
            "duration_time": {"numerator": 1, "denominator": 2},
        }
        with self.assertRaises(PlanInvalid):
            compile_plan("audio", [operation], has_secondary=False)

    def test_empty_exact_range_is_invalid(self) -> None:
        operation = exact_op((1, 2), end=(1, 2))
        with self.assertRaises(PlanInvalid):
            compile_plan("audio", [operation], has_secondary=False)

    def test_exact_trim_requires_an_endpoint(self) -> None:
        operation = {
            "type": "media_trim",
            "start_time": {"numerator": 0, "denominator": 1},
        }
        with self.assertRaises(PlanInvalid):
            compile_plan("audio", [operation], has_secondary=False)

    def test_oversized_fraction_string_is_rejected(self) -> None:
        for start in ("1/" + "9" * 30, "+1/2", "1/-2", "1/2/3"):
            with self.subTest(start=start):
                operation = {
                    "type": "media_trim",
                    "start_time": start,
                    "end_time": "1/1",
                }
                with self.assertRaises(PlanInvalid):
                    compile_plan("audio", [operation], has_secondary=False)

    def test_float_numerator_is_rejected(self) -> None:
        operation = {
            "type": "media_trim",
            "start_time": {"numerator": 1.0, "denominator": 2},
            "duration_time": {"numerator": 3, "denominator": 2},
        }
        with self.assertRaises(PlanInvalid):
            compile_plan("audio", [operation], has_secondary=False)


# ------------------------------------------------------------------------------
# Stored-plan interpretation stability
# ------------------------------------------------------------------------------


class StoredPlanInterpretationTests(unittest.TestCase):
    def test_exact_operation_without_recorded_policy_is_refused(self) -> None:
        clip = decode_y4m(make_index_ramp_y4m(frame_count=4))
        operation = {
            "type": "media_trim",
            "start_time": {"numerator": 0, "denominator": 1},
            "duration_time": {"numerator": 1, "denominator": 12},
        }
        with self.assertRaises(Exception) as raised:
            trim_video(clip, operation)
        self.assertIsInstance(raised.exception, MediaOperationFailure)
        self.assertEqual(raised.exception.code, "VIDEO_TRIM_TIME_INVALID")

    def test_unknown_boundary_rule_is_refused(self) -> None:
        clip = decode_y4m(make_index_ramp_y4m(frame_count=4))
        operation = {
            "type": "media_trim",
            "time_mode": "exact",
            "boundary_rule": "round",
            "start_time": {"numerator": 0, "denominator": 1},
            "duration_time": {"numerator": 1, "denominator": 12},
        }
        with self.assertRaises(Exception) as raised:
            trim_video(clip, operation)
        self.assertIsInstance(raised.exception, MediaOperationFailure)
        self.assertEqual(raised.exception.code, "VIDEO_TRIM_TIME_INVALID")


# ------------------------------------------------------------------------------
# End-to-end HTTP + replay
# ------------------------------------------------------------------------------


class ExactTrimHTTPTests(unittest.TestCase):
    def test_audio_exact_trim_is_observable_and_replayable(self) -> None:
        source = make_index_ramp_wav(8000, 4000)
        with ServerFixture() as (client, server):
            asset = client.upload("ramp.wav", source)
            rendition = client.render(
                {
                    "primary_asset_id": asset["id"],
                    "output_name": "seam.wav",
                    "operations": [
                        exact_op((1, 4), end=(2, 4)),
                    ],
                }
            )
            self.assertEqual(rendition["status"], "completed")
            output_bytes = _fetch_output(server, rendition)
            values = first_channel_values(output_bytes)
            self.assertEqual(values, list(range(2000, 4000)))
            descriptor = rendition["output_asset"]["descriptor"]
            self.assertEqual(
                descriptor["properties"]["frame_count"],
                2000,
                "exact trim length",
            )

            status, check = client.request(
                "POST",
                f"/v1/renditions/{rendition['id']}/integrity-checks",
                {},
            )
            self.assertEqual(status, 201)
            self.assertEqual(check["status"], "verified")

    def test_video_exact_trim_seam_is_observable_and_replayable(self) -> None:
        source = make_index_ramp_y4m(frame_count=8, rate_numerator=12)
        with ServerFixture() as (client, server):
            asset = client.upload("ramp.y4m", source)
            rendition = client.render(
                {
                    "primary_asset_id": asset["id"],
                    "output_name": "seam.y4m",
                    "operations": [
                        exact_op((2, 12), end=(4, 12)),
                    ],
                }
            )
            output_bytes = _fetch_output(server, rendition)
            values = frame_luma_values(decode_y4m(output_bytes))
            self.assertEqual(values, [2, 3])
            descriptor = rendition["output_asset"]["descriptor"]
            self.assertEqual(
                descriptor["properties"]["frame_count"],
                2,
                "exact trim frame count",
            )

            status, check = client.request(
                "POST",
                f"/v1/renditions/{rendition['id']}/integrity-checks",
                {},
            )
            self.assertEqual(status, 201)
            self.assertEqual(check["status"], "verified")

    def test_adjacent_selections_via_http_do_not_overlap(self) -> None:
        source = make_index_ramp_wav(8000, 4000)
        with ServerFixture() as (client, server):
            asset = client.upload("ramp.wav", source)
            left = client.render(
                {
                    "primary_asset_id": asset["id"],
                    "output_name": "left.wav",
                    "operations": [exact_op((1000, 8000), end=(1500, 8000))],
                }
            )
            right = client.render(
                {
                    "primary_asset_id": asset["id"],
                    "output_name": "right.wav",
                    "operations": [exact_op((1500, 8000), end=(2000, 8000))],
                }
            )
            joined = first_channel_values(_fetch_output(server, left))
            joined += first_channel_values(_fetch_output(server, right))
            self.assertEqual(joined, list(range(1000, 2000)))
            self.assertEqual(
                left["output_asset"]["descriptor"]["properties"]["frame_count"],
                500,
                "left selection length",
            )
            self.assertEqual(
                right["output_asset"]["descriptor"]["properties"]["frame_count"],
                500,
                "right selection length",
            )
            for rendition in (left, right):
                status, check = client.request(
                    "POST",
                    f"/v1/renditions/{rendition['id']}/integrity-checks",
                    {},
                )
                self.assertEqual(status, 201)
                self.assertEqual(check["status"], "verified")

    def test_exact_and_equivalent_seconds_plan_agree_on_output(self) -> None:
        # 0.05 s * 8000 and 0.3 s * 8000 are integral, so an exact plan at
        # those fractions must reproduce the same bytes as the legacy plan.
        source = make_index_ramp_wav(8000, 4000)
        with ServerFixture() as (client, server):
            asset = client.upload("ramp.wav", source)
            exact_rendition = client.render(
                {
                    "primary_asset_id": asset["id"],
                    "output_name": "exact.wav",
                    "operations": [exact_op((1, 20), duration=(3, 10))],
                }
            )
            legacy_rendition = client.render(
                {
                    "primary_asset_id": asset["id"],
                    "output_name": "legacy.wav",
                    "operations": [
                        {
                            "type": "media_trim",
                            "start_seconds": 0.05,
                            "duration_seconds": 0.3,
                        }
                    ],
                }
            )
            self.assertNotEqual(
                exact_rendition["plan_digest"],
                legacy_rendition["plan_digest"],
                "the two coordinate systems must not share a plan digest",
            )
            self.assertEqual(
                _fetch_output(server, exact_rendition),
                _fetch_output(server, legacy_rendition),
                "integral exact times must match the legacy conversion",
            )

    def test_exact_defaults_would_disagree_for_split_seconds(self) -> None:
        # These legacy roundings agree: round(0.0004*8000)=3 and
        # round(0.2*8000)=1600. An exact plan must land on floor(3.2)=3 and
        # floor(1600)=1600, so lengths match while the digest differs.
        source = make_index_ramp_wav(8000, 4000)
        with ServerFixture() as (client, server):
            asset = client.upload("ramp.wav", source)
            rendition = client.render(
                {
                    "primary_asset_id": asset["id"],
                    "output_name": "exact.wav",
                    "operations": [exact_op((1, 2500), end=(1, 5))],
                }
            )
            descriptor = rendition["output_asset"]["descriptor"]
            self.assertEqual(
                descriptor["properties"]["frame_count"],
                1597,
                "floor end boundary length",
            )
            status, check = client.request(
                "POST",
                f"/v1/renditions/{rendition['id']}/integrity-checks",
                {},
            )
            self.assertEqual(status, 201)
            self.assertEqual(check["status"], "verified")

    def test_legacy_seconds_rendition_verifies_after_extension(self) -> None:
        source = make_sine_wav(frequency=330)
        with ServerFixture() as (client, _server):
            asset = client.upload("legacy.wav", source)
            rendition = client.render(
                {
                    "primary_asset_id": asset["id"],
                    "output_name": "legacy-output.wav",
                    "operations": [
                        {
                            "type": "media_trim",
                            "start_seconds": 0.05,
                            "duration_seconds": 0.3,
                        }
                    ],
                }
            )
            status, check = client.request(
                "POST",
                f"/v1/renditions/{rendition['id']}/integrity-checks",
                {},
            )
            self.assertEqual(status, 201)
            self.assertEqual(check["status"], "verified")


def _fetch_output(server: Any, rendition: dict[str, Any]) -> bytes:
    asset = server.application.repository.get_asset(
        rendition["output_asset"]["id"]
    )
    return server.application.content_files.read(asset.content_key)


def _rewrite_operations(
    server: Any,
    rendition_id: str,
    rewrite: Any,
) -> None:
    """Simulate a tampered persisted plan to probe replay policy."""

    def mutate(state: dict[str, Any]) -> None:
        state["renditions"][rendition_id]["operations"] = rewrite(
            state["renditions"][rendition_id]["operations"]
        )

    server.application.state_file.mutate(mutate)


class StoredPlanReplayTests(unittest.TestCase):
    def test_tampered_exact_plan_becomes_mismatch_not_verification(self) -> None:
        source = make_index_ramp_wav(8000, 4000)
        with ServerFixture() as (client, server):
            asset = client.upload("ramp.wav", source)
            rendition = client.render(
                {
                    "primary_asset_id": asset["id"],
                    "output_name": "seam.wav",
                    "operations": [exact_op((1, 4), end=(2, 4))],
                }
            )

            def strip_policy(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
                operations[0].pop("time_mode", None)
                operations[0].pop("boundary_rule", None)
                return operations

            _rewrite_operations(server, rendition["id"], strip_policy)
            status, body = client.request(
                "POST",
                f"/v1/renditions/{rendition['id']}/integrity-checks",
                {},
            )
            self.assertEqual(status, 409)
            audit = body["error"]["details"]["audit"]
            self.assertEqual(audit["status"], "mismatch")
            # The stored plan no longer matches its recorded digest, so replay
            # refuses to reinterpret it at all.
            self.assertEqual(audit["reason"], "stored_plan_drift")

    def test_tampered_rule_cannot_switch_rounding(self) -> None:
        source = make_index_ramp_wav(8000, 4000)
        with ServerFixture() as (client, server):
            asset = client.upload("ramp.wav", source)
            rendition = client.render(
                {
                    "primary_asset_id": asset["id"],
                    "output_name": "seam.wav",
                    "operations": [exact_op((1, 4), end=(2, 4))],
                }
            )

            def switch_rule(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
                operations[0]["boundary_rule"] = "round"
                return operations

            _rewrite_operations(server, rendition["id"], switch_rule)
            status, body = client.request(
                "POST",
                f"/v1/renditions/{rendition['id']}/integrity-checks",
                {},
            )
            self.assertEqual(status, 409)
            self.assertEqual(
                body["error"]["details"]["audit"]["reason"],
                "stored_plan_drift",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
