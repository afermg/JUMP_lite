from copy import deepcopy
import io
import json
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from botocore.exceptions import ClientError, EndpointConnectionError

from jump_full_compression.incidents import SourceReadError, retry_evidence
from jump_full_compression.pipeline import read_source, decode_stack
from jump_full_compression import cli
from jump_full_compression.missing_sources import (
    confirm_missing_sources,
    confirmed_missing,
)


class SourceReadEvidenceTests(unittest.TestCase):
    uri = "s3://cellpainting-gallery/public/source.tif"

    def test_default_reader_retains_original_clienterror_contract(self):
        error = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        client = mock.Mock()
        client.get_object.side_effect = error
        with mock.patch(
            "jump_full_compression.pipeline._s3_client", return_value=client
        ):
            with self.assertRaises(ClientError) as caught:
                read_source(self.uri, attempts=2, sleep=lambda _: None)
        self.assertIs(caught.exception, error)
        self.assertEqual(client.get_object.call_count, 2)

    def test_transport_failure_receipt_does_not_leak_endpoint_or_exception_message(
        self,
    ):
        error = EndpointConnectionError(
            endpoint_url="https://SECRET:PRIVATE@example.invalid/?X-Amz-Signature=PRIVATE"
        )
        client = mock.Mock()
        client.get_object.side_effect = error
        with mock.patch(
            "jump_full_compression.pipeline._s3_client", return_value=client
        ):
            with self.assertRaises(SourceReadError) as caught:
                read_source(
                    self.uri,
                    attempts=2,
                    sleep=lambda _: None,
                    collect_source_errors=True,
                )
        evidence = caught.exception.failure
        self.assertEqual(evidence["attempt_count"], 2)
        self.assertEqual(
            evidence["attempts"][-1]["error_code"], "EndpointConnectionError"
        )
        self.assertNotIn("PRIVATE", json.dumps(evidence))
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_unknown_or_malformed_s3_error_metadata_is_not_serialized(self):
        error = ClientError(
            {
                "Error": {"Code": "SECRET"},
                "ResponseMetadata": {
                    "HTTPStatusCode": "SECRET",
                    "RetryAttempts": "PRIVATE",
                    "HTTPHeaders": {"authorization": "PRIVATE"},
                },
            },
            "GetObject",
        )
        evidence = retry_evidence(error, 1, 0)
        self.assertEqual(evidence["error_code"], "OtherClientError")
        self.assertIsNone(evidence["http_status"])
        self.assertIsNone(evidence["sdk_retry_attempts"])
        self.assertNotIn("SECRET", json.dumps(evidence))
        self.assertNotIn("PRIVATE", json.dumps(evidence))

    def test_signed_private_source_uris_and_unbounded_retries_are_rejected(self):
        for uri in (
            self.uri + "?X-Amz-Signature=SECRET",
            self.uri + "#SECRET",
            "https://example.invalid/SECRET",
            "s3://user:SECRET@cellpainting-gallery/a",
        ):
            with (
                self.subTest(uri=uri),
                mock.patch("jump_full_compression.pipeline._s3_client") as client,
            ):
                with self.assertRaises(ValueError) as caught:
                    read_source(uri, collect_source_errors=True)
                self.assertNotIn("SECRET", str(caught.exception))
                client.assert_not_called()
        for attempts in (0, 9, True, -1):
            with self.assertRaises(ValueError):
                read_source(self.uri, attempts=attempts, collect_source_errors=True)

    def test_transient_recovery_and_success_observation_unchanged(self):
        client = mock.Mock()
        client.get_object.side_effect = [
            OSError("SECRET"),
            {
                "Body": io.BytesIO(b"abc"),
                "ContentLength": 3,
                "ETag": '"etag"',
            },
        ]
        with mock.patch(
            "jump_full_compression.pipeline._s3_client", return_value=client
        ):
            payload, observed = read_source(
                self.uri, sleep=lambda _: None, collect_source_errors=True
            )
        self.assertEqual(payload, b"abc")
        self.assertEqual(observed, {"uri": self.uri, "size": 3, "etag": "etag"})

    def test_truncated_reads_report_only_bounded_safe_evidence(self):
        def truncated(**_):
            return {"Body": io.BytesIO(b"a"), "ContentLength": 2}

        with mock.patch(
            "jump_full_compression.pipeline._s3_client",
            return_value=mock.Mock(get_object=truncated),
        ):
            with self.assertRaises(SourceReadError) as caught:
                read_source(
                    self.uri,
                    attempts=2,
                    sleep=lambda _: None,
                    collect_source_errors=True,
                )
        self.assertEqual(
            caught.exception.failure["attempts"][-1]["error_code"], "OSError"
        )

    def test_successful_all_zero_source_is_not_reclassified_as_missing(self):
        image = io.BytesIO()
        Image.fromarray(np.zeros((4, 4), dtype=np.uint16)).save(image, format="TIFF")
        payload = image.getvalue()

        def source(**_):
            return {
                "Body": io.BytesIO(payload),
                "ContentLength": len(payload),
                "ETag": '"etag"',
            }

        row = {"channels": ["DNA", "ER"], "urls": [self.uri, self.uri]}
        with mock.patch(
            "jump_full_compression.pipeline._s3_client",
            return_value=mock.Mock(get_object=source),
        ):
            stack, observations = decode_stack(row, False, collect_source_errors=True)
        self.assertEqual(stack.shape, (2, 4, 4))
        self.assertFalse(stack.any())
        self.assertEqual(len(observations), 2)

    def test_confirmed_missing_requires_consistent_get_retries_and_independent_head(
        self,
    ):
        missing = ClientError(
            {
                "Error": {"Code": "NoSuchKey"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObject",
        )
        failure = {
            "uri": self.uri,
            "channel": "DNA",
            "attempt_count": 5,
            "attempts": [
                retry_evidence(missing, i, min(8.0, 0.25 * 2**i) if i < 5 else 0.0)
                for i in range(1, 6)
            ],
        }
        for code in ("404", "NotFound", "NoSuchKey"):
            client = mock.Mock()
            client.head_object.side_effect = ClientError(
                {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": 404}},
                "HeadObject",
            )
            observed = confirm_missing_sources([failure], lambda: client)
            self.assertTrue(confirmed_missing(observed))
            client.head_object.assert_called_once_with(
                Bucket="cellpainting-gallery", Key="public/source.tif"
            )
        for code, status in (
            ("AccessDenied", 403),
            ("NoSuchBucket", 404),
            ("NoSuchVersion", 404),
            ("RequestTimeout", 408),
            ("NoSuchKey", 403),
        ):
            with self.subTest(code=code, status=status):
                changed = deepcopy(failure)
                changed["attempts"][2].update(error_code=code, http_status=status)
                client = mock.Mock()
                self.assertFalse(
                    confirmed_missing(
                        confirm_missing_sources([changed], lambda: client)
                    )
                )
                client.head_object.assert_not_called()
        for head_error in (
            ClientError(
                {
                    "Error": {"Code": "NoSuchBucket"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "HeadObject",
            ),
            EndpointConnectionError(endpoint_url="https://SECRET.example.invalid"),
        ):
            client = mock.Mock()
            client.head_object.side_effect = head_error
            observed = confirm_missing_sources([failure], lambda: client)
            self.assertFalse(confirmed_missing(observed))
            self.assertNotIn("SECRET", json.dumps(observed))
        for count in (0, 1, 4, 9):
            changed = {**failure, "attempt_count": count}
            client = mock.Mock()
            self.assertFalse(
                confirmed_missing(confirm_missing_sources([changed], lambda: client))
            )
            client.head_object.assert_not_called()

    def test_cli_routes_exact_source_acknowledgement(self):
        arguments = ["production-acknowledge-source-incidents"]
        for flag in (
            "manifest",
            "audit-report",
            "build-report",
            "exclusion-policy",
            "damaged-objects",
            "damaged-sites",
            "qc-plates",
            "output-root",
            "state-root",
            "production-id",
            "inventory-digest",
            "manifest-sha256",
            "audit-sha256",
            "build-report-sha256",
            "exclusion-policy-sha256",
            "damaged-objects-sha256",
            "damaged-sites-sha256",
            "qc-plates-sha256",
        ):
            arguments += ["--" + flag, "unused"]
        arguments += [
            "--manifest-size",
            "1",
            "--site-count",
            "1",
            "--expected-count",
            "2",
        ]
        with (
            mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            cli.parser().parse_args(arguments)
        arguments += ["--expected-list-sha256", "a" * 64, "--apply"]
        parsed = cli.parser().parse_args(arguments)
        self.assertEqual(parsed.missing_source_policy, "record-and-skip")
        self.assertEqual(parsed.max_missing_fraction, 0.05)
        with (
            mock.patch.object(cli, "_production_config", return_value="config"),
            mock.patch.object(
                cli, "acknowledge_production_source_incidents", return_value={}
            ) as acknowledge,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(cli.main(arguments), 0)
        acknowledge.assert_called_once_with("config", 2, "a" * 64, True)
