#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import sys

import requests


def pause(host, port, timeout):
    url = f"http://{host}:{port}/fault_tolerance/apply"
    payload = {
        "fault_tolerance_instruction": "pause",
        "fault_tolerance_timeout": timeout,
        "fault_tolerance_params": {"soft_pause": False},
    }

    headers = {"Content-Type": "application/json"}

    print(f"Sending pause request to {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            print("Pause request successful!")
            return True
        else:
            print("Pause request failed!")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False


def scale(host, port, timeout, exclude_dp_ranks):
    url = f"http://{host}:{port}/fault_tolerance/apply"
    payload = {
        "fault_tolerance_instruction": "descale",
        "fault_tolerance_timeout": timeout,
        "fault_tolerance_params": {"exclude_dp_ranks": exclude_dp_ranks},
    }
    headers = {"Content-Type": "application/json"}

    print(f"Sending scale request to {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            print("Scale down request successful!")
            return True
        else:
            print("Scale down request failed!")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test scale down functionality")
    parser.add_argument("--host", default="localhost", help="API server host")
    parser.add_argument("--port", type=int, default=8006, help="API server port")
    parser.add_argument(
        "--timeout", type=int, default=30, help="Fault recovery timeout"
    )
    parser.add_argument(
        "--exclude-dp-ranks",
        type=int,
        nargs="*",
        default=[0],
        help="The dp_ranks that will be excluded (scaled down)",
    )

    args = parser.parse_args()
    success_pause = pause(args.host, args.port, args.timeout)
    if not success_pause:
        sys.exit(1)
    success_scale = scale(args.host, args.port, args.timeout, args.exclude_dp_ranks)
    sys.exit(0 if success_scale else 1)


if __name__ == "__main__":
    main()
