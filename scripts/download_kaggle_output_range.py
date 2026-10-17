"""Resume a large private Kaggle kernel output with verified HTTP byte ranges.

The Kaggle CLI currently materializes each response in memory; this helper keeps
each request bounded and can resume after an interrupted transfer. It uses the
same authenticated output listing as the CLI and never prints signed URLs.
"""
import argparse
from pathlib import Path
import re
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest


def output_url(kernel, remote_name):
    owner, slug = kernel.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    request = ApiListKernelSessionOutputRequest()
    request.user_name = owner
    request.kernel_slug = slug
    api._set_paging(request, 200, None)
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
    matches = [item.url for item in response.files or [] if item.file_name == remote_name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one output named {remote_name!r}, found {len(matches)}")
    return matches[0]


def transfer(kernel, remote_name, destination, chunk_size=32 * 1024 * 1024):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    url = output_url(kernel, remote_name)
    with requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60) as probe:
        probe.raise_for_status()
        match = re.fullmatch(r"bytes 0-0/(\d+)", probe.headers.get("Content-Range", ""))
        if probe.status_code != 206 or not match:
            raise RuntimeError("Kaggle output server did not honor the range probe")
        total = int(match.group(1))
    if destination.is_file() and destination.stat().st_size == total:
        print(f"Already complete: {destination} ({total} bytes)", flush=True)
        return
    if part.is_file() and part.stat().st_size > total:
        raise ValueError(f"Partial download exceeds remote size: {part}")
    offset = part.stat().st_size if part.is_file() else 0
    while offset < total:
        end = min(offset + chunk_size, total) - 1
        for attempt in range(5):
            try:
                with requests.get(url, headers={"Range": f"bytes={offset}-{end}"},
                                  timeout=(30, 90)) as response:
                    response.raise_for_status()
                    expected = f"bytes {offset}-{end}/{total}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Unexpected Content-Range at offset {offset}")
                    data = response.content
                if len(data) != end - offset + 1:
                    raise IOError(f"Short range at offset {offset}: {len(data)} bytes")
                with part.open("ab") as handle:
                    handle.write(data)
                offset = end + 1
                print(f"{offset}/{total} bytes", flush=True)
                break
            except (requests.RequestException, IOError) as exc:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 16))
                url = output_url(kernel, remote_name)
    part.replace(destination)
    print(f"Complete: {destination} ({total} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="owner/kernel-slug")
    parser.add_argument("remote_name", help="Exact output path from Kaggle listing")
    parser.add_argument("destination")
    args = parser.parse_args()
    transfer(args.kernel, args.remote_name, args.destination)
