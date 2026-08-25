"""Upload every useful file from the T7 model-analysis backup to patient FTP folders."""

from __future__ import annotations

import argparse
import json
import os
import re
from ftplib import FTP, error_perm
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_SOURCE = Path(
    "/Volumes/T7/TAF KUTATÁS 2024/éles/modellanalízis backup mentés"
)
IGNORED_NAMES = {".DS_Store"}
MANIFEST_FILENAME = ".model-analysis-manifest.json"


def patient_taj(directory_name):
    digits = "".join(re.findall(r"\d", directory_name))
    return digits[-9:] if len(digits) >= 9 else None


def uploadable_files(patient_directory):
    return sorted(
        path
        for path in patient_directory.rglob("*")
        if path.is_file()
        and path.name not in IGNORED_NAMES
        and not path.name.startswith("._")
    )


def flattened_name(patient_directory, path):
    return "__".join(path.relative_to(patient_directory).parts)


def ensure_directory(ftp, parts):
    for part in parts:
        try:
            ftp.cwd(part)
        except error_perm:
            ftp.mkd(part)
            ftp.cwd(part)


def remote_size(ftp, filename):
    try:
        ftp.voidcmd("TYPE I")
        return ftp.size(filename)
    except error_perm:
        return None


def upload_file(ftp, local_path, remote_name):
    expected_size = local_path.stat().st_size
    if remote_size(ftp, remote_name) == expected_size:
        return "skipped"
    temporary_name = f".{remote_name}.uploading"
    with local_path.open("rb") as source:
        ftp.storbinary(f"STOR {temporary_name}", source, blocksize=1024 * 1024)
    if remote_size(ftp, remote_name) is not None:
        ftp.delete(remote_name)
    ftp.rename(temporary_name, remote_name)
    return "uploaded"


def write_manifest(ftp, patient_directory, files):
    payload = {
        "files": [
            {
                "name": flattened_name(patient_directory, path),
                "size": path.stat().st_size,
            }
            for path in files
        ]
    }
    temporary_name = f"{MANIFEST_FILENAME}.uploading"
    content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    ftp.storbinary(f"STOR {temporary_name}", BytesIO(content))
    try:
        ftp.delete(MANIFEST_FILENAME)
    except error_perm:
        pass
    ftp.rename(temporary_name, MANIFEST_FILENAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    load_dotenv(".env")
    host = os.getenv("NAS_HOST")
    user = os.getenv("NAS_USER")
    password = os.getenv("NAS_PASS")
    nas_root = os.getenv("NAS_DIR")
    patients_folder = os.getenv("NAS_PATIENTS_DIR", "patients")
    if not all((host, user, password, nas_root)):
        raise RuntimeError("Incomplete NAS FTP configuration.")
    if not args.source.is_dir():
        raise RuntimeError(f"Source directory is unavailable: {args.source}")

    patients = []
    for directory in sorted(args.source.iterdir()):
        taj = patient_taj(directory.name) if directory.is_dir() else None
        if taj:
            patients.append((taj, directory, uploadable_files(directory)))

    uploaded = skipped = uploaded_bytes = 0
    with FTP(host, timeout=60) as ftp:
        ftp.login(user, password)
        ftp.cwd(nas_root)
        nas_root_path = ftp.pwd()
        for index, (taj, patient_directory, files) in enumerate(patients, start=1):
            ftp.cwd(nas_root_path)
            ensure_directory(ftp, (patients_folder, taj, "model-analysis"))
            for path in files:
                remote_name = flattened_name(patient_directory, path)
                result = upload_file(ftp, path, remote_name)
                if result == "uploaded":
                    uploaded += 1
                    uploaded_bytes += path.stat().st_size
                else:
                    skipped += 1
            write_manifest(ftp, patient_directory, files)
            print(
                f"patient {index}/{len(patients)}: {len(files)} files; "
                f"uploaded={uploaded}, skipped={skipped}",
                flush=True,
            )

    print(
        f"complete: patients={len(patients)}, uploaded={uploaded}, "
        f"skipped={skipped}, uploaded_bytes={uploaded_bytes}"
    )


if __name__ == "__main__":
    main()
