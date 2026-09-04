import argparse
import os
import subprocess
import sys
import urllib.parse


BASEURL = "https://af01p-ba.devtools.intel.com/artifactory/aipc_releases-ba-local/"
ARTIFACTORY_HOST = urllib.parse.urlparse(BASEURL).hostname


def get_all_files(path):
    collected = []
    for root, _, files in os.walk(path):
        for file_name in files:
            collected.append(os.path.join(root, file_name))
    return collected


def ensure_no_proxy(hostname):
    if not hostname:
        return

    for env_name in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(env_name, "").strip()
        entries = [item.strip() for item in current.split(",") if item.strip()]
        if hostname not in entries:
            entries.append(hostname)
        os.environ[env_name] = ",".join(entries)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="target folder")
    parser.add_argument("--path", type=str, default="local-ci", help="target path under aipc_releases-ba-local")
    parser.add_argument("--file", type=str, help="target file")
    parser.add_argument("--cred", type=str, default="", help="user credential for upload")
    return parser.parse_args()


def resolve_credential(args):
    if args.cred.strip():
        return args.cred.strip()

    raw_credential = os.environ.get("UPLOAD_ARTIFACTORY_CREDENTIALS", "").strip()
    if raw_credential:
        return raw_credential

    username = os.environ.get("ARTIFACTORY_USERNAME", "").strip()
    password = os.environ.get("ARTIFACTORY_PASSWORD", "").strip()
    if username or password:
        return f"{username}:{password}"

    return ""


def main():
    args = parse_arguments()
    credential = resolve_credential(args)
    if not credential:
        print("Missing Artifactory credential: use --cred, UPLOAD_ARTIFACTORY_CREDENTIALS, or ARTIFACTORY_USERNAME/ARTIFACTORY_PASSWORD", file=sys.stderr)
        return 2

    if args.folder:
        folder = os.path.abspath(args.folder)
        all_files = get_all_files(folder)
    elif args.file:
        folder = ""
        all_files = [os.path.abspath(args.file)]
    else:
        print("please make sure you use folder argument or file argument", file=sys.stderr)
        return 2

    ensure_no_proxy(ARTIFACTORY_HOST)
    upload_root = args.path.strip().strip("/")
    upload_base = BASEURL if not upload_root else f"{BASEURL}{upload_root}/"

    for each_file in all_files:
        if args.folder:
            relative_path = os.path.relpath(each_file, folder)
        else:
            relative_path = os.path.basename(each_file)

        relative_path = relative_path.replace(os.sep, "/")
        url = f"{upload_base}{relative_path};retention.days=2800"
        print(url)

        cmd = ["curl", "-k", "-u", credential, "-T", each_file, url]
        print("curl -k -u <redacted> -T " + each_file + ' "' + url + '"')
        result = subprocess.run(cmd, check=False)
        print(result.returncode)
        if result.returncode != 0:
            return result.returncode

    return 0


if __name__ == '__main__':
    raise SystemExit(main())