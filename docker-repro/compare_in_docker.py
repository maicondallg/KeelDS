from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

SOURCE_DIR = Path('/src').resolve()
WORK_DIR = Path('/work/KeelDS').resolve()
REFERENCE_RELATIVE_PATH = Path('keel_ds/data/balanced/processed/australian (Cópia).npz')
GENERATED_RELATIVE_PATH = Path('keel_ds/data/balanced/processed/australian.npz')
RAW_RELATIVE_PATH = Path('keel_ds/data/balanced/raw/australian.dat')
EXPECTED_REFERENCE_SHA256 = '88ed71029877c6ced3a9243c5475a1895358353a5bec16ad645552a628b58977'
EXPECTED_RAW_SHA256 = 'ccc64bf31674bc1c282e11f9ba2bb3c5777ca15f03e3d96142ed0817bf7fedce'
EXPECTED_PROCESS_SHA256 = '68bf1f4040eff44d8aa68da10512633ed2a01d3b6c4d710386ed0dc3303ecd2b'
EXPECTED_PYPROJECT_SHA256 = 'dd9a3c2d0deb45d9a80590a7ad2c23a75acc4289036e70edc0c507e60d978b2e'
EXPECTED_UV_LOCK_SHA256 = 'd9a9214149b03994295938309aae3713a4ee5b81338db60e6f46c098a7178990'


def run(command: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def print_expected_hash(label: str, path: Path, expected: str) -> bool:
    if not path.exists():
        print(label, 'MISSING')
        print(f'expected_{label}', expected)
        return False

    actual = sha256(path)
    ok = actual == expected
    print(label, actual)
    print(f'expected_{label}', expected)
    print(f'{label}_matches_expected', ok)
    return ok


def print_source_git_state() -> None:
    print('\nSOURCE GIT STATE')
    subprocess.run(['git', 'config', '--global', '--add', 'safe.directory', str(SOURCE_DIR)], check=False)
    for command in (
        ['git', '-C', str(SOURCE_DIR), 'rev-parse', 'HEAD'],
        ['git', '-C', str(SOURCE_DIR), 'status', '--short'],
    ):
        completed = subprocess.run(command, text=True, capture_output=True)
        print(f"$ {' '.join(command)}")
        if completed.stdout:
            print(completed.stdout.rstrip())
        if completed.stderr:
            print(completed.stderr.rstrip())
        print('exit_code', completed.returncode)


def copy_project() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError('Expected the repository to be mounted at /src')

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    ignore = shutil.ignore_patterns(
        '.git',
        '.venv',
        'venv',
        'build',
        'dist',
        '*.egg-info',
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        '.ruff_cache',
    )
    shutil.copytree(SOURCE_DIR, WORK_DIR, ignore=ignore)


def print_environment() -> None:
    print('platform_system', platform.system())
    print('platform_machine', platform.machine())
    print('platform_platform', platform.platform())
    run(['python', '--version'])
    run(['uname', '-a'])
    run(['gcc', '--version'])
    run(['g++', '--version'])
    run(['uv', '--version'])


def compare_npz(generated_path: Path, reference_path: Path) -> bool:
    print('\nFILE LEVEL')
    print('generated_path', generated_path)
    print('reference_path', reference_path)
    print('generated_size', generated_path.stat().st_size)
    print('reference_size', reference_path.stat().st_size)
    print('generated_sha256', sha256(generated_path))
    print('reference_sha256', sha256(reference_path))
    file_bytes_identical = generated_path.read_bytes() == reference_path.read_bytes()
    print('file_bytes_identical', file_bytes_identical)

    print('\nZIP MEMBER LEVEL')
    with zipfile.ZipFile(generated_path) as zg, zipfile.ZipFile(reference_path) as zr:
        generated_infos = zg.infolist()
        reference_infos = zr.infolist()
        same_member_names = [i.filename for i in generated_infos] == [i.filename for i in reference_infos]
        same_member_count = len(generated_infos) == len(reference_infos)
        same_payloads = same_member_names and all(
            zg.read(gi.filename) == zr.read(ri.filename)
            for gi, ri in zip(generated_infos, reference_infos)
        )
        same_metadata = same_member_names and all(
            gi.compress_type == ri.compress_type
            and gi.file_size == ri.file_size
            and gi.compress_size == ri.compress_size
            and gi.CRC == ri.CRC
            and gi.date_time == ri.date_time
            and gi.create_system == ri.create_system
            for gi, ri in zip(generated_infos, reference_infos)
        )
        print('same_member_names', same_member_names)
        print('same_member_count', len(generated_infos), len(reference_infos), same_member_count)
        print('all_member_payloads_identical', same_payloads)
        print('all_member_metadata_identical', same_metadata)

    print('\nNUMPY ARRAY LEVEL')
    array_compare_script = WORK_DIR / '.docker_compare_arrays.py'
    array_compare_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "import numpy as np\n"
        "\n"
        "generated_path = Path(sys.argv[1])\n"
        "reference_path = Path(sys.argv[2])\n"
        "generated = np.load(generated_path)\n"
        "reference = np.load(reference_path)\n"
        "same_keys_order = generated.files == reference.files\n"
        "same_arrays = same_keys_order\n"
        "for key in generated.files:\n"
        "    ga = generated[key]\n"
        "    ra = reference[key]\n"
        "    ok = ga.shape == ra.shape and ga.dtype == ra.dtype and np.array_equal(ga, ra)\n"
        "    same_arrays = same_arrays and ok\n"
        "    if not ok:\n"
        "        print('DIFF_ARRAY', key)\n"
        "        print('  generated shape/dtype', ga.shape, ga.dtype)\n"
        "        print('  reference shape/dtype', ra.shape, ra.dtype)\n"
        "        if ga.shape == ra.shape:\n"
        "            diff = ga != ra\n"
        "            print('  num_diff', int(np.count_nonzero(diff)))\n"
        "            if np.any(diff):\n"
        "                idx = np.argwhere(diff)[0]\n"
        "                print('  first_diff_idx', idx.tolist())\n"
        "                print('  generated_value', ga[tuple(idx)])\n"
        "                print('  reference_value', ra[tuple(idx)])\n"
        "print('same_keys_order', same_keys_order)\n"
        "print('all_arrays_equal_shape_dtype_values', same_arrays)\n"
        "raise SystemExit(0 if same_arrays else 1)\n"
    )
    array_compare = subprocess.run(
        ['uv', 'run', 'python', str(array_compare_script), str(generated_path), str(reference_path)],
        cwd=WORK_DIR,
    )
    same_arrays = array_compare.returncode == 0

    return file_bytes_identical and same_payloads and same_metadata and same_arrays


def main() -> int:
    print_environment()
    print_source_git_state()
    copy_project()

    reference_path = WORK_DIR / REFERENCE_RELATIVE_PATH
    generated_path = WORK_DIR / GENERATED_RELATIVE_PATH
    raw_path = WORK_DIR / RAW_RELATIVE_PATH
    process_path = WORK_DIR / 'process.py'
    pyproject_path = WORK_DIR / 'pyproject.toml'
    uv_lock_path = WORK_DIR / 'uv.lock'

    if not reference_path.exists():
        print(f'Missing reference file: {reference_path}', file=sys.stderr)
        print('Make sure australian (Cópia).npz exists before running Docker comparison.', file=sys.stderr)
        return 2

    print('\nPROJECT HASHES')
    print_expected_hash('process_sha256', process_path, EXPECTED_PROCESS_SHA256)
    print_expected_hash('pyproject_sha256', pyproject_path, EXPECTED_PYPROJECT_SHA256)
    print_expected_hash('uv_lock_sha256', uv_lock_path, EXPECTED_UV_LOCK_SHA256)

    print('\nINPUT HASHES')
    print_expected_hash('raw_sha256', raw_path, EXPECTED_RAW_SHA256)
    print_expected_hash('reference_sha256', reference_path, EXPECTED_REFERENCE_SHA256)

    run(['uv', 'sync', '--locked'], cwd=WORK_DIR)

    print('\nRESOLVED PYTHON ENV')
    run([
        'uv', 'run', 'python', '-c',
        "import sys,numpy,pandas,sklearn,mdlp; "
        "print('python', sys.version); "
        "print('numpy', numpy.__version__); "
        "print('pandas', pandas.__version__); "
        "print('sklearn', sklearn.__version__); "
        "print('mdlp', mdlp.__file__)"
    ], cwd=WORK_DIR)

    if generated_path.exists():
        generated_path.unlink()

    run(['uv', 'run', 'python', 'process.py'], cwd=WORK_DIR)

    ok = compare_npz(generated_path, reference_path)
    print('\nRESULT', 'MATCH' if ok else 'DIFFERENT')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
