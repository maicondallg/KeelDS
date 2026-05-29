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
EXPECTED_PROCESS_SHA256 = '183cd18ae95aada3ea6f4410a60e66a53ac00aba2c447b51c9d25059a6798977'
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


def print_installed_binary_hashes() -> None:
    print('\nINSTALLED BINARY HASHES')
    binary_hash_script = WORK_DIR / '.docker_binary_hashes.py'
    binary_hash_script.write_text(
        "from pathlib import Path\n"
        "import hashlib\n"
        "import mdlp\n"
        "import numpy as np\n"
        "\n"
        "def sha(path):\n"
        "    return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n"
        "\n"
        "mdlp_dir = Path(mdlp.__file__).parent\n"
        "print('mdlp_init', mdlp.__file__)\n"
        "print('mdlp_init_sha256', sha(mdlp.__file__))\n"
        "print('mdlp_binary_files', [str(p) for p in sorted(mdlp_dir.glob('_mdlp*.so'))])\n"
        "for path in sorted(mdlp_dir.glob('_mdlp*.so')):\n"
        "    print('mdlp_binary_sha256', path.name, sha(path), 'size', path.stat().st_size)\n"
        "print('numpy_file', np.__file__)\n"
        "print('numpy_version', np.__version__)\n"
        "print('numpy_runtime_start')\n"
        "if hasattr(np, 'show_runtime'):\n"
        "    np.show_runtime()\n"
        "else:\n"
        "    np.__config__.show()\n"
        "print('numpy_runtime_end')\n"
    )
    run(['uv', 'run', 'python', str(binary_hash_script)], cwd=WORK_DIR)


def print_mdlp_stage_diagnostics() -> None:
    print('\nMDLP STAGE DIAGNOSTICS')
    diagnostics_script = WORK_DIR / '.docker_mdlp_stage_diagnostics.py'
    diagnostics_script.write_text(
        "from __future__ import annotations\n"
        "\n"
        "import hashlib\n"
        "import io\n"
        "import numpy as np\n"
        "from mdlp.discretization import MDLP\n"
        "from sklearn.preprocessing import LabelEncoder\n"
        "from process import Dataset\n"
        "\n"
        "def array_sha(array):\n"
        "    bio = io.BytesIO()\n"
        "    np.save(bio, np.asarray(array), allow_pickle=False)\n"
        "    return hashlib.sha256(bio.getvalue()).hexdigest()\n"
        "\n"
        "def cut_points_sha(cut_points):\n"
        "    h = hashlib.sha256()\n"
        "    for cp in cut_points:\n"
        "        if cp is None:\n"
        "            h.update(b'NONE')\n"
        "            continue\n"
        "        arr = np.asarray(cp, dtype=np.float64)\n"
        "        h.update(np.asarray([arr.size], dtype=np.int64).tobytes())\n"
        "        h.update(np.ascontiguousarray(arr).view(np.uint8).tobytes())\n"
        "    return h.hexdigest()\n"
        "\n"
        "ds = Dataset('australian', 'keel_ds/data/balanced/raw/australian.dat', balance='balanced')\n"
        "ds.set_attributes_to_discretize()\n"
        "attributes = ds.get_attributes_to_discretize()\n"
        "print('attributes_to_discretize', attributes)\n"
        "ds.split(k_folds=10)\n"
        "for i, fold in enumerate(ds.data_folds):\n"
        "    x_train, y_train, x_test, y_test = fold\n"
        "    print('fold', i)\n"
        "    print('  pre_x_train_sha', array_sha(x_train))\n"
        "    print('  pre_y_train_sha', array_sha(y_train))\n"
        "    print('  pre_x_test_sha', array_sha(x_test))\n"
        "    print('  pre_y_test_sha', array_sha(y_test))\n"
        "    le = LabelEncoder()\n"
        "    y_train_enc = le.fit_transform(y_train)\n"
        "    y_test_enc = le.transform(y_test)\n"
        "    disct = MDLP(random_state=ds.random_state, min_depth=1)\n"
        "    x_train_discr = x_train[:, attributes]\n"
        "    x_test_discr = x_test[:, attributes]\n"
        "    x_train_disc = disct.fit_transform(x_train_discr, y_train_enc)\n"
        "    x_test_disc = disct.transform(x_test_discr)\n"
        "    print('  y_train_encoded_sha', array_sha(y_train_enc))\n"
        "    print('  y_test_encoded_sha', array_sha(y_test_enc))\n"
        "    print('  cut_points_sha', cut_points_sha(disct.cut_points_))\n"
        "    print('  cut_points_lengths', [None if cp is None else len(cp) for cp in disct.cut_points_])\n"
        "    if i == 0:\n"
        "        print('  fold0_cut_points_repr', [None if cp is None else np.asarray(cp).tolist() for cp in disct.cut_points_])\n"
        "    print('  mdlp_x_train_disc_sha', array_sha(x_train_disc))\n"
        "    print('  mdlp_x_test_disc_sha', array_sha(x_test_disc))\n"
    )
    run(['uv', 'run', 'python', str(diagnostics_script)], cwd=WORK_DIR)


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
    print_installed_binary_hashes()
    print_mdlp_stage_diagnostics()

    if generated_path.exists():
        generated_path.unlink()

    run(['uv', 'run', 'python', 'process.py'], cwd=WORK_DIR)

    ok = compare_npz(generated_path, reference_path)
    print('\nRESULT', 'MATCH' if ok else 'DIFFERENT')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
