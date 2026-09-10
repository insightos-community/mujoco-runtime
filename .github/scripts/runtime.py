import hashlib, importlib.util, json, os, re, shutil, subprocess, sys, tempfile, zipfile
from dataclasses import replace
from email.parser import BytesParser
from pathlib import Path
import yaml
HERE=Path.cwd()/'.output'; (HERE/'.build').mkdir(exist_ok=True)
POINTER=b'version https://git-lfs.github.com/spec/v1'
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024*1024): h.update(block)
    return h.hexdigest()

def run(args, cwd=None, env=None):
    print('构建:', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)

def copy_file(source, dest):
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'需要普通产物文件: {source}')
    with source.open('rb') as f:
        if f.read(128).startswith(POINTER):
            raise ValueError(f'LFS 指针未替换为真实产物: {source}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    dest.chmod(0o755 if os.access(source, os.X_OK) else 0o644)

def copy_tree(source, dest):
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'拒绝复制符号链接: {path}')
        if path.is_file():
            copy_file(path, dest/path.relative_to(source))

def wheel_metadata(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip(): raise ValueError(f'损坏的 Wheel: {path}')
        names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names) != 1: raise ValueError(f'Wheel 缺少唯一元数据: {path}')
        metadata = BytesParser().parsebytes(archive.read(names[0]))
        return metadata['Name'], metadata['Version']

def build_runtime(workspace, output, python):
    """Reuse the native Runtime Pack schema; replace source-path locks with wheel-only pins."""
    runtime = Path.cwd()
    spec = importlib.util.spec_from_file_location('semantic_runtime_pack_builder', runtime/'tools/build_runtime_pack.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    profile = replace(module.profile_table()['native-mujoco'], python=python)
    with tempfile.TemporaryDirectory(prefix='native-pack-', dir=HERE/'.build') as tmp:
        stage = Path(tmp)
        wheels = module.build_wheels(profile, stage, '0.4.0-dev.0')
        house = stage/'wheelhouse'
        house.mkdir()
        run(['uv', 'build', '--wheel', '--project', runtime/'packages/mujoco-visuals', '--out-dir', house])
        # uv.lock is the source of dependency versions, but local path entries cannot be shipped.
        exported = subprocess.check_output(['uv', 'export', '--project', str(runtime), '--frozen', '--no-dev',
            '--no-hashes', '--no-emit-project', '--no-emit-package', 'semantic-mujoco-visuals'], text=True)
        requirements = stage/'build-requirements.txt'
        lines = [line for line in exported.splitlines() if line.strip() and not line.lstrip().startswith('#')]
        if any(not re.match(r'^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==', line) for line in lines):
            raise ValueError('Runtime 依赖锁包含未固定版本或源码路径')
        requirements.write_text('\n'.join(lines)+'\n')
        run(['uv', 'run', '--isolated', '--no-project', '--python', python, '--with', 'pip',
             'python', '-m', 'pip', 'download', '--only-binary=:all:', '--dest', house, '-r', requirements])
        dependencies = sorted(house.glob('*.whl'))
        lock = stage/'locks/requirements.lock'
        lock.parent.mkdir()
        lock.write_text('\n'.join(f'{name}=={version}' for name, version in sorted(wheel_metadata(w) for w in dependencies))+'\n')
        catalog, resources, smoke, verification = module.copy_metadata(profile, '0.4.0-dev.0', stage)
        # Runtime repo's authoring fixtures can lag behind the running Framework schema.
        # Ship the current Framework's validated native scene documents, not its live .output.
        (stage/'catalog').rename(stage/'legacy-catalog')
        catalog = stage/'catalog/catalog.yaml'
        catalog.parent.mkdir()
        copy_tree(workspace/'semantic-framework/configs/scenes.d/authoring/depalletizing-r1pro',
                  stage/'catalog/authoring/depalletizing-r1pro')
        doc = yaml.safe_load((workspace/'semantic-framework/configs/scenes.d/mujoco-platforms.yaml').read_text())
        doc['entries'] = [e for e in doc['entries'] if e['compatible_runtime_profile'] == 'native-mujoco']
        resources = sorted(p for p in (stage/'catalog').rglob('*') if p.is_file())
        # Current asset catalog is a development snapshot with this exact spelling.
        asset_version = json.loads((workspace/'semantic-scene/mujoco-asset/asset-catalog.v1.json').read_text())['catalog_version']
        doc['catalog_version'] = asset_version
        for entry in doc['entries']:
            for version in entry['versions']:
                if 'authoring' in version:
                    version['authoring']['asset_catalog_version'] = asset_version
        catalog.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))
        record = lambda p: module.file_record(stage, p)
        manifest = dict(schema_version=1, pack_id=profile.pack_id, pack_version='0.4.0-dev.0',
            profile=profile.profile, runner=profile.runner, python_version=python, endpoint=profile.endpoint,
            hardware_requirements=profile.hardware_requirements, requirements_lock=record(lock),
            wheels=list(map(record, wheels)), wheelhouse=list(map(record, dependencies)),
            scene_catalog=record(catalog), scene_resources=list(map(record, resources)),
            licenses=list(map(record, sorted((stage/'licenses').iterdir()))), verification_files=[record(verification)],
            smoke_scene_key='palletizing_depalletizing_tote_v1', smoke_request=record(smoke),
            content_requirements=profile.content_requirements)
        (stage/'runtime-pack.yaml').write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False))
        output.parent.mkdir(parents=True, exist_ok=True)
        run(['tar', '--zstd', '-cf', output, '-C', stage, 'runtime-pack.yaml', 'wheels', 'wheelhouse',
             'locks', 'catalog', 'licenses', 'verification', 'smoke'])
build_runtime(Path.cwd().parent/'dependencies',Path.cwd()/'.output/payload/native-mujoco-0.4.0-dev.0.runtime.tar.zst','3.10.19')
records=[]
for local in ('semantic-framework','semantic-scene/mujoco-asset'):
 p=Path.cwd().parent/'dependencies'/local
 records.append(dict(component=local,source_commit=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()))
(Path.cwd()/'.output/dependencies.json').write_text(json.dumps(records,indent=2)+'\n')
