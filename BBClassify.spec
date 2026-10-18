# -*- mode: python ; coding: utf-8 -*-
# Build a standalone console program with:  pyinstaller BBClassify.spec
# The executable is written to dist/BBClassify(.exe). Put control cards and
# data files next to it, or give full paths when prompted.

a = Analysis(
    ['bbclassify/__main__.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=['scipy.special._cdflib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pandas', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BBClassify',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
