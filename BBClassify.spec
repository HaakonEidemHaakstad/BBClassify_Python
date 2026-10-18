# -*- mode: python ; coding: utf-8 -*-
# Build the console program with:  pyinstaller BBClassify.spec
#
# This builds a one-folder program (dist/BBClassify/BBClassify.exe plus an
# _internal folder) rather than a single .exe. A single-file .exe unpacks
# itself into a temporary folder every time it starts, which takes seconds
# (more with antivirus scanning or OneDrive); the one-folder program starts
# almost immediately. Distribute the whole dist/BBClassify folder, e.g. zipped.

a = Analysis(
    ['bbclassify/__main__.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['scipy', 'matplotlib', 'pandas', 'tkinter', 'pytest', 'unittest', 'pydoc', 'email', 'http', 'xml'],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BBClassify',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='BBClassify',
)
