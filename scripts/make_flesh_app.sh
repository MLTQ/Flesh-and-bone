#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=src python scripts/export_flesh_runtime.py
swift build -c release

APP="dist/FleshAndBoneLab.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp ".build/release/FleshAndBoneLab" "$APP/Contents/MacOS/FleshAndBoneLab"
cp runtime/Assets/*.fnb runtime/Assets/*.fnm runtime/Assets/manifest.json \
  "$APP/Contents/Resources/"
cp runtime/Info.plist "$APP/Contents/Info.plist"

codesign --force --sign - "$APP" 2>/dev/null || true
echo "Built $APP"
