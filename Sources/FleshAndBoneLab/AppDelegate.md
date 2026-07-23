# `AppDelegate.swift`

## Purpose

Compose the ordinary resizable laboratory window from one Metal organism view
and one explicit control/metrics panel.

## Components

### `AppDelegate`

- **Does:** Resolves Metal/assets, creates the view graph, constrains a 330 px
  control rail, and surfaces startup errors.
- **Interacts with:** `RuntimeAssets`, `FleshMetalView`, and `ControlPanel`.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `main.swift` | default app lifecycle uses `AppDelegate` | Class/lifecycle |
| app bundle | generated assets live in `Contents/Resources` | Asset resolution |
| user | closing the lab terminates it | Termination behavior |
